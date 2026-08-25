################################################################################
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
"""E2E tests for internal sub-agents on a real Flink job.

A child agent registered directly as an ``AGENT`` resource exercises the whole
operator round-trip: the Python-compiled child plan is materialized into a Java
``InternalSubagentSetup``, the call event is sent on the mailbox thread, the
mailbox is released while the caller awaits, and the child's actions are
dispatched and their output accumulated into the caller's ``Result``.

Unlike the Java caller path, this does not need JDK 21: a Python coroutine
releases the mailbox by yielding at ``await``.

SKIPPED: a Python child agent is not scope-aware yet. The Python side binds one
``FlinkRunnerContext`` at operator start, wrapping the root Java context and the
root plan, and ``PythonActionTask`` passes only the event and key to pemja — the
child context the operator creates never reaches Python. So a Python child's
``send_event`` lands on the root context instead of being accumulated into the
call (the caller sees an empty result), its ``get_resource`` resolves against the
root plan, and an awaitable child action would hit a ClassCastException because
``PythonInternalSubagentContextImpl`` is not a ``PythonRunnerContextImpl``. These
tests are the executable spec for that work; the caller side (bootstrap/await
over pemja, scope-aware resource cache) is already in place, so a Java child
driven from a Python caller does work.

Following the HEAD deferred-setup convention, the Python internal setup requires
explicit ``(session_id, call_id)`` identities on ``submit``; the ids below are
fixed because each test runs one record through one key.
"""

import os
import sysconfig
from pathlib import Path
from typing import Any

import pytest
from pyflink.common import Encoder
from pyflink.common.typeinfo import Types
from pyflink.datastream import KeySelector, StreamExecutionEnvironment
from pyflink.datastream.connectors.file_system import StreamingFileSink

from flink_agents.api.agents.agent import Agent
from flink_agents.api.decorators import action
from flink_agents.api.events.event import Event, InputEvent, OutputEvent
from flink_agents.api.execution_environment import AgentsExecutionEnvironment
from flink_agents.api.resource import ResourceType
from flink_agents.api.runner_context import RunnerContext

os.environ["PYTHONPATH"] = sysconfig.get_paths()["purelib"]

# A Python child agent is not scope-aware yet (see the module docstring), so
# these end-to-end tests remain skipped until that work lands; they are the
# executable spec for it. The caller side (bootstrap/await over pemja) is in
# place and is exercised from the Java caller path.
pytestmark = pytest.mark.skip(
    reason="Python child agents are not scope-aware yet; tracked separately."
)

CHILD_SCOPE = "reviewer"
GRANDCHILD_SCOPE = "speller"


class InputKeySelector(KeySelector):
    """Keys every element by itself."""

    def get_key(self, value: Any) -> Any:
        """Return the element itself as key."""
        return value


class EchoChildAgent(Agent):
    """Child agent echoing the prompt back as one output event."""

    @action(InputEvent.EVENT_TYPE)
    @staticmethod
    def echo(event: Event, ctx: RunnerContext) -> None:
        """Emit the prompt back, which the caller collects as the call result."""
        ctx.send_event(OutputEvent(output=f"echo:{InputEvent.from_event(event).input}"))


class FailingChildAgent(Agent):
    """Child agent failing instead of producing output."""

    @action(InputEvent.EVENT_TYPE)
    @staticmethod
    def refuse(event: Event, ctx: RunnerContext) -> None:
        """Raise, so the caller's result reports the failure."""
        msg = f"child refused {InputEvent.from_event(event).input}"
        raise RuntimeError(msg)


class NestingChildAgent(Agent):
    """Child agent that awaits a grandchild of its own."""

    def __init__(self) -> None:
        """Register the grandchild this agent delegates to."""
        super().__init__()
        self.add_resource(GRANDCHILD_SCOPE, ResourceType.AGENT, EchoChildAgent())

    @action(InputEvent.EVENT_TYPE)
    @staticmethod
    async def relay(event: Event, ctx: RunnerContext) -> None:
        """Delegate one level deeper and forward the grandchild's output."""
        prompt = InputEvent.from_event(event).input
        grandchild = ctx.get_resource(GRANDCHILD_SCOPE, ResourceType.AGENT)
        result = await grandchild.submit(
            ctx, f"deep:{prompt}", session_id="s2", call_id="c2"
        )
        ctx.send_event(OutputEvent(output=f"relayed:{result.result[0]}"))


class RootAgent(Agent):
    """Caller agent awaiting the internal sub-agent from an async action."""

    @action(InputEvent.EVENT_TYPE)
    @staticmethod
    async def orchestrate(event: Event, ctx: RunnerContext) -> None:
        """Await the child and emit its outcome."""
        prompt = InputEvent.from_event(event).input
        child = ctx.get_resource(CHILD_SCOPE, ResourceType.AGENT)
        result = await child.submit(ctx, prompt, session_id="s1", call_id="c1")
        if result.success:
            ctx.send_event(OutputEvent(output=result.result[0]))
        else:
            ctx.send_event(OutputEvent(output=f"failed:{result.error_message}"))


def _run_and_collect(child: Agent, tmp_path: Path) -> str:
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)
    input_stream = env.from_collection(["hello"])

    root = RootAgent()
    root.add_resource(CHILD_SCOPE, ResourceType.AGENT, child)

    agents_env = AgentsExecutionEnvironment.get_execution_environment(env=env)
    output_datastream = (
        agents_env.from_datastream(input=input_stream, key_selector=InputKeySelector())
        .apply(root)
        .to_datastream()
    )

    result_dir = tmp_path / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    output_datastream.map(str, Types.STRING()).add_sink(
        StreamingFileSink.for_row_format(
            base_path=str(result_dir.absolute()),
            encoder=Encoder.simple_string_encoder(),
        ).build()
    )
    agents_env.execute()
    return "".join(p.read_text() for p in result_dir.rglob("*") if p.is_file())


def test_python_internal_subagent_call_end_to_end(tmp_path: Path) -> None:
    """The caller receives the child's accumulated output."""
    contents = _run_and_collect(EchoChildAgent(), tmp_path)

    assert "echo:hello" in contents, f"missing child output; collected: {contents!r}"


def test_python_internal_subagent_failure_surfaces_via_result(tmp_path: Path) -> None:
    """A failing child action is reported through the caller's ``Result``."""
    contents = _run_and_collect(FailingChildAgent(), tmp_path)

    # The child failure surfaces as a failed Result (the caller emits
    # "failed:...") instead of failing the job. The child's original message is
    # not asserted: it crosses the pemja boundary, which wraps it in a generic
    # PythonException whose message does not carry the cause's text (unlike
    # Java's in-process stack trace).
    assert "failed:" in contents, f"failure not reported; collected: {contents!r}"


def test_python_internal_subagent_nested_call(tmp_path: Path) -> None:
    """A child may itself await a grandchild registered in its own plan."""
    contents = _run_and_collect(NestingChildAgent(), tmp_path)

    assert "relayed:echo:deep:hello" in contents, (
        f"nested call did not complete; collected: {contents!r}"
    )
