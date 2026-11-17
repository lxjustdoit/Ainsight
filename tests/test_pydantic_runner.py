import unittest
from types import SimpleNamespace

from pydantic_ai.models.test import TestModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.capabilities import NativeTool

from app.core.config import LangfuseConfig
from app.services.ai.agents.qa_agent import QAAgent
from app.services.ai.model_config import AgentModelConfig
from app.services.ai.output_repair import OutputRepairFailure, parse_repaired_output
from app.services.ai.prompt.prompts import IDENTITY_RULES, build_system_prompt
from app.services.ai.pydantic_outputs import QAOutput
from app.services.ai.pydantic_runner import PydanticAIAgentRunner, PydanticAIUsage
from app.services.observability.context import observability_scope
from app.services.observability.langfuse_client import InMemoryLangfuseEventSink, _configure_langfuse


class SequenceRunner(PydanticAIAgentRunner):
    def __init__(self, outputs):
        super().__init__(model=object())
        self.outputs = list(outputs)
        self.prompts = []
        self.capabilities = []
        self.attempts = []

    def _request_llm_once(
        self,
        *,
        model,
        model_config,
        output_type,
        schema_output_type,
        validation_output_type,
        system_prompt,
        user_prompt,
        capabilities=None,
        attempt=0,
    ):
        self.prompts.append(user_prompt)
        self.capabilities.append(capabilities or [])
        self.attempts.append(attempt)
        raw_output = self.outputs.pop(0)
        usage = PydanticAIUsage(input_tokens=1, output_tokens=2, total_tokens=3)
        if validation_output_type is None:
            return raw_output, usage
        try:
            parsed = parse_repaired_output(
                raw_output,
                output_type=validation_output_type,
                task_type=model_config.task_type,
            )
        except OutputRepairFailure as exc:
            exc._ainsight_provider_result = SimpleNamespace(usage=usage)
            raise
        return parsed, usage


class PydanticAIAgentRunnerTests(unittest.TestCase):
    def setUp(self):
        self.langfuse = InMemoryLangfuseEventSink()
        _configure_langfuse(LangfuseConfig(), sink=self.langfuse, synchronous=True)

    def test_runner_uses_pydantic_ai_agent_and_structured_output(self):
        runner = PydanticAIAgentRunner(model=TestModel(custom_output_text='{"answer":"Only current page context."}'))

        result = runner.run(
            model_config=AgentModelConfig(
                provider="local",
                model_name="test",
                task_type="qa",
                prompt_version="qa_v1",
                input_token_price_per_1k=0.1,
                output_token_price_per_1k=0.2,
            ),
            output_type=QAOutput,
            system_prompt="Answer from current page only.",
            user_prompt="question: what is this page about?",
        )

        self.assertEqual(result.output.answer, "Only current page context.")
        self.assertGreater(result.usage.total_tokens, 0)
        self.assertEqual(result.usage.cost, 0.0)

    def test_runner_records_prompt_response_and_usage_at_provider_boundary(self):
        runner = PydanticAIAgentRunner(model=TestModel(custom_output_text='{"answer":"Observed answer."}'))

        with observability_scope(trace_id="00000000000070008000000000000009"):
            runner.run(
                model_config=AgentModelConfig(
                    provider="local",
                    model_name="test",
                    task_type="qa",
                    prompt_version="qa_v1",
                ),
                output_type=QAOutput,
                system_prompt="Return JSON.",
                user_prompt="question: what is this page about?",
            )

        event = self.langfuse.events[-1]
        self.assertEqual(event.trace_id, "00000000000070008000000000000009")
        self.assertEqual(event.prompt["system"], "Return JSON.")
        self.assertEqual(event.prompt["user"], "question: what is this page about?")
        self.assertIn("Observed answer", event.response)
        self.assertGreater(event.total_tokens, 0)
        self.assertEqual(event.validation_result, "passed")
        self.assertRegex(event.llm_call_id, r"^llm_[0-9a-f]{32}$")
        self.assertEqual(self.langfuse.starts[-1].llm_call_id, event.llm_call_id)

    def test_runner_marks_invalid_provider_outputs_in_langfuse(self):
        runner = PydanticAIAgentRunner(model=TestModel(custom_output_text="{}"))

        with self.assertRaises(OutputRepairFailure):
            runner.run(
                model_config=AgentModelConfig(
                    provider="local",
                    model_name="test",
                    task_type="qa",
                    prompt_version="qa_v1",
                ),
                output_type=QAOutput,
                system_prompt="Return JSON.",
                user_prompt="question: what is this page about?",
            )

        self.assertEqual(len(self.langfuse.events), 2)
        self.assertTrue(all(event.status == "error" for event in self.langfuse.events))
        self.assertTrue(all(event.validation_result == "failed" for event in self.langfuse.events))
        self.assertEqual(len({event.trace_id for event in self.langfuse.events}), 1)

    def test_runner_keeps_debug_llm_error_while_redaction_is_paused(self):
        class FailingAgent:
            def run_sync(self, *_args, **_kwargs):
                raise RuntimeError("provider failed Authorization: Bearer private-token")

        runner = PydanticAIAgentRunner(model=object())
        with self.assertRaises(RuntimeError):
            runner._start_llm_call(
                agent=FailingAgent(),
                user_prompt="question",
                system_prompt="system",
                model_config=AgentModelConfig(
                    provider="local",
                    model_name="test",
                    task_type="qa",
                    prompt_version="qa_v1",
                ),
            )

        event = self.langfuse.events[-1]
        self.assertIn("private-token", event.error_message)
        self.assertNotIn("[REDACTED]", event.error_message)

    def test_runner_wraps_plain_text_qa_output_when_structured_output_is_disabled(self):
        runner = PydanticAIAgentRunner(model=TestModel(custom_output_text="Only current page context."))

        result = runner.run(
            model_config=AgentModelConfig(provider="local", model_name="test", task_type="qa", prompt_version="qa_v1"),
            output_type=QAOutput,
            system_prompt="Return JSON.",
            user_prompt="question: what is this page about?",
        )

        self.assertEqual(result.output.answer, "Only current page context.")

    def test_runner_text_output_returns_plain_text(self):
        runner = PydanticAIAgentRunner(model=TestModel(custom_output_text="Plain final answer."))

        result = runner.run_text(
            model_config=AgentModelConfig(provider="local", model_name="test", task_type="qa_final", prompt_version="qa_v1"),
            system_prompt="Return plain text.",
            user_prompt="question: what is this page about?",
        )

        self.assertEqual(result.output, "Plain final answer.")

    def test_runner_regenerates_once_with_feedback_when_required_fields_are_missing(self):
        runner = SequenceRunner(
            [
                "{}",
                '{"answer":"Regenerated answer."}',
            ]
        )

        result = runner.run(
            model_config=AgentModelConfig(provider="local", model_name="test", task_type="qa", prompt_version="qa_v1"),
            output_type=QAOutput,
            system_prompt="Return JSON.",
            user_prompt="question: what is this page about?",
        )

        self.assertEqual(result.output.answer, "Regenerated answer.")
        self.assertEqual(result.usage.total_tokens, 6)
        self.assertIn("previousFailureFeedback", runner.prompts[1])
        self.assertEqual(runner.attempts, [0, 1])

    def test_prompt_builder_includes_identity_rules(self):
        prompt = build_system_prompt(
            agent_name="QAAgent",
            task_rules=["Answer from page context."],
            output_rules=['Example output: {"answer":"..."}'],
        )

        self.assertIn(IDENTITY_RULES, prompt)
        self.assertIn("Do not mention the underlying model provider", prompt)

    def test_qa_agent_web_search_capability_is_config_controlled(self):
        disabled_agent = QAAgent(
            AgentModelConfig(provider="local", model_name="test", task_type="qa", prompt_version="qa_v1"),
            ai_runner=object(),
        )
        enabled_agent = QAAgent(
            AgentModelConfig(
                provider="local",
                model_name="test",
                task_type="qa",
                prompt_version="qa_v1",
                tools_enabled=True,
            ),
            ai_runner=object(),
        )

        disabled = disabled_agent._qa_capabilities()
        enabled = enabled_agent._qa_capabilities()

        self.assertEqual(disabled, [])
        self.assertEqual(len(enabled), 1)
        self.assertIsInstance(enabled[0], NativeTool)

    def test_runner_accepts_capabilities_from_agent_call(self):
        capability = object()
        runner = SequenceRunner(['{"answer":"Uses passed capabilities."}'])

        runner.run(
            model_config=AgentModelConfig(provider="local", model_name="test", task_type="qa", prompt_version="qa_v1"),
            output_type=QAOutput,
            system_prompt="Return JSON.",
            user_prompt="question: what is this page about?",
            capabilities=[capability],
        )

        self.assertEqual(runner.capabilities[0], [capability])

    def test_builds_minimax_anthropic_model_without_calling_network(self):
        runner = PydanticAIAgentRunner()

        model = runner._build_model(
            AgentModelConfig(
                provider="anthropic_compatible",
                model_name="MiniMax-M3",
                task_type="qa",
                prompt_version="qa_v1",
                base_url="https://api.minimax.io/anthropic",
                api_key="test-key",
            )
        )

        self.assertIsInstance(model, AnthropicModel)


if __name__ == "__main__":
    unittest.main()
