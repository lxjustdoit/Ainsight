import json
import os
import tempfile
import unittest
from pathlib import Path

from app.core.config import load_config


class ConfigTests(unittest.TestCase):
    def test_loads_llm_config_from_file_and_env_override(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as config_file:
            json.dump(
                {
                    "ai": {
                        "summary": {
                            "provider": "openai_compatible",
                            "model_name": "configured-model",
                            "base_url": "https://llm.example/v1",
                            "api_key": "file-key",
                            "prompt_version": "summary_v9",
                            "structured_output": True,
                            "tools_enabled": True,
                            "target_output_tokens": 700,
                            "max_output_tokens": 1100,
                            "recent_qa_messages": 5,
                            "search_findings_min_chars": 42,
                            "search_findings_similarity_threshold": 0.41,
                            "search_findings_max_results": 4,
                            "search_findings_sentence_limit": 2,
                            "search_findings_max_chars_per_result": 600,
                            "search_findings_max_total_chars": 1800,
                            "input_token_price_per_1k": 0.1,
                            "output_token_price_per_1k": 0.2,
                        },
                        "embedding": {
                            "enabled": True,
                            "provider": "openai_compatible",
                            "model_name": "embedding-model",
                            "base_url": "https://embedding.example/v1",
                            "api_key": "embedding-file-key",
                            "timeout_seconds": 3.5,
                            "dimensions": 256,
                            "max_input_chars": 4096,
                        },
                    }
                },
                config_file,
            )
            config_file.flush()
            os.environ["SUMMARY_LLM_API_KEY"] = "env-key"
            try:
                config = load_config(config_file.name)
            finally:
                os.environ.pop("SUMMARY_LLM_API_KEY", None)

        self.assertEqual(config.ai.summary.model_name, "configured-model")
        self.assertEqual(config.ai.summary.base_url, "https://llm.example/v1")
        self.assertEqual(config.ai.summary.api_key, "env-key")
        self.assertEqual(config.ai.summary.prompt_version, "summary_v9")
        self.assertTrue(config.ai.summary.structured_output)
        self.assertTrue(config.ai.summary.tools_enabled)
        self.assertEqual(config.ai.summary.target_output_tokens, 700)
        self.assertEqual(config.ai.summary.max_output_tokens, 1100)
        self.assertEqual(config.ai.summary.recent_qa_messages, 5)
        self.assertEqual(config.ai.summary.search_findings_min_chars, 42)
        self.assertEqual(config.ai.summary.search_findings_similarity_threshold, 0.41)
        self.assertEqual(config.ai.summary.search_findings_max_results, 4)
        self.assertEqual(config.ai.summary.search_findings_sentence_limit, 2)
        self.assertEqual(config.ai.summary.search_findings_max_chars_per_result, 600)
        self.assertEqual(config.ai.summary.search_findings_max_total_chars, 1800)
        self.assertEqual(config.ai.summary.input_token_price_per_1k, 0.1)
        self.assertEqual(config.ai.summary.output_token_price_per_1k, 0.2)
        self.assertTrue(config.ai.embedding.enabled)
        self.assertEqual(config.ai.embedding.model_name, "embedding-model")
        self.assertEqual(config.ai.embedding.base_url, "https://embedding.example/v1")
        self.assertEqual(config.ai.embedding.api_key, "embedding-file-key")
        self.assertEqual(config.ai.embedding.timeout_seconds, 3.5)
        self.assertEqual(config.ai.embedding.dimensions, 256)
        self.assertEqual(config.ai.embedding.max_input_chars, 4096)

    def test_loads_unified_yaml_config(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as config_file:
            config_file.write(
                """
cache:
  page_context_ttl_hours: 12
  summary_ttl_hours: 11
  graph_ttl_hours: 10
  qa_session_ttl_hours: 9
observability:
  service_name: ainsight-test
  logging:
    level: DEBUG
    max_field_length: 1500
  otel:
    enabled: true
    exporter_endpoint: http://otel.example:4317
    exporter_headers:
      x-api-key: otel-key
    exporter_insecure: false
    exporter_protocol: http/protobuf
    traces:
      enabled: true
      sampling_ratio: 0.5
    metrics:
      enabled: true
    queue:
      max_queue_size: 1024
      schedule_delay_ms: 1000
      max_export_batch_size: 128
      export_timeout_ms: 5000
  langfuse:
    enabled: true
    host: http://langfuse.example
    public_key: public
    secret_key: secret
    debug: true
    flush_interval_ms: 2000
    queue:
      max_size: 500
      enqueue_timeout_ms: 3
      flush_timeout_seconds: 4
      drop_when_full: false
ai:
  summary:
    provider: openai_compatible
    model_name: yaml-summary-model
    base_url: https://yaml.example/v1
    api_key: yaml-key
    prompt_version: summary_yaml
    structured_output: true
    tools_enabled: true
    target_output_tokens: 750
    max_output_tokens: 1150
    recent_qa_messages: 7
    search_findings_min_chars: 45
    search_findings_similarity_threshold: 0.42
    search_findings_max_results: 5
    search_findings_sentence_limit: 4
    search_findings_max_chars_per_result: 700
    search_findings_max_total_chars: 1900
  embedding:
    enabled: true
    provider: openai_compatible
    model_name: yaml-embedding-model
    base_url: https://embedding.yaml/v1
    api_key: yaml-embedding-key
    timeout_seconds: 4
    dimensions: 512
    max_input_chars: 5000
"""
            )
            config_file.flush()
            config = load_config(config_file.name)

        self.assertEqual(config.cache.page_context_ttl_hours, 12)
        self.assertEqual(config.cache.summary_ttl_hours, 11)
        self.assertEqual(config.cache.graph_ttl_hours, 10)
        self.assertEqual(config.cache.qa_session_ttl_hours, 9)
        self.assertEqual(config.observability.service_name, "ainsight-test")
        self.assertEqual(config.observability.logging.level, "DEBUG")
        self.assertEqual(config.observability.logging.max_field_length, 1500)
        self.assertTrue(config.observability.otel.enabled)
        self.assertEqual(config.observability.otel.exporter_endpoint, "http://otel.example:4317")
        self.assertEqual(config.observability.otel.exporter_headers, {"x-api-key": "otel-key"})
        self.assertFalse(config.observability.otel.exporter_insecure)
        self.assertEqual(config.observability.otel.exporter_protocol, "http/protobuf")
        self.assertEqual(config.observability.otel.traces.sampling_ratio, 0.5)
        self.assertTrue(config.observability.otel.metrics.enabled)
        self.assertEqual(config.observability.otel.queue.max_queue_size, 1024)
        self.assertEqual(config.observability.otel.queue.schedule_delay_ms, 1000)
        self.assertEqual(config.observability.otel.queue.max_export_batch_size, 128)
        self.assertEqual(config.observability.otel.queue.export_timeout_ms, 5000)
        self.assertTrue(config.observability.langfuse.enabled)
        self.assertEqual(config.observability.langfuse.host, "http://langfuse.example")
        self.assertEqual(config.observability.langfuse.public_key, "public")
        self.assertEqual(config.observability.langfuse.secret_key, "secret")
        self.assertTrue(config.observability.langfuse.debug)
        self.assertEqual(config.observability.langfuse.flush_interval_ms, 2000)
        self.assertEqual(config.observability.langfuse.queue.max_size, 500)
        self.assertEqual(config.observability.langfuse.queue.enqueue_timeout_ms, 3)
        self.assertEqual(config.observability.langfuse.queue.flush_timeout_seconds, 4)
        self.assertFalse(config.observability.langfuse.queue.drop_when_full)
        self.assertTrue(config.observability.langfuse_enabled)
        self.assertTrue(config.observability.signoz_enabled)
        self.assertEqual(config.ai.summary.model_name, "yaml-summary-model")
        self.assertEqual(config.ai.summary.api_key, "yaml-key")
        self.assertTrue(config.ai.summary.structured_output)
        self.assertTrue(config.ai.summary.tools_enabled)
        self.assertEqual(config.ai.summary.target_output_tokens, 750)
        self.assertEqual(config.ai.summary.max_output_tokens, 1150)
        self.assertEqual(config.ai.summary.recent_qa_messages, 7)
        self.assertEqual(config.ai.summary.search_findings_min_chars, 45)
        self.assertEqual(config.ai.summary.search_findings_similarity_threshold, 0.42)
        self.assertEqual(config.ai.summary.search_findings_max_results, 5)
        self.assertEqual(config.ai.summary.search_findings_sentence_limit, 4)
        self.assertEqual(config.ai.summary.search_findings_max_chars_per_result, 700)
        self.assertEqual(config.ai.summary.search_findings_max_total_chars, 1900)
        self.assertTrue(config.ai.embedding.enabled)
        self.assertEqual(config.ai.embedding.model_name, "yaml-embedding-model")
        self.assertEqual(config.ai.embedding.dimensions, 512)
        self.assertEqual(config.ai.embedding.max_input_chars, 5000)

    def test_environment_overrides_yaml_system_config(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as config_file:
            config_file.write(
                """
cache:
  page_context_ttl_hours: 12
observability:
  service_name: from-file
  logging:
    level: INFO
  langfuse:
    enabled: false
ai:
  summary:
    model_name: from-file
    structured_output: false
    tools_enabled: false
  embedding:
    enabled: false
    model_name: file-embedding
"""
            )
            config_file.flush()
            os.environ["PAGE_CONTEXT_TTL_HOURS"] = "3"
            os.environ["SERVICE_NAME"] = "from-env"
            os.environ["LOG_LEVEL"] = "ERROR"
            os.environ["LANGFUSE_ENABLED"] = "true"
            os.environ["OTEL_ENABLED"] = "true"
            os.environ["SUMMARY_LLM_MODEL"] = "env-model"
            os.environ["SUMMARY_STRUCTURED_OUTPUT"] = "true"
            os.environ["SUMMARY_TOOLS_ENABLED"] = "true"
            os.environ["SUMMARY_TARGET_OUTPUT_TOKENS"] = "760"
            os.environ["SUMMARY_MAX_OUTPUT_TOKENS"] = "1160"
            os.environ["SUMMARY_RECENT_QA_MESSAGES"] = "8"
            os.environ["SUMMARY_SEARCH_FINDINGS_MIN_CHARS"] = "50"
            os.environ["SUMMARY_SEARCH_FINDINGS_SIMILARITY_THRESHOLD"] = "0.5"
            os.environ["SUMMARY_SEARCH_FINDINGS_MAX_RESULTS"] = "6"
            os.environ["SUMMARY_SEARCH_FINDINGS_SENTENCE_LIMIT"] = "5"
            os.environ["SUMMARY_SEARCH_FINDINGS_MAX_CHARS_PER_RESULT"] = "800"
            os.environ["SUMMARY_SEARCH_FINDINGS_MAX_TOTAL_CHARS"] = "2000"
            os.environ["EMBEDDING_ENABLED"] = "true"
            os.environ["EMBEDDING_MODEL"] = "env-embedding"
            os.environ["EMBEDDING_DIMENSIONS"] = "1024"
            try:
                config = load_config(config_file.name)
            finally:
                os.environ.pop("PAGE_CONTEXT_TTL_HOURS", None)
                os.environ.pop("SERVICE_NAME", None)
                os.environ.pop("LOG_LEVEL", None)
                os.environ.pop("LANGFUSE_ENABLED", None)
                os.environ.pop("OTEL_ENABLED", None)
                os.environ.pop("SUMMARY_LLM_MODEL", None)
                os.environ.pop("SUMMARY_STRUCTURED_OUTPUT", None)
                os.environ.pop("SUMMARY_TOOLS_ENABLED", None)
                os.environ.pop("SUMMARY_TARGET_OUTPUT_TOKENS", None)
                os.environ.pop("SUMMARY_MAX_OUTPUT_TOKENS", None)
                os.environ.pop("SUMMARY_RECENT_QA_MESSAGES", None)
                os.environ.pop("SUMMARY_SEARCH_FINDINGS_MIN_CHARS", None)
                os.environ.pop("SUMMARY_SEARCH_FINDINGS_SIMILARITY_THRESHOLD", None)
                os.environ.pop("SUMMARY_SEARCH_FINDINGS_MAX_RESULTS", None)
                os.environ.pop("SUMMARY_SEARCH_FINDINGS_SENTENCE_LIMIT", None)
                os.environ.pop("SUMMARY_SEARCH_FINDINGS_MAX_CHARS_PER_RESULT", None)
                os.environ.pop("SUMMARY_SEARCH_FINDINGS_MAX_TOTAL_CHARS", None)
                os.environ.pop("EMBEDDING_ENABLED", None)
                os.environ.pop("EMBEDDING_MODEL", None)
                os.environ.pop("EMBEDDING_DIMENSIONS", None)

        self.assertEqual(config.cache.page_context_ttl_hours, 3)
        self.assertEqual(config.observability.service_name, "from-env")
        self.assertEqual(config.observability.logging.level, "ERROR")
        self.assertTrue(config.observability.langfuse_enabled)
        self.assertTrue(config.observability.signoz_enabled)
        self.assertEqual(config.ai.summary.model_name, "env-model")
        self.assertTrue(config.ai.summary.structured_output)
        self.assertTrue(config.ai.summary.tools_enabled)
        self.assertEqual(config.ai.summary.target_output_tokens, 760)
        self.assertEqual(config.ai.summary.max_output_tokens, 1160)
        self.assertEqual(config.ai.summary.recent_qa_messages, 8)
        self.assertEqual(config.ai.summary.search_findings_min_chars, 50)
        self.assertEqual(config.ai.summary.search_findings_similarity_threshold, 0.5)
        self.assertEqual(config.ai.summary.search_findings_max_results, 6)
        self.assertEqual(config.ai.summary.search_findings_sentence_limit, 5)
        self.assertEqual(config.ai.summary.search_findings_max_chars_per_result, 800)
        self.assertEqual(config.ai.summary.search_findings_max_total_chars, 2000)
        self.assertTrue(config.ai.embedding.enabled)
        self.assertEqual(config.ai.embedding.model_name, "env-embedding")
        self.assertEqual(config.ai.embedding.dimensions, 1024)

    def test_discovers_default_yaml_when_env_is_not_set(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(
                """
ai:
  summary:
    model_name: discovered-model
""",
                encoding="utf-8",
            )
            previous_cwd = os.getcwd()
            os.chdir(directory)
            try:
                config = load_config()
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(config.ai.summary.model_name, "discovered-model")

    def test_discovers_base_yaml_and_overlays_local_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.yml").write_text(
                """
observability:
  service_name: base-service
  logging:
    level: INFO
    max_field_length: 1000
ai:
  summary:
    model_name: base-model
    base_url: https://base.example/v1
""",
                encoding="utf-8",
            )
            Path(directory, "config.local.yaml").write_text(
                """
observability:
  logging:
    level: DEBUG
ai:
  summary:
    api_key: local-key
""",
                encoding="utf-8",
            )
            previous_cwd = os.getcwd()
            os.chdir(directory)
            try:
                config = load_config()
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(config.observability.service_name, "base-service")
        self.assertEqual(config.observability.logging.level, "DEBUG")
        self.assertEqual(config.observability.logging.max_field_length, 1000)
        self.assertEqual(config.ai.summary.model_name, "base-model")
        self.assertEqual(config.ai.summary.base_url, "https://base.example/v1")
        self.assertEqual(config.ai.summary.api_key, "local-key")

    def test_legacy_embedding_threshold_key_maps_to_similarity_threshold(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as config_file:
            json.dump(
                {
                    "ai": {
                        "qa": {
                            "search_findings_embedding_threshold": 0.33,
                        }
                    }
                },
                config_file,
            )
            config_file.flush()
            config = load_config(config_file.name)

        self.assertEqual(config.ai.qa.search_findings_similarity_threshold, 0.33)


if __name__ == "__main__":
    unittest.main()
