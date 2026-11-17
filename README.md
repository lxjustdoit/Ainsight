# Ainsight Backend

Backend core and interactive CLI for the Ainsight AI reading assistant.

The current implementation provides page summaries, structure graphs, and Q&A through Pydantic AI, with structured output validation and an in-memory page context cache. Optional OpenTelemetry and Langfuse adapters provide tracing and monitoring.

## Setup

Use Python 3.12. Run these commands from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp config.example.yml config.local.yml
```

Fill in provider settings and credentials in `config.local.yml`. The YML example contains generic settings, empty credentials, and disabled telemetry.

Actual configuration, credentials, logs, test reports, and internal documents are kept locally. Review the exact staged files before committing.

## Configuration

Without an explicit path, the application loads the first existing base file from `config.yaml`, `config.yml`, and `config.json`, then overlays the first existing local file from `config.local.yaml`, `config.local.yml`, and `config.local.json`. All runtime configuration files are ignored by Git.

Environment variables override file values. Provider credential variables include `SUMMARY_LLM_API_KEY`, `GRAPH_LLM_API_KEY`, `QA_LLM_API_KEY`, and `EMBEDDING_API_KEY`.

To load a single file instead of merging base and local configuration, set `AINSIGHT_CONFIG_FILE` to its path.

Use one local format at a time, since YAML files take precedence over JSON during automatic discovery.

## Run

With the virtual environment activated and provider credentials configured, start the interactive CLI or summary demo.

```bash
python cli.py
python main.py
```

The CLI supports page loading, summaries, graphs, and Q&A. Enter `/help` for commands. The demo and model-backed commands make real provider requests.

## Tests

Unit tests use mocks and in-memory adapters. They require installed dependencies and can run without real provider credentials or observability services.

```bash
python -m unittest discover -s tests
```

Run an affected test module when checking a change.

```bash
python -m unittest tests.test_config
```

GitHub CI and report synchronization are planned. This initial repository does not include CI or deployment automation.

## Optional Observability

OpenTelemetry and Langfuse are disabled in the configuration example. Enable them in local configuration only after configuring the corresponding service endpoints and credentials.

Application entry points use `configure_observability()` and flush resources with `shutdown_observability()`. Restart the process to change the runtime configuration after shutdown.

JSON logging defaults to stdout. Set `observability.logging.output_path`, or its environment override `LOG_OUTPUT_PATH`, to write a JSONL file. For CLI collection, use that structured log file as the Collector input.

`otel-collector-config.yaml` is a generic local Collector example for the `otel/opentelemetry-collector-contrib:0.123.0` image. It receives OTLP on container ports `4317` and `4318`, reads application JSONL files mounted at `/var/log/ainsight`, and exports to `SIGNOZ_OTLP_ENDPOINT`. Its default downstream endpoint is `host.docker.internal:4317`. It uses plaintext local transport and replays existing files after restart.

Detailed LLM tracing can include prompt and response content in Langfuse. Keep real traces, service exports, credentials, and environment-specific deployment configuration outside Git.
