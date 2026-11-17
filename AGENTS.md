# Ainsight Backend Implementation Guidelines

> When modifying this document, only change content that has been explicitly confirmed. Do not add, remove, rewrite, or infer unapproved content without the user's permission.

## 1. Project and Sources of Truth

Ainsight is an AI assistant project for web reading. The current backend MVP is located in this directory.

Apply requirements in the following order:

1. The user's explicit instructions and approved conclusions in the current conversation.
2. This file.
3. Relevant design documents.
4. Existing code, tests, configuration samples, and the README.

Design documents define approved goals, functional rules, and architecture. Existing code and tests reflect behavior that is already implemented and verifiable.

When design documents, current code, and approved requirements are inconsistent, do not independently choose an implementation or represent planned capabilities as completed. First describe the discrepancy and wait for a user decision.

## 2. Planning, Documentation, and Code Changes

### 2.1 Design Document Index

Do not duplicate volatile functional details in this file. Read the following sources before making the relevant type of change:

| Change type | Required design source |
| --- | --- |
| Data, privacy, and caching | `../ainsight_backend_design.md`, sections 3, 4, and 9 |
| APIs, context upload, summaries, graphs, Q&A, and bookmarks | `../ainsight_backend_design.md`, sections 7 and 8 |
| Long-document handling | `../ainsight_backend_design.md`, section 10 |
| Agents, models, prompts, streaming, output repair, and token budgets | `../ainsight_backend_design.md`, section 11 |
| Admin capabilities | `../ainsight_backend_design.md`, section 12 |
| Logging, tracing, SigNoz, and Langfuse | `../ainsight_backend_design.md`, section 13 |
| Authentication, authorization, and security boundaries | `../permission_security_design.md` |
| Cross-end interaction and user experience | `../ainsight_frontend_design.md` |
| Detailed agent review rules | `.agents/review-guidelines.md` |
| GitHub delivery, PR preparation, CI, and report synchronization | `../agent/github-guidelines.md` |

### 2.2 Plan Review

After proposing an implementation plan, functional design, architecture change, or documentation update, wait for user review. Continue only after receiving explicit instruction to proceed, revise, or confirm.

### 2.3 Documentation Changes

When changing documentation:

- First collect requirements, decisions, and terminology that are explicit and approved in the current conversation.
- Wait until the change content, scope, and conclusion for the current task are fully confirmed, then update the relevant documents together.
- Ask the user before documenting anything that is absent from both the current conversation and existing documentation and could affect functionality, interfaces, data, performance, or future development.
- Do not infer product requirements, interface fields, architecture components, implementation choices, or delivery phases.

### 2.4 Code Changes

Before changing code:

- Read the relevant design documents, implementation, callers, tests, and configuration.
- Check that approved conversation context, design documents, and current implementation agree.
- If they disagree, describe the difference. Synchronize and obtain approval for documentation before implementing code.
- Do not modify application or test code until both the implementation plan and scope are explicitly approved and the user explicitly instructs the agent to begin code changes. Plan approval alone is not execution authorization.
- If authorization to begin code changes is unclear, ask the user for explicit confirmation before changing code.
- Implement only the approved scope. Do not implement unrelated design-document TODOs.

## 3. Architecture and Functional Design Standards

### 3.1 Lightweight, Clear, and Extensible Design

- Build the smallest complete solution for the approved requirement. Do not introduce frameworks, services, message queues, or complex abstractions for hypothetical future needs.
- Define an interface or adapter only when a capability genuinely needs interchangeable implementations, multiple external providers, an independent lifecycle, or reuse across modules.
- Keep a single, stable, non-reused capability direct and simple; avoid abstractions without a concrete benefit.
- Modules collaborate through public methods, explicit parameters, return values, exceptions, and configuration. Do not read or mutate another module's internal state.
- Avoid cyclic dependencies, overlapping responsibilities, and copied logic across modules.
- When a task explicitly requires evaluating or changing the current solution, research and compare current, stable mainstream approaches. Describe the current approach, alternatives, applicability, trade-offs, and potential issues in the current approach, then wait for user review before implementation.
- The API layer depends on business services and is responsible for protocol and boundary validation. Business services are responsible for workflows and include abstractions for model providers, caches, databases, and other middleware. The infrastructure layer adapts model providers, caches, databases, and external systems, and must not depend back on business services.
- Pydantic AI, model-provider, and tool layers must not own authentication, authorization, resource ownership, cache lifecycle, or product privacy rules.
- Prefer official documentation, mainstream open-source projects, or authoritative technical sources for solution research, and state the sources used.

### 3.2 Design Checklist

For a new feature, interface, cache, model, storage mechanism, concurrency behavior, or external dependency, a proposal must describe:

- Goals, boundaries, non-goals, and affected modules.
- Inputs, outputs, error paths, and compatibility.
- Data lifecycle, resource ownership, timeouts, retries, fallback behavior, and idempotency.
- Latency, external-call count, token usage, and cache strategy.
- Documentation, configuration, tests, instrumentation, and callers that require updates.

Before adding a complex component or dependency, explain its necessity, alternatives, and maintenance cost.

## 4. Code Quality, Documentation, and Reuse

- Keep code style consistent, concise, readable, maintainable, and extensible.
- Add a concise description at the beginning of each business file, covering its business function, purpose, boundary, and primary collaborators.
- Add a concise description at the beginning of each business class, covering its responsibility, business purpose, and primary inputs and outputs.
- Add explanatory comments for core business steps, non-obvious decisions, complex branches, retry and fallback behavior, concurrency controls, and privacy handling. Keep comments concise and readable.
- Clearly mark any non-test code that exists solely to support tests with a concise `TEST-ONLY` comment or docstring explaining its purpose. Keep test-support code isolated and do not invoke it from production paths.
- Document core business configuration values with their purpose, unit or valid range, default behavior, and functional effect. Never put real credentials in comments.
- Keep comments synchronized with code. Update related comments when changing behavior, boundaries, or configuration semantics.
- Avoid comments that merely repeat code, are stale, or provide no useful information.
- Use Pydantic models, `dataclass`, `TypedDict`, or explicit types for known structures. Avoid unconstrained `Any` and unstructured dictionaries.
- Use `snake_case` for modules, functions, and variables, and `PascalCase` for classes.
- Extract repeated, semantically identical logic within a file into a reusable local method.
- Maintain deterministic, low-coupling, reusable utilities shared across modules in an appropriate `utils/` module.
- Do not place business workflows, domain rules, or external dependencies in `utils/` under the guise of utility functions.
- Keep APIs, schemas, cache fields, configuration fields, prompt fields, and tests consistent. When changing a contract, inspect all producers and consumers.
- Add a dependency only when necessary, compatible with the supported Python version and license requirements, and explicitly approved by the user.

## 5. Security, Privacy, and Data Boundaries

- During the current development and debugging phase, privacy-information masking and redaction are out of scope for implementation and review until the user explicitly enables them.
- This temporary exception does not permit exposing real passwords, API keys, tokens, cookies, or OAuth credentials.
- Determine user identity, authorization, resource ownership, and sensitive configuration from trusted backend context. Do not trust client-declared identities or internal identifiers.
- Follow relevant design documents for persistence, temporary caches, resource isolation, context identifiers, Q&A sessions, and privacy boundaries.
- Do not expose real passwords, API keys, tokens, cookies, or OAuth credentials in source code, tests, documentation, logs, or commits.
- Do not include content prohibited by the design documents in logs, traces, or error messages.

## 6. Observability and Instrumentation

When implementing a feature, determine whether instrumentation must be added or adjusted. Follow `../ainsight_backend_design.md`, section 13, for field names, event names, and implementation locations.

The following scenarios must have correlated instrumentation where applicable:

- Request entry and completion.
- Main business-operation start, success, failure, and duration.
- LLM calls, retries, fallbacks, streaming completion, and failures.
- External dependencies such as Redis, databases, OAuth providers, and model providers.
- Important state changes, including bookmarks, model configuration, roles, permissions, and cache entries.

Instrumentation rules:

- Use the project's common logging, tracing, and LLM-monitoring entry points. Do not create an isolated instrumentation approach in business code.
- Use `trace_id` to correlate logs, spans, and LLM calls within a request.
- Give every real LLM call a distinct call identifier.
- Record safe metadata such as operation, task type, status, status code, duration, model name, token usage, error code, and safe resource identifiers.
- Do not collect content prohibited by the design documents.
- Add relevant instrumentation tests or assertions when introducing a main flow, external dependency, or important state change.

## 7. Code Review Standards

Detailed review tiers, timing, reviewer responsibilities, required checks,
finding priorities, and re-review rules are defined in
`.agents/review-guidelines.md`.
Follow `../agent/github-guidelines.md` for PR preparation, submission, and CI report synchronization.

Apply the following summary:

- Documentation, comments, and formatting changes require self-review and applicable lightweight validation.
- Non-core changes require self-review and inspection of the affected call chains and contracts.
- Complete all fixes for the current round, merge the latest `main`, and run affected local tests before unified review. Resolve all review findings for a round and rerun affected tests before unified re-review.
- Run full unit testing in PR CI after review and PR submission or update. An agent or developer explicitly triggers CI.
- Main-flow or high-risk changes require independent review.
- Contract, security, persistence, concurrency, LLM, privacy, streaming, or observability changes may require independent review even when they are not part of the main flow.
- Do not report completion until required review and validation are complete.

## 8. Testing and Validation

Run tests from `ainsight_backend/`.

- For code changes, run local tests covering the affected call chains after completing the current round's fixes and merging the latest `main`. Resolve failures before unified review or re-review. Documentation-only changes require applicable lightweight checks.
- After review and PR submission or update, explicitly trigger CI to run the full unit suite. PR creation and pushes do not automatically trigger CI.
- Example local command for an affected test module:

  ```bash
  PYTHONPATH=.deps .venv/bin/python -m unittest tests.test_config
  ```

- If CI is unavailable, report PR validation as incomplete and list any local checks separately.
- An output contract includes return values, schema or API fields, defaults, exception types, error codes, configuration behavior, and caller-visible side effects.
- Update or add tests for changes to output contracts, errors, configuration, cache semantics, retries, instrumentation, or privacy boundaries.
- During testing, record any severe bug that materially affects a main flow. Keep the record concise and focus on the root cause, impact scope, and resolution.
- If a severe main-flow bug remains unresolved, state the reason it remains unresolved.
- At the end of testing, list all recorded severe main-flow bugs and their current resolution status.
- Report only commands actually run and their results. Explain any validation not run.

### 8.1 Performance Evaluation

After testing, evaluate performance when appropriate.

Potential measures include:

- Total request duration and duration of critical steps.
- External dependency call count and failure rate.
- Model input, output, and total token usage.
- Cache hit behavior.
- Time to first streaming event.
- Retry, fallback, and error causes.

Use observed data to decide follow-up optimization. Do not add complexity based only on assumptions.

## 9. Completion Report

At task completion, report:

- Files changed and resulting behavior.
- Alignment with the approved plan and design documents.
- Review, tests, and validation performed.
- Severe main-flow bugs found during testing, including their root cause, impact scope, resolution, and any reason an item remains unresolved.
- Validation not run, risks, limitations, and unresolved follow-up items.
- Whether the work affects configuration, data, dependencies, privacy, external services, or deployment.

Do not perform real external writes, data migrations, deployment, commits, or pushes without explicit user authorization.
