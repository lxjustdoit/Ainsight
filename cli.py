"""Interactive agent debugging with config-driven telemetry and shared business flows."""

import html
import re
from dataclasses import dataclass
from typing import Optional
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pydantic_core.core_schema import none_schema

from app.core.config import AppConfig, load_config
from app.core.errors import AinsightError
from app.schemas.content import PageContent
from app.schemas.graph import FallbackTree, StructureGraph
from app.schemas.qa import QAResult
from app.schemas.summary import SummaryResult
from app.services.ai.agents.qa_agent import QAAgent
from app.services.ai.model_config import model_config_from_app
from app.services.ai.orchestrator import AIOrchestrator,QAResponse
from app.services.ai.pydantic_outputs import QAOutput
from app.services.ai.pydantic_runner import PydanticAIAgentRunner, PydanticAIResult
from app.services.ai.utils.text_utils import compact
from app.services.cache.context_cache import InMemoryContextCache
from app.services.observability.context import observability_scope
from app.services.observability.log_tracer import log_tracer
from app.services.observability.runtime import configure_observability, shutdown_observability
from app.services.ai.prompt.prompts import build_system_prompt_test

@dataclass
class CliState:
    """Keep the loaded page and result identifiers across interactive commands."""

    page: Optional[PageContent] = None
    context_id: Optional[str] = None
    summary_id: Optional[str] = None
    graph_id: Optional[str] = None
    last_trace_id: Optional[str] = None


def is_valid_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def fetch_page(url: str) -> PageContent:
    request = Request(url, headers={"User-Agent": "AinsightAgentDebugCLI/0.1"})
    with urlopen(request, timeout=20) as response:
        raw = response.read(700_000)
        charset = response.headers.get_content_charset() or "utf-8"
    markup = raw.decode(charset, errors="replace")
    return PageContent(url=url, title=extract_title(markup) or url, content=extract_text(markup))


def extract_title(markup: str) -> Optional[str]:
    match = re.search(r"<title[^>]*>(.*?)</title>", markup, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return compact(html.unescape(re.sub(r"\s+", " ", match.group(1))).strip(), 160)


def extract_text(markup: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", markup)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</(p|div|section|article|h[1-6]|li)>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def build_orchestrator(config) -> AIOrchestrator:
    return AIOrchestrator(
        cache=InMemoryContextCache(ttl_seconds=config.cache.page_context_ttl_hours * 3600),
        ai_runner=PydanticAIAgentRunner(),
        summary_model_config=model_config_from_app("summary", config.ai.summary),
        graph_model_config=model_config_from_app("graph", config.ai.graph),
        qa_model_config=model_config_from_app("qa", config.ai.qa),
    )


def prompt_for_page(state: CliState) -> None:
    value = input("请输入网页地址，留空或无效地址将进入普通问答模式: ").strip()
    if not is_valid_url(value):
        state.page = None
        state.context_id = None
        state.summary_id = None
        state.graph_id = None
        print("未检测到有效网址，当前为普通问答模式。使用 /page 可重新输入网址。")
        return
    try:
        print("正在读取页面内容...")
        state.page = fetch_page(value)
        state.context_id = None
        state.summary_id = None
        state.graph_id = None
        print(f"已加载页面: {state.page.title}")
    except (URLError, TimeoutError, UnicodeDecodeError, OSError) as exc:
        state.page = None
        state.context_id = None
        state.summary_id = None
        state.graph_id = None
        print(f"页面读取失败，已进入普通问答模式: {exc}")


def print_help() -> None:
    print(
        "\n命令:\n"
        "  /summery  生成当前网页总结，别名 /summary\n"
        "  /summery-stream  流式生成当前网页总结，别名 /summary-stream\n"
        "  /graph    基于当前网页和已生成总结生成结构图\n"
        "  /ask      基于当前网页问答；无网页时为普通问答\n"
        "  /ask-stream  流式问答；无网页时为普通问答流式调试\n"
        "  /page     重新输入网页地址，替换当前页面\n"
        "  /state    查看当前 CLI 状态\n"
        "  /clear    清空当前页面和上下文，回到普通问答模式\n"
        "  /help     查看帮助\n"
        "  /exit     退出\n"
        "普通问答模式下也可以直接输入问题。\n"
    )


def print_summary(summary: SummaryResult) -> None:
    print(f"\n标题: {summary.title}")
    print(f"语言: {summary.language}")
    print(f"整体总结: {summary.overall_summary}\n")
    print("段落总结:")
    for index, section in enumerate(summary.sections, start=1):
        print(f"{index}. {section.heading}")
        print(f"   {section.summary}")
    if summary.key_takeaways:
        print("\n关键结论:")
        for item in summary.key_takeaways:
            print(f"- {item}")


def print_tree(node: FallbackTree, indent: int = 0) -> None:
    prefix = "  " * indent
    print(f"{prefix}- {node.label}")
    if node.summary:
        print(f"{prefix}  {node.summary}")
    for child in node.children:
        print_tree(child, indent + 1)


def print_graph(graph: StructureGraph) -> None:
    print(f"\n结构图类型: {graph.type}")
    print("\nMermaid:")
    print(graph.mermaid)
    print("\nFallback tree:")
    print_tree(graph.fallback_tree)


def require_page(state: CliState) -> bool:
    if state.page is None:
        print("当前没有有效网页地址。请先使用 /page 输入网址。")
        return False
    return True


def run_summary(orchestrator: AIOrchestrator, state: CliState) -> None:
    if not require_page(state):
        return
    language = input("总结语言 auto/zh/en/es/fr [auto]: ").strip() or "auto"
    response = orchestrator.generate_summary(
        page=state.page,
        target_language=language,
        request_source="cli",
    )
    state.context_id = response.context_id
    state.summary_id = response.summary_id
    state.last_trace_id = response.trace_id
    print_summary(response.summary)
    print(f"trace_id={response.trace_id}")


def run_summary_stream(orchestrator: AIOrchestrator, state: CliState) -> None:
    if not require_page(state):
        return
    language = input("总结语言 auto/zh/en/es/fr [auto]: ").strip() or "auto"
    print("\n流式总结:")
    for event in orchestrator.stream_summary(
        page=state.page,
        target_language=language,
        request_source="cli",
    ):
        if event.event == "item":
            kind = event.data.get("type")
            if kind == "title":
                print(f"\n标题: {event.data.get('text', '')}")
            elif kind == "overall_summary":
                print(f"\n整体总结: {event.data.get('text', '')}")
            elif kind == "section":
                print(f"\n{int(event.data.get('index', 0)) + 1}. {event.data.get('heading', '')}")
                print(f"   {event.data.get('summary', '')}")
            elif kind == "key_takeaway":
                print(f"- {event.data.get('text', '')}")
        elif event.event == "final":
            state.context_id = event.data.get("context_id")
            state.summary_id = event.data.get("summary_id")
            state.last_trace_id = event.trace_id
        elif event.event == "usage":
            data = event.data
            print(f"\ntokens={data.get('input_tokens', 0)}/{data.get('output_tokens', 0)}/{data.get('total_tokens', 0)}")
        elif event.event == "error":
            print(f"\n错误: {event.data.get('message', 'stream failed')}")


def run_graph(orchestrator: AIOrchestrator, state: CliState) -> None:
    if not require_page(state):
        return
    if not state.context_id or not state.summary_id:
        print("请先使用 /summery 生成页面总结，再生成结构图。")
        return
    response = orchestrator.generate_graph(
        context_id=state.context_id,
        summary_id=state.summary_id,
        request_source="cli",
    )
    state.graph_id = response.graph_id
    state.last_trace_id = response.trace_id
    print_graph(response.graph)
    print(f"trace_id={response.trace_id}")


def run_page_qa(orchestrator: AIOrchestrator, state: CliState, question: str) -> None:
    if not require_page(state):
        return
    if not state.context_id:
        bootstrap = orchestrator.generate_summary(page=state.page, target_language="auto", request_source="cli")
        state.context_id = bootstrap.context_id
        state.summary_id = bootstrap.summary_id
    response = orchestrator.answer_question(context_id=state.context_id, question=question, request_source="cli")
    state.last_trace_id = response.trace_id
    print(f"\n回答: {response.answer.answer}")
    print(f"trace_id={response.trace_id}")


def run_page_qa_stream(orchestrator: AIOrchestrator, state: CliState, question: str) -> None:
    if not require_page(state):
        return
    if not state.context_id:
        bootstrap = orchestrator.generate_summary(page=state.page, target_language="auto", request_source="cli")
        state.context_id = bootstrap.context_id
        state.summary_id = bootstrap.summary_id
    print("\n回答: ", end="", flush=True)
    for event in orchestrator.qa_stream(context_id=state.context_id, question=question, request_source="cli"):
        if event.event == "delta":
            print(event.data.get("content", ""), end="", flush=True)
        elif event.event == "final":
            state.last_trace_id = event.trace_id
        elif event.event == "usage":
            data = event.data
            print(f"\n\ntokens={data.get('input_tokens', 0)}/{data.get('output_tokens', 0)}/{data.get('total_tokens', 0)}")
        elif event.event == "error":
            print(f"\n错误: {event.data.get('message', 'stream failed')}")


@log_tracer("qa.answer", task_type="qa", path="/cli/qa", method="CLI")
def run_direct_qa(config, runner: PydanticAIAgentRunner, question: str) -> None:
    model_config = model_config_from_app("qa", config.ai.qa)
    # output_rules=[
    #             "Return only the final answer text.",
    #             "Do not return JSON.",
    #             "Do not include field names.",
    #             f"Try to keep the answer around 2048 tokens.",
    #             "If a complete answer would exceed the output limit, summarize first while preserving the core conclusion, key evidence, and important uncertainty.",
    # ]
    # result: PydanticAIResult = runner.run(
    #     model_config=model_config,
    #     output_type=QAOutput,
    #     system_prompt=build_system_prompt_test(agent_name="QA Assistant", task_rules=(), output_rules=output_rules),
    #     user_prompt=question,
    # )
    qa_agent = QAAgent(model_config=model_config, ai_runner=runner)
    result: QAResult = qa_agent.run(content="", question=question, qa_messages=[])
    print(f"\n回答: {result.answer}")
    # print(
    #     f"tokens={result.usage.input_tokens}/{result.usage.output_tokens}/{result.usage.total_tokens}"
    # )


@log_tracer("qa.stream", task_type="qa", path="/cli/qa/stream", method="CLI")
def run_direct_qa_stream(config, runner: PydanticAIAgentRunner, question: str) -> None:
    model_config = model_config_from_app("qa", config.ai.qa)
    print("\n回答: ", end="", flush=True)
    final_usage = None
    for event in runner.run_text_stream(
        model_config=model_config,
        system_prompt=build_system_prompt_test("Ainsight QAAgent"),
        user_prompt=question,
    ):
        if event.event == "delta":
            print(event.content, end="", flush=True)
        elif event.event == "final":
            final_usage = event.usage
    if final_usage:
        print(
            f"\n\ntokens={final_usage.input_tokens}/{final_usage.output_tokens}/{final_usage.total_tokens}"
        )


def print_state(state: CliState) -> None:
    print("\n当前状态:")
    print(f"- mode: {'page' if state.page else 'qa'}")
    print(f"- url: {state.page.url if state.page else '-'}")
    print(f"- title: {state.page.title if state.page else '-'}")
    print(f"- context_id: {state.context_id or '-'}")
    print(f"- summary_id: {state.summary_id or '-'}")
    print(f"- graph_id: {state.graph_id or '-'}")
    print(f"- last_trace_id: {state.last_trace_id or '-'}")


def main() -> None:
    config = load_config()
    # Former test-only wiring, retained for reference (not a CLI mode):
    # from app.services.observability.langfuse_client import InMemoryLangfuseEventSink
    # recorder = InMemoryLangfuseEventSink()
    # configure_observability(
    #     config.observability, langfuse_sink=recorder, synchronous_langfuse=True,
    # )
    try:
        configure_observability(config.observability)
        with observability_scope(request_source="cli"):
            _run_cli(config)
    finally:
        shutdown_observability()


def _run_cli(config: AppConfig) -> None:
    runner = PydanticAIAgentRunner()
    orchestrator = build_orchestrator(config)
    state = CliState()

    print("Ainsight Agent Debug CLI")
    print_help()
    prompt_for_page(state)

    while True:
        try:
            command = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if not command:
            continue
        try:
            if command in {"/exit", "exit", "quit"}:
                print("bye")
                return
            if command == "/help":
                print_help()
            elif command == "/page":
                prompt_for_page(state)
            elif command in {"/summery", "/summary"}:
                run_summary(orchestrator, state)
            elif command in {"/summery-stream", "/summary-stream"}:
                run_summary_stream(orchestrator, state)
            elif command == "/graph":
                run_graph(orchestrator, state)
            elif command == "/ask":
                question = input("请输入问题: ").strip()
                if not question:
                    print("问题不能为空。")
                elif state.page:
                    run_page_qa(orchestrator, state, question)
                else:
                    run_direct_qa(config, runner, question)
            elif command == "/ask-stream":
                question = input("请输入问题: ").strip()
                if not question:
                    print("问题不能为空。")
                elif state.page:
                    run_page_qa_stream(orchestrator, state, question)
                else:
                    run_direct_qa_stream(config, runner, question)
            elif command == "/state":
                print_state(state)
            elif command == "/clear":
                state = CliState()
                print("已清空当前上下文，进入普通问答模式。")
            elif state.page:
                run_page_qa(orchestrator, state, command)
            else:
                run_direct_qa(config, runner, command)
        except AinsightError as exc:
            print(f"错误: {exc.error_code} {exc.message}")
        except Exception as exc:
            print(f"执行失败: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
