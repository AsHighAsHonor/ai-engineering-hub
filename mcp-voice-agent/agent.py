"""
MCP voice agent that routes queries either to Firecrawl web search or to Supabase via MCP.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
from typing import Any, Callable, List, Optional

import inspect
import httpx
from dotenv import load_dotenv
from firecrawl import FirecrawlApp, ScrapeOptions
from pydantic_ai.mcp import MCPServerStdio

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.agents import APIConnectOptions
from livekit.plugins import assemblyai, openai, silero, cartesia

# ------------------------------------------------------------------------------
# Configuration & Logging
# ------------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
SUPABASE_TOKEN = os.getenv("SUPABASE_ACCESS_TOKEN")

if not FIRECRAWL_API_KEY:
    logger.error("FIRECRAWL_API_KEY is not set in environment.")
    raise EnvironmentError("Please set FIRECRAWL_API_KEY env var.")

if not SUPABASE_TOKEN:
    logger.error("SUPABASE_ACCESS_TOKEN is not set in environment.")
    raise EnvironmentError("Please set SUPABASE_ACCESS_TOKEN env var.")

firecrawl_app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)


def _py_type(schema: dict) -> Any:
    """Convert JSON schema types into Python typing annotations."""
    t = schema.get("type")
    mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "object": dict,
    }

    if isinstance(t, list):
        if "array" in t:
            return List[_py_type(schema.get("items", {}))]
        t = t[0]

    if isinstance(t, str) and t in mapping:
        return mapping[t]
    if t == "array":
        return List[_py_type(schema.get("items", {}))]

    return Any


def schema_to_google_docstring(description: str, schema: dict) -> str:
    """
    Generate a Google-style docstring section from a JSON schema.
    """
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    lines = [description or "", "Args:"]

    for name, prop in props.items():
        t = prop.get("type", "Any")
        if isinstance(t, list):
            if "array" in t:
                subtype = prop.get("items", {}).get("type", "Any")
                py_type = f"List[{subtype.capitalize()}]"
            else:
                py_type = t[0].capitalize()
        elif t == "array":
            subtype = prop.get("items", {}).get("type", "Any")
            py_type = f"List[{subtype.capitalize()}]"
        else:
            py_type = t.capitalize()

        if name not in required:
            py_type = f"Optional[{py_type}]"

        desc = prop.get("description", "")
        lines.append(f"    {name} ({py_type}): {desc}")

    return "\n".join(lines)


@function_tool
async def firecrawl_search(
    context: RunContext,
    query: str,
    limit: int = 5
) -> List[str]:
    """
    Search the web via Firecrawl.

    Args:
        context (RunContext): LiveKit runtime context.
        query (str): Search query string.
        limit (int): Maximum pages to crawl.

    Returns:
        List[str]: Raw page contents.
    """
    # Normalize limit: LiveKit may pass null; default to 5 and ensure int
    try:
        limit = int(limit) if limit else 5
        if limit <= 0:
            limit = 5
    except Exception:
        limit = 5
    url = f"https://www.google.com/search?q={query}"
    logger.debug("Starting Firecrawl for URL: %s (limit=%d)", url, limit)

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: firecrawl_app.crawl_url(
                url,
                limit=limit,
                scrape_options=ScrapeOptions(formats=["markdown"])
            )
        )

        # Normalize Firecrawl SDK return types into List[str]
        def _item_to_text(item: Any) -> str:
            if isinstance(item, str):
                return item
            if isinstance(item, dict):
                return (
                    item.get("markdown")
                    or item.get("content")
                    or item.get("html")
                    or item.get("rawHtml")
                    or json.dumps(item)
                )
            # SDK objects: try common attributes
            for attr in ("markdown", "content", "html", "rawHtml", "text"):
                v = getattr(item, attr, None)
                if isinstance(v, str):
                    return v
            return str(item)

        pages: List[str] = []
        # Case 1: direct list
        if isinstance(result, list):
            pages = [_item_to_text(it) for it in result]
        # Case 2: dict-like
        elif isinstance(result, dict):
            data = (
                result.get("data")
                or result.get("items")
                or result.get("results")
                or result.get("pages")
            )
            if isinstance(data, list):
                pages = [_item_to_text(it) for it in data]
        else:
            # Case 3: SDK object
            data = (
                getattr(result, "data", None)
                or getattr(result, "items", None)
                or getattr(result, "results", None)
                or getattr(result, "pages", None)
            )
            if isinstance(data, list):
                pages = [_item_to_text(it) for it in data]
            else:
                # Some SDKs return a job/status object; if it already has data attr, prefer it
                job_data = getattr(result, "data", None)
                if isinstance(job_data, list):
                    pages = [_item_to_text(it) for it in job_data]

        logger.info("Firecrawl returned %d pages", len(pages))
        return pages
    except Exception as e:
        logger.error("Firecrawl search failed: %s", e, exc_info=True)
        return []


async def build_livekit_tools(server: MCPServerStdio) -> List[Callable]:
    """
    Build LiveKit tools from a Supabase MCP server.
    """
    tools: List[Callable] = []
    all_tools = await server.list_tools()
    logger.info("Found %d MCP tools", len(all_tools))

    for td in all_tools:
        if td.name == "deploy_edge_function":
            logger.warning("Skipping tool %s", td.name)
            continue

        # MCP python-sdk Tool exposes JSON Schema on `inputSchema` per spec (camelCase).
        # Older code used `parameters_json_schema` which no longer exists on `mcp_types.Tool`.
        schema = copy.deepcopy(getattr(td, "inputSchema", {}) or {})

        if td.name == "list_tables":
            props = schema.setdefault("properties", {})
            props["schemas"] = {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "default": []
            }
            schema["required"] = [r for r in schema.get("required", []) if r != "schemas"]

        props = schema.get("properties", {})
        required = set(schema.get("required", []))

        def make_proxy(
            tool_def=td,
            _props=props,
            _required=required,
            _schema=schema
        ) -> Callable:
            async def proxy(context: RunContext, **kwargs):
                # Convert None → [] for array params
                for k, v in list(kwargs.items()):
                    if ((_props[k].get("type") == "array"
                         or "array" in (_props[k].get("type") or []))
                            and v is None):
                        kwargs[k] = []

                # Use the MCPServer direct call helper; it expects args dict, not `arguments=`.
                response = await server.direct_call_tool(tool_def.name, kwargs or {})

                # Normalize typical MCP results: may be str, list, or binary
                if isinstance(response, list):
                    return response
                if isinstance(response, (bytes, bytearray)):
                    return response
                if isinstance(response, str):
                    try:
                        return json.loads(response)
                    except json.JSONDecodeError:
                        return response
                return response

            # Build signature from schema
            params = [
                inspect.Parameter("context", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=RunContext)
            ]
            ann = {"context": RunContext}

            for name, ps in _props.items():
                default = ps.get("default", inspect._empty if name in required else None)
                params.append(
                    inspect.Parameter(
                        name,
                        inspect.Parameter.KEYWORD_ONLY,
                        annotation=_py_type(ps),
                        default=default,
                    )
                )
                ann[name] = _py_type(ps)

            proxy.__signature__ = inspect.Signature(params)
            proxy.__annotations__ = ann
            proxy.__name__ = tool_def.name
            # Build docstring, but clamp length to reduce prompt bloat (improves TTFB for local LLMs).
            _doc = schema_to_google_docstring(tool_def.description or "", _schema)
            max_chars = int(os.getenv("TOOL_DOC_MAX_CHARS", "1200"))
            if len(_doc) > max_chars:
                _doc = _doc[:max_chars] + "\n\n(…truncated for latency)"

            # Special-case extremely long descriptions like Supabase docs GraphQL schema.
            if tool_def.name == "search_docs":
                _doc = (
                    "Search the Supabase documentation using GraphQL. Provide a `graphql_query` string. "
                    "Return matching docs and metadata."
                )
            proxy.__doc__ = _doc
            return function_tool(proxy)

        tools.append(make_proxy())

    return tools

def _build_cartesia_tts_from_env():
    """Create a Cartesia TTS instance using env vars with sensible defaults."""
    if not os.getenv("CARTESIA_API_KEY"):
        logger.error("CARTESIA_API_KEY is not set in environment.")
        raise EnvironmentError("Please set CARTESIA_API_KEY env var.")

    model = os.getenv("CARTESIA_MODEL", "sonic-2")
    voice = os.getenv("CARTESIA_VOICE")
    language = os.getenv("CARTESIA_LANGUAGE")
    speed_str = os.getenv("CARTESIA_SPEED")
    emotion = os.getenv("CARTESIA_EMOTION")

    kwargs = {"model": model}
    if voice:
        kwargs["voice"] = voice
    if language:
        kwargs["language"] = language
    if speed_str:
        try:
            kwargs["speed"] = float(speed_str)
        except ValueError:
            logger.warning("Invalid CARTESIA_SPEED value: %s", speed_str)
    if emotion:
        kwargs["emotion"] = emotion

    return cartesia.TTS(**kwargs)

async def _ollama_direct_warmup(model: str, base_url: str) -> None:
    """Prime the local Ollama model via /v1/chat/completions to avoid LiveKit client timeouts.
    We call Ollama directly with long HTTP timeouts so the model can load fully before use.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "stream": False,
        "temperature": 0,
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "ping"}],
    }
    headers = {"Content-Type": "application/json", "Authorization": "Bearer ollama"}

    timeout = httpx.Timeout(connect=30.0, read=300.0, write=300.0, pool=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            logger.info("Ollama direct warm-up success: %s", resp.status_code)
        except Exception as e:
            logger.warning("Ollama direct warm-up skipped: %s", e)

class TimeoutLLM(openai.LLM):
    """OpenAI LLM wrapper that forces a larger conn_options.timeout for every chat call."""
    def __init__(self, *args, request_timeout: float | None = None, **kwargs):
        # Default to env `LLM_REQUEST_TIMEOUT` (seconds) or 60s if unset
        self._default_conn_options = APIConnectOptions(
            timeout=float(os.getenv("LLM_REQUEST_TIMEOUT", request_timeout or 60.0)),
        )
        super().__init__(*args, **kwargs)

    def chat(self, *, chat_ctx, tools=None, conn_options=None, **kwargs):
        # If caller didn't pass conn_options, use our default with larger timeout
        if conn_options is None:
            conn_options = self._default_conn_options
        else:
            # Ensure a sensible timeout even if the caller passed a struct without timeout
            try:
                if getattr(conn_options, "timeout", None) is None:
                    conn_options = APIConnectOptions(
                        max_retry=getattr(conn_options, "max_retry", 3),
                        retry_interval=getattr(conn_options, "retry_interval", 2.0),
                        timeout=self._default_conn_options.timeout,
                    )
            except Exception:
                conn_options = self._default_conn_options
        return super().chat(chat_ctx=chat_ctx, tools=tools, conn_options=conn_options, **kwargs)

async def entrypoint(ctx: JobContext) -> None:
    """
    Main entrypoint for the LiveKit agent.
    """
    await ctx.connect()
    server = MCPServerStdio(
        "npx",
        args=["-y", "@supabase/mcp-server-supabase@latest", "--access-token", SUPABASE_TOKEN],
    )
    await server.__aenter__()

    try:
        supabase_tools = await build_livekit_tools(server)
        tools = [firecrawl_search] + supabase_tools

        agent = Agent(
            instructions=(
                "You can either perform live web searches via `firecrawl_search` or "
                "database queries via Supabase MCP tools. "
                "Choose the appropriate tool based on whether the user needs fresh web data "
                "(news, external facts) or internal Supabase data."
            ),
            tools=tools,
        )

        # Some versions of livekit-plugins-assemblyai don't expose `word_boost`.
        # Build STT kwargs dynamically for maximum compatibility.
        _stt_kwargs = {}
        try:
            if "word_boost" in inspect.signature(assemblyai.STT).parameters:
                _stt_kwargs["word_boost"] = ["Supabase"]
        except Exception:
            pass

        model_name = "qwen2.5:7b-instruct"
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

        # Direct warm-up to avoid read timeouts during first token generation
        await _ollama_direct_warmup(model=model_name, base_url=base_url)

        llm = TimeoutLLM(
            model=model_name,
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),  # any non-empty string for Ollama
            base_url=base_url,
            temperature=0.4,
            timeout=httpx.Timeout(connect=30.0, read=300.0, write=300.0, pool=30.0),
            request_timeout=float(os.getenv("LLM_REQUEST_TIMEOUT", 60.0)),
        )

        session = AgentSession(
            vad=silero.VAD.load(min_silence_duration=0.1),
            stt=assemblyai.STT(**_stt_kwargs),
            llm=llm,
            tts=_build_cartesia_tts_from_env(),           # ← 已切 Cartesia
            use_tts_aligned_transcript=True,              # ← 启用对齐转写
        )

        # Warm up the local Ollama model to avoid initial cold-start timeouts
        try:
            warmup_ctx = agent.new_context()
            async with llm.chat(chat_ctx=warmup_ctx) as stream:
                await stream.send_user_text("ping")
                # drain the first chunk to ensure model is loaded
                async for _ in stream:
                    break
            logger.info("Ollama warm-up completed")
        except Exception as _e:
            logger.warning("Ollama warm-up skipped: %s", _e)

        await session.start(agent=agent, room=ctx.room)
        await session.generate_reply(instructions="Hello! How can I assist you today?")

        # Keep the session alive until cancelled
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Session cancelled, shutting down.")

    finally:
        await server.__aexit__(None, None, None)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
