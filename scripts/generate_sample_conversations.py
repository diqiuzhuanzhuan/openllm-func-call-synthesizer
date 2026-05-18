"""Generate sample conversation data where the assistant actually invokes tools.

Architecture
------------
- An in-process FastMCP server exposes a small set of representative tools.
- fastmcp.Client connects to it directly (no HTTP, no external server needed).
- litellm calls gpt-4o-mini with the full tool-schema list so the model can
  generate tool_calls.
- ToolCallingLoop drives the agentic loop: call LLM → execute tools via MCP →
  call LLM again until a final plain-text answer is produced.
- The resulting messages (including tool_calls and tool results) are saved as
  JSONL training data.
"""

import asyncio
import json
from pathlib import Path

import litellm
from dotenv import load_dotenv
from rich.console import Console
from rich.rule import Rule

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import fastmcp  # noqa: E402

from openllm_func_call_synthesizer.core.synthesizer import ToolCallingLoop  # noqa: E402
from openllm_func_call_synthesizer.utils import convert_to_openai_tools  # noqa: E402

MODEL = "gpt-4o-mini"
console = Console()

# ---------------------------------------------------------------------------
# In-process FastMCP server — tools are executed here, no HTTP needed
# ---------------------------------------------------------------------------

mcp_server = fastmcp.FastMCP("synthesizer-tools")


@mcp_server.tool()
def get_weather(city: str) -> str:
    """Get current weather conditions for a city."""
    data = {
        "Tokyo": "Cloudy, 18 °C, 80 % chance of rain — bring an umbrella.",
        "London": "Overcast, 12 °C, light drizzle expected.",
        "Beijing": "Sunny, 22 °C, clear skies all day.",
        "New York": "Partly cloudy, 15 °C, pleasant.",
    }
    return data.get(city, f"{city}: Sunny, 20 °C.")


@mcp_server.tool()
def search_flights(origin: str, destination: str, date: str) -> str:
    """Search available flights between two cities on a given date (YYYY-MM-DD)."""
    return (
        f"Flights {origin} → {destination} on {date}:\n"
        f"  CA837  08:00 → 14:30  ¥4,200\n"
        f"  MU551  13:00 → 19:40  ¥3,850  ← cheapest\n"
        f"  BA038  22:00 → 06:10+1 ¥3,600  (overnight)"
    )


@mcp_server.tool()
def get_calendar_events(date: str) -> str:
    """List calendar events for a given date (YYYY-MM-DD)."""
    return f"Your schedule on {date}:\n  10:00  Standup (30 min)\n  14:00  Project review (1 hr)\n  17:30  Team dinner"


@mcp_server.tool()
def set_reminder(title: str, datetime: str) -> str:
    """Create a reminder with a title at a specific date-time string."""
    return f"✓ Reminder set: '{title}' at {datetime}."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCENARIOS = [
    "Check the weather in Tokyo. Should I bring an umbrella today?",
    "Find me the cheapest flight from Beijing to London on 2026-03-10.",
    (
        "Set a reminder for my team meeting at 3 PM tomorrow (2026-03-06), "
        "then show me what else is on my calendar that day."
    ),
]


async def build_tools_schema() -> list[dict]:
    """List tools from the in-process server and return them in OpenAI format."""
    async with fastmcp.Client(mcp_server) as client:
        mcp_tools = await client.list_tools()
    return convert_to_openai_tools(mcp_tools)["tools"]


def make_llm_callable(tools_schema: list[dict]):
    """Return a sync callable that sends messages to gpt-4o-mini with tool schemas."""

    def call(messages: list[dict]) -> dict:
        response = litellm.completion(
            model=MODEL,
            messages=messages,
            tools=tools_schema,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        # model_dump() serialises tool_calls to plain dicts
        return msg.model_dump()

    return call


def print_conversation(idx: int, scenario: str, turns: list[dict]) -> None:
    console.print(Rule(f"[bold cyan]Sample {idx}[/bold cyan]"))
    console.print(f"[dim]Scenario:[/dim] {scenario}\n")
    for turn in turns:
        role = turn.get("role", "")
        if role == "system":
            continue
        elif role == "user":
            console.print(f"[bold green]USER:[/bold green] {turn.get('content', '')}\n")
        elif role == "assistant":
            tool_calls = turn.get("tool_calls") or []
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    console.print(f"[bold yellow]TOOL_CALL:[/bold yellow] {fn.get('name')}({fn.get('arguments')})\n")
            content = turn.get("content")
            if content:
                console.print(f"[bold blue]ASSISTANT:[/bold blue] {content}\n")
        elif role == "tool":
            console.print(
                f"[bold magenta]TOOL_RESULT[/bold magenta] "
                f"[dim]({turn.get('tool_call_id')})[/dim]: {turn.get('content', '')}\n"
            )


async def main() -> None:
    tools_schema = await build_tools_schema()
    tool_names = [t["function"]["name"] for t in tools_schema]
    console.print(f"[bold]Available tools:[/bold] {tool_names}\n")

    llm_callable = make_llm_callable(tools_schema)
    samples = []

    for idx, scenario in enumerate(SCENARIOS, 1):
        loop = ToolCallingLoop(
            llm_callable=llm_callable,
            mcp_client=fastmcp.Client(mcp_server),
            max_iterations=8,
        )
        turns = await loop.run(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. "
                        "Always use the available tools to answer the user's request — "
                        "never make up information."
                    ),
                },
                {"role": "user", "content": scenario},
            ]
        )
        print_conversation(idx, scenario, turns)
        samples.append({"scenario": scenario, "turns": turns})

    output_path = Path(__file__).resolve().parents[1] / "data" / "sample_conversations.jsonl"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    console.print(f"[bold green]Saved {len(samples)} samples → {output_path}[/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
