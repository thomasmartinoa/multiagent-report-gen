import os
import re
from typing import TypedDict
from dotenv import load_dotenv
from tavily import TavilyClient

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

load_dotenv()
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

MODEL_NAME = "qwen3.6-16k:latest"
NUM_CTX = 16384                  
MAX_REVISIONS = 1             


def make_llm(temperature: float = 0.0) -> ChatOllama:
    """One place to configure the model — swap here to move to an API later."""
    return ChatOllama(
        model=MODEL_NAME,
        temperature=temperature,
        num_ctx=NUM_CTX,
    )


class ReportState(TypedDict):
    topic: str
    research: str
    draft: str
    critique: str
    approved: bool
    revisions: int


@tool
def web_search(query: str) -> str:
    """Search the web for current information on a topic.

    Use for facts, news, statistics, and expert opinions you don't already know.
    Returns top results with titles, URLs and content snippets.
    """
    response = tavily.search(query=query, max_results=5)
    return "\n\n---\n\n".join(
        f"TITLE: {r['title']}\nURL: {r['url']}\nCONTENT: {r['content']}"
        for r in response["results"]
    )


researcher = create_agent(
    model=make_llm(),
    tools=[web_search],
    system_prompt=(
        "You are a research specialist. Given a topic, run several targeted "
        "web searches to gather facts, figures, and differing viewpoints.\n"
        "Output organised research notes. Every fact must be followed by its "
        "source URL. Do not write prose or a report — notes only. "
        "Flag anything you could not verify."
    ),
)

writer = create_agent(
    model=make_llm(temperature=0.3),
    tools=[],   
    system_prompt=(
        "You are a report writer. Using ONLY the research notes provided, write "
        "a clear report in markdown with: a title, a 3-sentence executive "
        "summary, 3-4 themed sections, and a Sources list.\n"
        "Cite inline as [source](url). Never add facts not present in the notes. "
        "If revision feedback is provided, address every point in it."
    ),
)

critic = create_agent(
    model=make_llm(),
    tools=[],
    system_prompt=(
        "You are a critical reviewer. Check the draft against the research notes.\n"
        "Look for: claims not supported by the notes, missing citations, vague "
        "filler, poor structure.\n"
        "Respond in exactly this format:\n"
        "VERDICT: APPROVED or NEEDS_REVISION\n"
        "FEEDBACK: <specific, actionable points, or 'None'>\n"
        "Approve if the draft is solid. Do not demand perfection — only flag "
        "real problems."
    ),
)


def research_node(state: ReportState) -> dict:
    result = researcher.invoke({"messages": [
        {"role": "user", "content": f"Research this topic thoroughly: {state['topic']}"}
    ]})
    return {"research": result["messages"][-1].content}


def write_node(state: ReportState) -> dict:
    prompt = f"TOPIC: {state['topic']}\n\nRESEARCH NOTES:\n{state['research']}"
    if state.get("critique"):
        prompt += f"\n\nREVISION FEEDBACK TO ADDRESS:\n{state['critique']}"

    result = writer.invoke({"messages": [{"role": "user", "content": prompt}]})
    return {
        "draft": result["messages"][-1].content,
        "revisions": state.get("revisions", 0) + 1,
    }


def critique_node(state: ReportState) -> dict:
    prompt = (
        f"RESEARCH NOTES:\n{state['research']}\n\n"
        f"DRAFT TO REVIEW:\n{state['draft']}"
    )
    result = critic.invoke({"messages": [{"role": "user", "content": prompt}]})
    text = result["messages"][-1].content

    match = re.search(r"VERDICT:\s*(\w+)", text, re.IGNORECASE)
    approved = bool(match) and match.group(1).upper().startswith("APPROVE")
    if not match:
        approved = True  
    return {"critique": text, "approved": approved}


def should_revise(state: ReportState) -> str:
    # The guardrail: approved OR out of revisions → stop. No exceptions.
    if state["approved"] or state["revisions"] > MAX_REVISIONS:
        return "end"
    return "revise"


builder = StateGraph(ReportState)
builder.add_node("researcher", research_node)
builder.add_node("writer", write_node)
builder.add_node("critic", critique_node)

builder.add_edge(START, "researcher")
builder.add_edge("researcher", "writer")
builder.add_edge("writer", "critic")
builder.add_conditional_edges(
    "critic",
    should_revise,
    {"revise": "writer", "end": END},
)

graph = builder.compile()


if __name__ == "__main__":
    initial = {
        "topic": "How Indian startups are adopting generative AI in 2026",
        "research": "", "draft": "", "critique": "",
        "approved": False, "revisions": 0,
    }
    for step in graph.stream(initial, stream_mode="updates"):
        for node, update in step.items():
            print(f"\n{'='*60}\n▶ {node.upper()} finished\n{'='*60}")
            for k, v in update.items():
                preview = str(v)[:300]
                print(f"{k}: {preview}...")