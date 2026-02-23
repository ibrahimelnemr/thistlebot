from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .chat_client import stream_chat

Speaker = Literal["agent_a", "agent_b"]


class MeetingState(TypedDict):
    history: list[dict[str, str]]
    next_speaker: Speaker
    turn_count: int
    max_turns: int
    stagnation_count: int
    directive: str | None
    session_id: str
    gateway_url: str
    model_a: str
    model_b: str
    should_stop: bool


@dataclass
class MeetingConfig:
    gateway_url: str
    session_id: str
    model_a: str
    model_b: str
    starter: str
    max_turns: int = 0


def _speaker_goal(speaker: Speaker) -> str:
    if speaker == "agent_a":
        return "You are the proposer. Add a concrete idea, plan, or decision each turn."
    return "You are the critic-editor. Improve, challenge, or tighten the last proposal with specifics."


def _build_prompt(state: MeetingState, speaker: Speaker) -> str:
    history = state["history"]
    recent = history[-12:]
    transcript = "\n".join(f"{item['speaker']}: {item['content']}" for item in recent)

    directive = state.get("directive")
    directive_text = f"\nGuardrail directive: {directive}" if directive else ""

    return (
        f"{_speaker_goal(speaker)}\n"
        "Do not ask for more details or clarification."
        " Move the conversation forward with concrete content."
        " Keep response concise and specific.\n"
        "Allowed moves:"
        " (1) add a new actionable point,"
        " (2) improve the previous point,"
        " (3) summarize a decision and next step.\n"
        f"{directive_text}\n\n"
        "Conversation so far:\n"
        f"{transcript}\n\n"
        f"Respond now as {speaker}:"
    )


def _similarity(left: str, right: str) -> float:
    left_clean = left.strip().lower()
    right_clean = right.strip().lower()
    if not left_clean or not right_clean:
        return 0.0
    return SequenceMatcher(None, left_clean, right_clean).ratio()


def _is_stagnating(history: list[dict[str, str]]) -> bool:
    if len(history) < 4:
        return False

    latest = history[-1]
    previous_same_speaker = history[-3]
    previous_turn = history[-2]

    same_speaker_sim = _similarity(latest["content"], previous_same_speaker["content"])
    adjacent_turn_sim = _similarity(latest["content"], previous_turn["content"])

    return same_speaker_sim > 0.86 or adjacent_turn_sim > 0.9


def run_meeting_graph(
    config: MeetingConfig,
    on_turn_start: Callable[[Speaker], None],
    on_turn_chunk: Callable[[str], None],
    on_turn_end: Callable[[], None],
) -> None:
    max_turns = config.max_turns if config.max_turns > 0 else 10_000_000

    def agent_turn(state: MeetingState, speaker: Speaker) -> dict:
        model = state["model_a"] if speaker == "agent_a" else state["model_b"]
        prompt = _build_prompt(state, speaker)

        on_turn_start(speaker)
        content = ""
        for chunk in stream_chat(
            gateway_url=state["gateway_url"],
            messages=[{"role": "user", "content": prompt}],
            model=model,
            session_id=f"{state['session_id']}-{speaker}",
        ):
            content += chunk
            on_turn_chunk(chunk)
        on_turn_end()

        updated_history = state["history"] + [{"speaker": speaker, "content": content.strip()}]
        next_speaker: Speaker = "agent_b" if speaker == "agent_a" else "agent_a"

        return {
            "history": updated_history,
            "next_speaker": next_speaker,
            "turn_count": state["turn_count"] + 1,
        }

    def agent_a_node(state: MeetingState) -> dict:
        return agent_turn(state, "agent_a")

    def agent_b_node(state: MeetingState) -> dict:
        return agent_turn(state, "agent_b")

    def guard_node(state: MeetingState) -> dict:
        stagnation_count = state["stagnation_count"]
        if _is_stagnating(state["history"]):
            stagnation_count += 1
        else:
            stagnation_count = max(0, stagnation_count - 1)

        directive = None
        if stagnation_count >= 1:
            directive = "Do not ask questions. Provide 2 concrete options and choose one."
        if stagnation_count >= 3:
            directive = "End with a concise decision and 2 actionable next steps."

        should_stop = state["turn_count"] >= state["max_turns"] or stagnation_count >= 5

        return {
            "stagnation_count": stagnation_count,
            "directive": directive,
            "should_stop": should_stop,
        }

    def route_after_guard(state: MeetingState) -> str:
        if state["should_stop"]:
            return END
        return state["next_speaker"]

    graph_builder = StateGraph(MeetingState)
    graph_builder.add_node("agent_a", agent_a_node)
    graph_builder.add_node("agent_b", agent_b_node)
    graph_builder.add_node("guard", guard_node)

    graph_builder.add_edge(START, "agent_b")
    graph_builder.add_edge("agent_a", "guard")
    graph_builder.add_edge("agent_b", "guard")
    graph_builder.add_conditional_edges("guard", route_after_guard, {"agent_a": "agent_a", "agent_b": "agent_b", END: END})

    graph = graph_builder.compile()

    initial_state: MeetingState = {
        "history": [{"speaker": "agent_a", "content": config.starter}],
        "next_speaker": "agent_b",
        "turn_count": 0,
        "max_turns": max_turns,
        "stagnation_count": 0,
        "directive": None,
        "session_id": config.session_id,
        "gateway_url": config.gateway_url,
        "model_a": config.model_a,
        "model_b": config.model_b,
        "should_stop": False,
    }

    graph.invoke(initial_state)
