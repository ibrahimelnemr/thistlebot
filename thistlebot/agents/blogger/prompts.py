from __future__ import annotations

from pathlib import Path


_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _prompt_text(name: str) -> str:
    path = _PROMPTS_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8")


def research_messages(topic: str) -> list[dict[str, str]]:
    """Build the messages list for the research step."""
    return [
        {
            "role": "system",
            "content": _prompt_text("researcher"),
        },
        {
            "role": "user",
            "content": f"Research the following topic for a blog article:\n\n{topic}",
        },
    ]


def draft_messages(topic: str, research: str) -> list[dict[str, str]]:
    """Build the messages list for the drafting step."""
    return [
        {
            "role": "system",
            "content": _prompt_text("writer"),
        },
        {
            "role": "user",
            "content": (
                f"Write a blog article on this topic: {topic}\n\n"
                f"Use the following research notes as your source material:\n\n"
                f"{research}"
            ),
        },
    ]


def publish_messages(draft: str, site: str, post_status: str) -> list[dict[str, str]]:
    """Build the messages list for the edit-and-publish step."""
    return [
        {
            "role": "system",
            "content": _prompt_text("publisher"),
        },
        {
            "role": "user",
            "content": (
                "Review, finalize, and publish the following draft article:\n\n"
                f"site={site}\nstatus={post_status}\n\n{draft}"
            ),
        },
    ]
