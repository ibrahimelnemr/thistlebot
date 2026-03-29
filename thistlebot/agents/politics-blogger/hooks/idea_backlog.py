from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ....core.agent_runtime import run_tool_agent
from ....core.tools.registry import ToolRegistry
from ....integrations.wordpress.rest_client import WordPressRestClient
from ....llm.base import BaseLLMClient
from ....storage.paths import agent_dir
from ....storage.state import load_config
from ...hooks.base import HookContext, HookResult

_IDEAS_SKILL_PATH = Path(__file__).resolve().parent.parent / "skills" / "idea_scout" / "SKILL.md"
_IDEAS_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "idea_scout.md"


def _ideas_dir(agent_name: str = "blogger") -> Path:
    return agent_dir(agent_name) / "ideas"


def _ideas_index_path(agent_name: str = "blogger") -> Path:
    return _ideas_dir(agent_name) / "index.json"


def _ideas_markdown_path(agent_name: str = "blogger") -> Path:
    return _ideas_dir(agent_name) / "ideas.md"


def _ideas_batches_dir(agent_name: str = "blogger") -> Path:
    return _ideas_dir(agent_name) / "batches"


def load_idea_index(agent_name: str = "blogger") -> dict[str, Any]:
    path = _ideas_index_path(agent_name)
    if not path.exists():
        return {"version": 1, "updated_at": "", "last_refresh_at": "", "ideas": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    data.setdefault("version", 1)
    data.setdefault("updated_at", "")
    data.setdefault("last_refresh_at", "")
    if not isinstance(data.get("ideas"), list):
        data["ideas"] = []
    return data


def list_ideas(*, agent_name: str = "blogger", status: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    ideas = load_idea_index(agent_name).get("ideas", [])
    if not isinstance(ideas, list):
        return []
    filtered = [item for item in ideas if isinstance(item, dict)]
    if status:
        s = status.strip().lower()
        filtered = [item for item in filtered if str(item.get("status") or "").lower() == s]
    filtered.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return filtered[: max(limit, 0)]


def refresh_idea_backlog(
    *,
    agent_name: str = "blogger",
    client: BaseLLMClient,
    registry: ToolRegistry,
    model: str,
    topic: str,
    count: int,
    query_count: int,
    max_iterations: int,
    prefer_web: bool,
    force: bool = False,
    min_refresh_interval_minutes: int = 0,
    prompt_override: str | None = None,
) -> dict[str, Any]:
    index = load_idea_index(agent_name)
    now = datetime.now(timezone.utc)
    existing_posts = _fetch_existing_posts_snapshot(limit=80)
    existing_post_titles = [item.get("title", "") for item in existing_posts]

    if not force and min_refresh_interval_minutes > 0:
        if not _refresh_due(index.get("last_refresh_at"), min_refresh_interval_minutes, now):
            return {
                "skipped": True,
                "reason": "refresh_interval_not_elapsed",
                "created_count": 0,
                "total_ideas": len(index.get("ideas", [])),
            }

    query_plan = _build_query_plan(topic=topic, query_count=query_count, now=now)
    prompt = (prompt_override or "").strip() or _load_idea_prompt()
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": _build_user_prompt(
                topic=topic,
                count=count,
                prefer_web=prefer_web,
                query_plan=query_plan,
                existing_post_titles=existing_post_titles,
            ),
        },
    ]

    refresh_error = ""
    output_text = ""
    events: list[dict[str, Any]] = []
    parsed: list[dict[str, Any]] = []
    try:
        response = run_tool_agent(
            client=client,
            registry=registry,
            model=model,
            messages=messages,
            max_iterations=max_iterations,
            return_events=True,
        )
        output_text, events = response if isinstance(response, tuple) else (str(response), [])
        parsed = _extract_ideas(output_text)
    except Exception as exc:
        refresh_error = str(exc)

    if not parsed:
        parsed = _fallback_ideas(topic=topic, count=count, now=now)
    new_items = _to_idea_records(
        parsed,
        topic=topic,
        existing_titles={_norm_title(item.get("title")) for item in index.get("ideas", []) if isinstance(item, dict)},
        existing_post_titles=existing_post_titles,
        web_research_used=_web_research_used(events),
    )

    ideas = index.get("ideas", [])
    if not isinstance(ideas, list):
        ideas = []
    ideas.extend(new_items)

    index["ideas"] = ideas
    index["updated_at"] = now.isoformat()
    index["last_refresh_at"] = now.isoformat()

    _save_idea_index(index, agent_name=agent_name)
    _write_batch_file(
        agent_name=agent_name,
        now=now,
        topic=topic,
        query_plan=query_plan,
        output_text=output_text,
        events=events,
        ideas=new_items,
    )
    _write_ideas_markdown(index, agent_name=agent_name)

    return {
        "skipped": False,
        "created_count": len(new_items),
        "total_ideas": len(ideas),
        "web_research_used": _web_research_used(events),
        "web_tool_calls": _web_tool_calls(events),
        "existing_posts_scanned": len(existing_posts),
        "error": refresh_error,
    }


def resolve_topic_from_backlog(
    *,
    agent_name: str = "blogger",
    explicit_topic: str | None,
    default_topic: str,
) -> tuple[str, dict[str, Any] | None]:
    if isinstance(explicit_topic, str) and explicit_topic.strip():
        return explicit_topic.strip(), None

    index = load_idea_index(agent_name)
    ideas = index.get("ideas", [])
    if not isinstance(ideas, list):
        return default_topic, None

    selected = [item for item in ideas if isinstance(item, dict) and str(item.get("status") or "") == "selected"]
    selected.sort(key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""), reverse=True)
    if selected:
        picked = selected[0]
        return str(picked.get("title") or default_topic), picked

    new_ideas = [item for item in ideas if isinstance(item, dict) and str(item.get("status") or "") == "new"]
    if not new_ideas:
        return default_topic, None

    new_ideas.sort(
        key=lambda item: (float(item.get("score") or 0.0), str(item.get("created_at") or "")),
        reverse=True,
    )
    picked = new_ideas[0]
    picked["status"] = "selected"
    picked["updated_at"] = datetime.now(timezone.utc).isoformat()
    picked["last_selected_at"] = picked["updated_at"]

    index["updated_at"] = picked["updated_at"]
    _save_idea_index(index, agent_name=agent_name)
    _write_ideas_markdown(index, agent_name=agent_name)
    return str(picked.get("title") or default_topic), picked


def write_selected_idea_artifact(*, run_dir: Path, idea: dict[str, Any] | None) -> None:
    if idea is None:
        return
    artifact = {
        "id": idea.get("id"),
        "title": idea.get("title"),
        "status": idea.get("status"),
        "score": idea.get("score"),
        "tags": idea.get("tags", []),
        "source_urls": idea.get("source_urls", []),
        "selected_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "selected_idea.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")


def update_selected_idea_outcome(
    *,
    agent_name: str = "blogger",
    idea_id: str | None,
    success: bool,
    on_failure: str = "new",
) -> None:
    if not isinstance(idea_id, str) or not idea_id.strip():
        return

    index = load_idea_index(agent_name)
    ideas = index.get("ideas", [])
    if not isinstance(ideas, list):
        return

    now = datetime.now(timezone.utc).isoformat()
    changed = False
    for item in ideas:
        if not isinstance(item, dict):
            continue
        if str(item.get("id") or "") != idea_id:
            continue
        if success:
            item["status"] = "used"
            item["last_used_at"] = now
        else:
            desired = on_failure if on_failure in {"new", "archived", "selected"} else "new"
            item["status"] = desired
        item["updated_at"] = now
        changed = True
        break

    if not changed:
        return

    index["updated_at"] = now
    _save_idea_index(index, agent_name=agent_name)
    _write_ideas_markdown(index, agent_name=agent_name)


def manual_select_idea(*, agent_name: str = "blogger", idea_id: str) -> bool:
    index = load_idea_index(agent_name)
    ideas = index.get("ideas", [])
    if not isinstance(ideas, list):
        return False

    found = False
    now = datetime.now(timezone.utc).isoformat()
    for item in ideas:
        if not isinstance(item, dict):
            continue
        if str(item.get("id") or "") == idea_id:
            item["status"] = "selected"
            item["updated_at"] = now
            item["last_selected_at"] = now
            found = True
        elif str(item.get("status") or "") == "selected":
            item["status"] = "new"
            item["updated_at"] = now

    if not found:
        return False

    index["updated_at"] = now
    _save_idea_index(index, agent_name=agent_name)
    _write_ideas_markdown(index, agent_name=agent_name)
    return True


class IdeaBacklogRefreshHook:
    hook_type = "idea_backlog_refresh"

    def execute(self, context: HookContext) -> HookResult:
        cfg = context.hook_config
        topic = str(context.agent_config.get("topic") or "")
        prompt_override: str | None = None
        try:
            prompt_override = context.agent_definition.load_skill("idea_scout").instructions
        except Exception:
            prompt_override = None
        out = refresh_idea_backlog(
            agent_name=context.agent_name,
            client=context.client,
            registry=context.registry,
            model=context.model,
            topic=topic,
            count=int(cfg.get("refresh_count", 6)),
            query_count=int(cfg.get("query_count", 8)),
            max_iterations=int(cfg.get("max_iterations", 14)),
            prefer_web=bool(cfg.get("prefer_web", True)),
            min_refresh_interval_minutes=int(cfg.get("min_refresh_interval_minutes", 180)),
            prompt_override=prompt_override,
        )
        return HookResult(ok=True, data=out)


class IdeaBacklogSelectHook:
    hook_type = "idea_backlog_select"

    def execute(self, context: HookContext) -> HookResult:
        explicit = context.agent_config.get("topic_override")
        default_topic = str(context.agent_config.get("topic") or "Latest politics news")
        topic, selected = resolve_topic_from_backlog(
            agent_name=context.agent_name,
            explicit_topic=str(explicit) if isinstance(explicit, str) else None,
            default_topic=default_topic,
        )
        if context.run_dir is not None and selected is not None:
            write_selected_idea_artifact(run_dir=context.run_dir, idea=selected)
        selected_id = str(selected.get("id") or "") if isinstance(selected, dict) else ""
        return HookResult(ok=True, data={"topic": topic, "selected_idea_id": selected_id or None})


class IdeaBacklogOutcomeHook:
    hook_type = "idea_backlog_outcome"

    def execute(self, context: HookContext) -> HookResult:
        result = context.result if isinstance(context.result, dict) else {}
        success = str(result.get("status") or "") == "completed"
        selected_path = (context.run_dir / "selected_idea.json") if context.run_dir is not None else None
        selected_id: str | None = None
        if selected_path is not None and selected_path.exists():
            try:
                selected = json.loads(selected_path.read_text(encoding="utf-8"))
                if isinstance(selected, dict):
                    selected_id = str(selected.get("id") or "") or None
            except Exception:
                selected_id = None

        on_failure = str(context.hook_config.get("failure_selected_action", "new"))
        update_selected_idea_outcome(
            agent_name=context.agent_name,
            idea_id=selected_id,
            success=success,
            on_failure=on_failure,
        )
        return HookResult(ok=True, data={"updated": bool(selected_id)})


# CLI helper functions for dynamic action handlers.
def cli_refresh(
    *,
    agent_name: str,
    client: BaseLLMClient,
    registry: ToolRegistry,
    model: str,
    args: dict[str, Any],
    **_: Any,
) -> dict[str, Any]:
    return refresh_idea_backlog(
        agent_name=agent_name,
        client=client,
        registry=registry,
        model=model,
        topic=str(args.get("topic") or "Latest politics news"),
        count=int(args.get("count", 6)),
        query_count=int(args.get("query_count", 8)),
        max_iterations=int(args.get("max_iterations", 14)),
        prefer_web=bool(args.get("prefer_web", True)),
        force=bool(args.get("force", False)),
        min_refresh_interval_minutes=int(args.get("min_refresh_interval_minutes", 180)),
    )


def cli_list(*, agent_name: str, args: dict[str, Any], **_: Any) -> list[dict[str, Any]]:
    return list_ideas(agent_name=agent_name, status=args.get("status"), limit=int(args.get("limit", 20)))


def cli_select(*, agent_name: str, args: dict[str, Any], **_: Any) -> dict[str, Any]:
    idea_id = str(args.get("id") or "").strip()
    if not idea_id:
        return {"ok": False, "error": "Missing --arg id=<idea_id>"}
    ok = manual_select_idea(agent_name=agent_name, idea_id=idea_id)
    return {"ok": ok, "id": idea_id}


def _load_idea_prompt() -> str:
    if _IDEAS_SKILL_PATH.exists():
        return _read_markdown_body(_IDEAS_SKILL_PATH)
    if _IDEAS_PROMPT_PATH.exists():
        return _IDEAS_PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"Idea backlog prompt not found. Expected one of: {_IDEAS_SKILL_PATH}, {_IDEAS_PROMPT_PATH}"
    )


def _read_markdown_body(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[2].lstrip("\n")
    return text


def _save_idea_index(index: dict[str, Any], *, agent_name: str) -> None:
    path = _ideas_index_path(agent_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=2), encoding="utf-8")


def _write_ideas_markdown(index: dict[str, Any], *, agent_name: str) -> None:
    ideas = [item for item in index.get("ideas", []) if isinstance(item, dict)]
    ideas.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)

    lines: list[str] = ["# Blogger Ideas Backlog", "", f"Updated: {index.get('updated_at') or ''}", ""]

    for status in ("selected", "new", "used", "archived"):
        lines.append(f"## {status.title()}")
        bucket = [item for item in ideas if str(item.get("status") or "") == status]
        if not bucket:
            lines.append("- (none)")
            lines.append("")
            continue

        for item in bucket:
            title = str(item.get("title") or "Untitled")
            idea_id = str(item.get("id") or "")
            score = item.get("score")
            tags = item.get("tags") if isinstance(item.get("tags"), list) else []
            lines.append(f"- [{idea_id}] {title} (score={score})")
            if tags:
                lines.append(f"  tags: {', '.join(str(tag) for tag in tags[:8])}")
            urls = item.get("source_urls") if isinstance(item.get("source_urls"), list) else []
            if urls:
                lines.append(f"  sources: {len(urls)}")
        lines.append("")

    path = _ideas_markdown_path(agent_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _write_batch_file(
    *,
    agent_name: str,
    now: datetime,
    topic: str,
    query_plan: list[str],
    output_text: str,
    events: list[dict[str, Any]],
    ideas: list[dict[str, Any]],
) -> None:
    batch_dir = _ideas_batches_dir(agent_name)
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_path = batch_dir / f"{now.strftime('%Y%m%d-%H%M%S')}.md"

    lines = [
        "# Idea Refresh Batch",
        "",
        f"Timestamp: {now.isoformat()}",
        f"Topic: {topic}",
        f"Ideas created: {len(ideas)}",
        f"Web research used: {_web_research_used(events)}",
        f"Web tool calls: {_web_tool_calls(events)}",
        "",
        "## Query Plan",
        "",
    ]
    for query in query_plan:
        lines.append(f"- {query}")

    lines.extend(["", "## Ideas", ""])
    for item in ideas:
        lines.append(f"- {item.get('title')} (id={item.get('id')}, score={item.get('score')})")

    lines.extend(["", "## Raw Model Output", "", "```text", output_text[:10000], "```", ""])
    batch_path.write_text("\n".join(lines), encoding="utf-8")


def _build_user_prompt(
    *,
    topic: str,
    count: int,
    prefer_web: bool,
    query_plan: list[str],
    existing_post_titles: list[str],
) -> str:
    trimmed_titles = [title for title in existing_post_titles if isinstance(title, str) and title.strip()][:30]

    lines = [
        f"Primary topic focus: {topic}",
        f"Target number of ideas: {count}",
        f"Prefer web research: {prefer_web}",
        "",
        "This workflow is laser-focused on latest politics news stories.",
        "Use DuckDuckGo News style queries when possible (for example: 'politics' on the News tab with newest-first).",
        "Run web/news research across these query angles (or improved equivalents):",
    ]
    lines.extend([f"- {query}" for query in query_plan])
    lines.extend(
        [
            "",
            "Output JSON only with this shape:",
            '{"ideas": [{"title": "...", "angle": "...", "audience": "...", "outline": ["..."], "reasoning_summary": "...", "source_urls": ["https://..."], "score": 0.0, "tags": ["..."]}]}',
            "",
            "Constraints:",
            "- Use concrete timely titles, not generic themes.",
            "- Select specific stories that look newest and still developing.",
            "- Prefer factual, neutral reporting angles (newsroom style).",
            "- Avoid stories already covered in prior posts list below.",
            "- Prioritize recency and real-world announcements/news.",
            "- Include source URLs for each idea when available.",
            "- Keep score in range 0..1 where higher means stronger/novel/timely.",
        ]
    )

    if trimmed_titles:
        lines.extend(["", "Existing published/draft post titles to avoid repeating:"])
        lines.extend([f"- {title}" for title in trimmed_titles])

    return "\n".join(lines)


def _build_query_plan(*, topic: str, query_count: int, now: datetime) -> list[str]:
    year = now.year
    month = now.strftime("%B")
    seeds = [
        "politics",
        f"politics latest news {year}",
        f"politics breaking news last 24 hours {year}",
        f"major policy announcement today {month} {year}",
        f"election campaign updates latest news {year}",
        f"government legislation update latest news {year}",
        f"parliament or congress debate highlights {month} {year}",
        f"supreme court political impact latest report {year}",
        f"international diplomacy summit latest news {year}",
        f"political leadership statement news {month} {year}",
    ]
    if topic and topic.strip().lower() not in {"politics", "politics news", "latest politics news"}:
        seeds.insert(1, f"{topic} latest politics news {year}")
    max_count = max(1, query_count)
    return seeds[:max_count]


def _fallback_ideas(*, topic: str, count: int, now: datetime) -> list[dict[str, Any]]:
    month = now.strftime("%B")
    year = now.year
    ideas: list[dict[str, Any]] = []
    templates = [
        ("Breaking Politics Story Tracker", "Fact-first report on the single biggest politics story right now"),
        ("Top Politics Headline Today", "Neutral summary of the latest major political announcement"),
        ("Policy Story Watch", "What the newest policy or regulation headline means"),
        ("Campaign Update Brief", "Timeline and factual breakdown of the latest campaign update"),
        ("Legislation Progress Report", "What advanced, what changed, and what remains uncertain"),
        ("Diplomacy News Brief", "Objective update on the latest international political development"),
    ]
    for idx, (title_seed, angle) in enumerate(templates[: max(count, 1)], start=1):
        ideas.append(
            {
                "title": f"{title_seed}: {month} {year} Edition ({idx})",
                "angle": angle,
                "audience": "Software engineers and technical leads",
                "outline": [
                    "What happened",
                    "Who announced it and when",
                    "What is confirmed by sources",
                    "Open questions and what to watch next",
                ],
                "reasoning_summary": f"Fallback idea generated without live LLM output for topic: {topic}",
                "source_urls": [],
                "score": 0.45,
                "tags": ["politics", "policy", "breaking"],
            }
        )
    return ideas


def _extract_ideas(text: str) -> list[dict[str, Any]]:
    payload = _extract_json_payload(text)
    if isinstance(payload, dict) and isinstance(payload.get("ideas"), list):
        return [item for item in payload["ideas"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _extract_json_payload(text: str) -> Any:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\\s*", "", stripped)
    stripped = re.sub(r"\\s*```$", "", stripped)

    for candidate in (stripped,):
        try:
            return json.loads(candidate)
        except Exception:
            pass

    start_obj = stripped.find("{")
    end_obj = stripped.rfind("}")
    if 0 <= start_obj < end_obj:
        snippet = stripped[start_obj : end_obj + 1]
        try:
            return json.loads(snippet)
        except Exception:
            pass

    start_arr = stripped.find("[")
    end_arr = stripped.rfind("]")
    if 0 <= start_arr < end_arr:
        snippet = stripped[start_arr : end_arr + 1]
        try:
            return json.loads(snippet)
        except Exception:
            pass

    return {}


def _to_idea_records(
    parsed: list[dict[str, Any]],
    *,
    topic: str,
    existing_titles: set[str],
    existing_post_titles: list[str],
    web_research_used: bool,
) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc).isoformat()
    created: list[dict[str, Any]] = []

    for item in parsed:
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        norm = _norm_title(title)
        if norm in existing_titles:
            continue
        if _is_duplicate_title_against_posts(title, existing_post_titles):
            continue

        source_urls_raw = item.get("source_urls")
        source_urls: list[str] = []
        if isinstance(source_urls_raw, list):
            for url in source_urls_raw:
                if isinstance(url, str) and url.startswith(("http://", "https://")):
                    source_urls.append(url)
        source_urls = source_urls[:8]

        tags_raw = item.get("tags")
        tags: list[str] = []
        if isinstance(tags_raw, list):
            tags = [str(tag).strip() for tag in tags_raw if str(tag).strip()]

        outline_raw = item.get("outline")
        if isinstance(outline_raw, list):
            outline = [str(line).strip() for line in outline_raw if str(line).strip()]
        else:
            outline = []

        score = _score(item.get("score"))

        created.append(
            {
                "id": f"idea-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}",
                "created_at": now,
                "updated_at": now,
                "status": "new",
                "title": title,
                "angle": str(item.get("angle") or "").strip(),
                "audience": str(item.get("audience") or "").strip(),
                "outline": outline,
                "reasoning_summary": str(item.get("reasoning_summary") or "").strip(),
                "source_urls": source_urls,
                "score": score,
                "tags": tags,
                "topic": topic,
                "web_research_used": bool(web_research_used),
            }
        )
        existing_titles.add(norm)

    return created


def _fetch_existing_posts_snapshot(*, limit: int = 80) -> list[dict[str, str]]:
    cfg = load_config()
    wp = (cfg.get("integrations", {}).get("wordpress", {}) if isinstance(cfg.get("integrations", {}).get("wordpress"), dict) else {})
    token = str(wp.get("token") or "").strip()
    site = str(wp.get("blog") or "").strip()
    if not token or not site:
        return []

    try:
        client = WordPressRestClient(access_token=token)
        response = client.list_posts(site, number=max(1, min(limit, 100)), status="any")
    except Exception:
        return []

    posts_raw = response.get("posts", []) if isinstance(response, dict) else []
    if not isinstance(posts_raw, list):
        return []

    posts: list[dict[str, str]] = []
    for item in posts_raw:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("URL") or item.get("url") or "").strip()
        if title:
            posts.append({"title": title, "url": url})
    return posts


def _is_duplicate_title_against_posts(candidate_title: str, existing_post_titles: list[str]) -> bool:
    candidate_norm = _norm_title(candidate_title)
    candidate_tokens = _title_tokens(candidate_norm)
    if not candidate_tokens:
        return False

    for existing in existing_post_titles:
        existing_norm = _norm_title(existing)
        if not existing_norm:
            continue
        if candidate_norm == existing_norm:
            return True

        existing_tokens = _title_tokens(existing_norm)
        if not existing_tokens:
            continue
        overlap = len(candidate_tokens & existing_tokens) / max(len(candidate_tokens), len(existing_tokens))
        if overlap >= 0.75:
            return True

    return False


def _title_tokens(title: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9]+", title.lower()) if len(tok) > 2}


def _score(raw: Any) -> float:
    try:
        score = float(raw)
    except Exception:
        score = 0.0
    return max(0.0, min(1.0, score))


def _norm_title(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def _web_research_used(events: list[dict[str, Any]]) -> bool:
    return _web_tool_calls(events) > 0


def _web_tool_calls(events: list[dict[str, Any]]) -> int:
    count = 0
    for event in events:
        if event.get("event") != "tool_call":
            continue
        tool = str(event.get("tool") or "").lower()
        if "search" in tool or "news" in tool or "web" in tool:
            count += 1
    return count


def _refresh_due(last_refresh_at: Any, min_minutes: int, now: datetime) -> bool:
    if not isinstance(last_refresh_at, str) or not last_refresh_at.strip():
        return True
    try:
        prev = datetime.fromisoformat(last_refresh_at)
    except Exception:
        return True
    if prev.tzinfo is None:
        prev = prev.replace(tzinfo=timezone.utc)
    return now - prev >= timedelta(minutes=max(min_minutes, 0))
