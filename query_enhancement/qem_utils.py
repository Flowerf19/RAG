"""
Utility helpers for the Query Enhancement Module (QEM).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence


def normalize_query(text: str) -> str:
    """
    Normalize query text for deduplication purposes.
    """
    return re.sub(r"\s+", " ", text.strip().lower())


def deduplicate_queries(queries: Sequence[str]) -> List[str]:
    """
    Remove duplicate queries (case-insensitive, whitespace-normalised) while preserving order.
    """
    seen = set()
    result: List[str] = []
    for query in queries:
        canonical = normalize_query(query)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        result.append(query.strip())
    return result


def parse_llm_list(raw_output: str) -> List[str]:
    """
    Attempt to parse the LLM output as a list of query variants.

    Supports JSON arrays as well as simple bullet / enumerated lists.
    """
    if not raw_output:
        return []

    text = raw_output.strip()

    # Strip Markdown fences such as ```json ... ``` to allow clean JSON parsing.
    fence_match = re.match(r"^```(?:\w+)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    # Try JSON first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
    except json.JSONDecodeError:
        pass

    # Fallback: split by lines, stripping leading bullet/numbering markers
    queries: List[str] = []
    pattern = re.compile(r"^[-*\u2022\d\.\)\s]+")
    for line in text.splitlines():
        raw_line = line.strip()
        if not raw_line or raw_line.startswith("```"):
            continue
        stripped = pattern.sub("", raw_line).strip().rstrip(",")
        if stripped in {"[", "]"}:
            continue
        if stripped:
            queries.append(stripped)
    return queries


def ensure_directory(path: Path) -> None:
    """
    Ensure the directory for the provided path exists.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def log_activity(log_path: Path, payload: dict, logger: logging.Logger | None = None) -> None:
    """
    Append QEM activity data to the configured log file.
    """
    try:
        ensure_directory(log_path)
        payload = dict(payload)
        payload.setdefault("timestamp", datetime.utcnow().isoformat())
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:
        if logger:
            logger.warning("Failed to write QEM log: %s", exc)


def clip_queries(queries: Sequence[str], max_size: int | None) -> List[str]:
    """
    Limit the number of queries to max_size while preserving order.
    """
    if max_size is None or max_size <= 0:
        return list(queries)
    return list(queries)[:max_size]


def summarise_queries(queries: Iterable[str]) -> str:
    """
    Produce a concise summary string for logging/debugging.
    """
    return " | ".join(q.strip() for q in queries if q.strip())
