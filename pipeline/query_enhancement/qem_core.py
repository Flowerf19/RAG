"""
Core orchestration logic for the Query Enhancement Module (QEM).
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore[assignment]

from .qem_lm_client import QEMLLMClient
from .qem_strategy import build_prompt
from .qem_utils import (
    clip_queries,
    deduplicate_queries,
    log_activity,
    parse_llm_list,
    summarise_queries,
)

DEFAULT_SETTINGS: Dict[str, Any] = {
    "enabled": True,
    "languages": {"vi": 2, "en": 2},
    "max_total_queries": 5,
    "log_path": "data/logs/qem_activity.json",
    "backend": None,
    "fallback_backend": "gemini",
    "llm_overrides": {},
    "system_prompt": None,
    "additional_instructions": None,
}


def _deep_merge(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, Mapping)
        ):
            base[key] = _deep_merge(copy.deepcopy(base[key]), value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_qem_settings(base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load QEM configuration from qem_config.yaml if available.
    """
    config_path = (
        Path(base_dir) / "qem_config.yaml"
        if base_dir
        else Path(__file__).with_name("qem_config.yaml")
    )
    settings = copy.deepcopy(DEFAULT_SETTINGS)
    if yaml is None:
        return settings

    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            if isinstance(data, Mapping):
                settings = _deep_merge(settings, data)
        except Exception as exc:  # pragma: no cover - logging only
            logging.getLogger(__name__).warning("Failed to load QEM config: %s", exc)
    return settings


class QueryEnhancementModule:
    """
    Main entry point for generating query variants prior to retrieval.
    """

    def __init__(
        self,
        app_config: Dict[str, Any],
        qem_settings: Optional[Dict[str, Any]] = None,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.app_config = app_config
        self.settings = qem_settings or load_qem_settings()
        self.languages = self._normalise_language_requirements(self.settings.get("languages"))
        self.log_path = Path(self.settings.get("log_path", "data/logs/qem_activity.json"))
        self.client = QEMLLMClient(app_config, self.settings, self.logger)

    @staticmethod
    def _normalise_language_requirements(config_value: Any) -> Dict[str, int]:
        if isinstance(config_value, Mapping):
            return {str(k).lower(): int(v) for k, v in config_value.items() if int(v) > 0}
        if isinstance(config_value, list):
            # Interpret list as sequence of language codes to produce at least one variant each.
            result: Dict[str, int] = {}
            for code in config_value:
                code_str = str(code).lower()
                result[code_str] = result.get(code_str, 0) + 1
            return result
        return {"vi": 2, "en": 2}

    def is_enabled(self) -> bool:
        return bool(self.settings.get("enabled", True))

    def enhance(self, user_query: str) -> List[str]:
        """
        Generate enhanced queries. On failure, return the original query only.
        """
        if not self.is_enabled():
            return [user_query]

        prompt = build_prompt(
            user_query=user_query,
            language_requirements=self.languages,
            additional_instructions=self.settings.get("additional_instructions"),
        )

        try:
            raw_output = self.client.generate_variants(prompt)
            variants = parse_llm_list(raw_output)
            queries = [user_query] + variants
            queries = deduplicate_queries(queries)
            queries = clip_queries(queries, self.settings.get("max_total_queries"))
            self._log_queries(user_query, queries, raw_output)
            self.logger.debug("QEM generated queries: %s", summarise_queries(queries))
            return queries or [user_query]
        except Exception as exc:
            self.logger.warning("QEM failed, falling back to original query: %s", exc)
            self._log_queries(user_query, [user_query], None, error=str(exc))
        return [user_query]

    def _log_queries(
        self,
        original_query: str,
        queries: List[str],
        raw_output: Optional[str],
        *,
        error: Optional[str] = None,
    ) -> None:
        payload = {
            "backend": self.client.backend,
            "query": original_query,
            "queries": queries,
            "raw_output": raw_output,
        }
        if error:
            payload["error"] = error
        log_activity(self.log_path, payload, self.logger)

        if error:
            self.logger.warning(
                "QEM fallback to original query due to error: %s", error
            )
        else:
            self.logger.info(
                "QEM expanded query '%s' -> %s",
                original_query.strip(),
                summarise_queries(queries),
            )
