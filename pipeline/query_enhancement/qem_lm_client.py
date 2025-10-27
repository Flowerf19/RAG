"""
LLM client wrapper for the Query Enhancement Module.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from llm.LLM_API import call_gemini
from llm.LLM_LOCAL import call_lmstudio


class QEMLLMClient:
    """
    Thin wrapper around existing LLM helpers with backend selection logic.
    """

    def __init__(
        self,
        app_config: Dict[str, Any],
        qem_config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.app_config = app_config
        self.qem_config = qem_config
        self.logger = logger or logging.getLogger(__name__)
        self.backend = self._resolve_backend()
        self.system_prompt = qem_config.get(
            "system_prompt",
            (
                "You rewrite and translate user search queries to maximise retrieval "
                "recall without changing intent. Respond exactly as instructed."
            ),
        )
        self.overrides = qem_config.get("llm_overrides", {})

    def _resolve_backend(self) -> str:
        """
        Determine which backend to use for QEM requests.
        """
        override_backend = self.qem_config.get("backend")
        if override_backend:
            return override_backend

        ui_cfg = self.app_config.get("ui", {})
        return str(ui_cfg.get("default_backend", self.qem_config.get("fallback_backend", "gemini"))).lower()

    def generate_variants(self, prompt: str) -> str:
        """
        Execute the enhancement prompt against the configured backend.

        Returns:
            Raw LLM text output (to be parsed by the caller).
        """
        if self.backend == "gemini":
            return self._call_gemini(prompt)
        if self.backend == "lmstudio":
            return self._call_lmstudio(prompt)

        raise ValueError(f"Unsupported QEM backend: {self.backend}")

    def _call_gemini(self, prompt: str) -> str:
        """
        Call Gemini through the shared helper.
        """
        temperature = self.overrides.get("temperature")
        max_tokens = self.overrides.get("max_tokens")
        model = self.overrides.get("model")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = call_gemini(messages, model_name=model, temperature=temperature, max_tokens=max_tokens)
        if self.logger:
            self.logger.debug("Gemini response: %s", response)
        return response

    def _call_lmstudio(self, prompt: str) -> str:
        """
        Call LM Studio through the shared helper.
        """
        temperature = self.overrides.get("temperature")
        top_p = self.overrides.get("top_p")
        max_tokens = self.overrides.get("max_tokens")
        model = self.overrides.get("model")

        # call_lmstudio expects floats or None; guard conversions
        kwargs: Dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        if top_p is not None:
            kwargs["top_p"] = float(top_p)
        if max_tokens is not None:
            kwargs["max_tokens"] = int(max_tokens)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = call_lmstudio(messages, model=model, **kwargs)
        if self.logger:
            self.logger.debug("LM Studio response: %s", response)
        return response
