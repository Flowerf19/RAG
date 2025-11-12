"""
Ollama LLM Client - Local Ollama REST API wrapper

Follows the same conventions as other clients in `llm/`:
- Resolve settings via `llm.config_loader.resolve_ollama_settings`
- Load system prompt from repo prompts path when available
- Accept OpenAI-style messages and convert them to a single prompt
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from llm.base_client import BaseLLMClient

try:
	from llm.config_loader import resolve_ollama_settings, paths_prompt_path
except Exception:
	try:
		from config_loader import resolve_ollama_settings, paths_prompt_path
	except Exception:
		resolve_ollama_settings = None  # type: ignore
		paths_prompt_path = None  # type: ignore

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 180
SYSTEM_PROMPT_FILENAME = "rag_system_prompt.txt"


class OllamaClient(BaseLLMClient):
	"""Simple Ollama client consistent with the repo's client patterns."""

	def __init__(self, config: Optional[Dict[str, Any]] = None):
		# Resolve settings via config loader when available
		cfg: Dict[str, Any] = {}
		if resolve_ollama_settings is not None:
			try:
				cfg = resolve_ollama_settings(
					override_base_url=(config or {}).get("base_url") if config else None,
					override_model=(config or {}).get("model") if config else None,
					override_timeout=(config or {}).get("timeout") if config else None,
				) or {}
			except Exception:
				cfg = {}

		# Merge provided config over resolved values
		merged = {**cfg, **(config or {})}

		# defaults
		merged.setdefault("base_url", "http://localhost:11434")
		merged.setdefault("model", "gemma3n:latest")
		merged.setdefault("timeout", DEFAULT_TIMEOUT)
		merged.setdefault("temperature", None)
		merged.setdefault("max_tokens", None)

		super().__init__(merged)

		self.base_url: str = str(self.config.get("base_url"))
		self.model: str = str(self.config.get("model"))
		self.timeout: int = int(self.config.get("timeout", DEFAULT_TIMEOUT))
		self.temperature: Optional[float] = self.config.get("temperature")
		self.max_tokens: Optional[int] = self.config.get("max_tokens")

		# Load system prompt from prompts directory if available
		self._system_prompt = self._load_system_prompt()

	def _load_system_prompt(self) -> Optional[str]:
		"""Load the system prompt from the repo prompts path (if present).

		Uses `llm.config_loader.paths_prompt_path()` which points to the prompts
		directory configured in `config/app.yaml` (paths.prompt_path).
		"""
		try:
			if paths_prompt_path is None:
				return None
			prompt_dir = paths_prompt_path()
			f = Path(prompt_dir) / SYSTEM_PROMPT_FILENAME
			if f.exists():
				try:
					return f.read_text(encoding="utf-8")
				except Exception:
					return None
			return None
		except Exception:
			return None

	def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
		"""Convert OpenAI-style messages into a single prompt string.

		The conversion uses labeled sections similar to `GeminiClient` to keep
		the prompt explicit: System/User/Assistant. If a repo system prompt
		exists and messages do not contain a system role, it will be prepended.
		"""
		parts: List[str] = []

		has_system = any(m.get("role") == "system" for m in messages)
		if self._system_prompt and not has_system:
			parts.append(f"System: {self._system_prompt.strip()}")

		for m in messages:
			role = m.get("role", "")
			content = m.get("content", "")
			if not content:
				continue
			if role == "system":
				parts.append(f"System: {content}")
			elif role == "user":
				parts.append(f"User: {content}")
			elif role == "assistant":
				parts.append(f"Assistant: {content}")
			else:
				parts.append(content)

		return "\n".join(parts)

	def _parse_generate_response(self, resp: requests.Response) -> str:
		"""Robustly parse Ollama generate responses.

		Handles single JSON responses and NDJSON/streaming lines that Ollama
		can emit. For NDJSON we collect the `response` chunks and join them.
		Fallback to raw text.
		"""
		raw = resp.text or ""

		# Try direct JSON first
		try:
			data = resp.json()
		except Exception:
			data = None

		if isinstance(data, dict):
			# common shapes
			if "response" in data and isinstance(data["response"], dict):
				return str(data["response"].get("content") or data["response"].get("text") or "").strip()
			if "text" in data:
				return str(data.get("text", "")).strip()
			if "result" in data:
				return str(data.get("result", "")).strip()
			if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
				first = data["choices"][0]
				if isinstance(first, dict):
					msg = first.get("message") or first.get("text") or first.get("content")
					if isinstance(msg, dict):
						return str(msg.get("content") or "").strip()
					return str(msg or "").strip()
			return str(data).strip()

		# NDJSON / streaming
		if "\n" in raw:
			pieces: List[str] = []
			for line in raw.splitlines():
				line = line.strip()
				if not line:
					continue
				try:
					obj = json.loads(line)
				except Exception:
					continue

				if isinstance(obj, dict):
					resp_chunk = None
					if "response" in obj:
						resp_chunk = obj.get("response")
					elif "text" in obj:
						resp_chunk = obj.get("text")
					elif "result" in obj:
						resp_chunk = obj.get("result")

					if isinstance(resp_chunk, str) and resp_chunk:
						pieces.append(resp_chunk)

					if obj.get("done") is True or obj.get("done_reason") is not None:
						break

			if pieces:
				return "".join(pieces).strip()

		return raw.strip()

	def generate(
		self,
		messages: List[Dict[str, str]],
		temperature: Optional[float] = None,
		max_tokens: Optional[int] = None,
		**kwargs,
	) -> str:
		prompt = self._convert_messages_to_prompt(messages)

		payload: Dict[str, Any] = {"model": self.model, "prompt": prompt}
		params: Dict[str, Any] = {}
		if temperature is not None:
			params["temperature"] = temperature
		elif self.temperature is not None:
			params["temperature"] = self.temperature

		if max_tokens is not None:
			params["max_tokens"] = max_tokens
		elif self.max_tokens is not None:
			params["max_tokens"] = self.max_tokens

		if params:
			payload["parameters"] = params

		url = self.base_url.rstrip("/") + "/api/generate"
		try:
			resp = requests.post(url, json=payload, timeout=self.timeout)
			resp.raise_for_status()
			return self._parse_generate_response(resp)
		except Exception as e:
			logger.exception("Ollama generate error: %s", e)
			raise RuntimeError(f"Ollama API error: {e}")

	def is_available(self) -> bool:
		try:
			url = self.base_url.rstrip("/") + "/api/models"
			resp = requests.get(url, timeout=5)
			resp.raise_for_status()
			try:
				data = resp.json()
			except Exception:
				return True

			if isinstance(data, list) and len(data) > 0:
				return True
			if isinstance(data, dict) and len(data.keys()) > 0:
				return True
		except Exception:
			# fallback to light generate probe
			try:
				probe_payload = {"model": self.model, "prompt": "Hello", "parameters": {"max_tokens": 1}}
				probe = requests.post(self.base_url.rstrip("/") + "/api/generate", json=probe_payload, timeout=5)
				probe.raise_for_status()
				if (probe.text or "").strip():
					return True
			except Exception:
				return False

		return False

