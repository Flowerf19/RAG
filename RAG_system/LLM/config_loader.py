from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional


def _repo_root() -> Path:
    # RAG_system/LLM -> repo root is two levels up
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        print("Warning: PyYAML is not installed. Cannot load YAML configuration.")
        return {}

    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if isinstance(data, dict):
                return data
            return {}
    except yaml.YAMLError as e:
        print(f"Warning: Error parsing YAML file {path}: {e}")
        return {}
    except IOError as e:
        print(f"Warning: Could not read file {path}: {e}")
        return {}


def _require(cfg: Dict[str, Any], dotted_key: str):
    node: Any = cfg
    for k in dotted_key.split("."):
        if not isinstance(node, dict) or k not in node:
            raise KeyError(f"Missing required key `{dotted_key}` in config/app.yaml")
        node = node[k]
    return node


_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def get_config() -> Dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    app_yaml = _repo_root() / "config" / "app.yaml"
    cfg = _load_yaml(app_yaml)
    if not cfg:
        raise RuntimeError(f"Missing or empty config file: {app_yaml}")

    _CONFIG_CACHE = cfg
    return cfg


def repo_path(*parts: str) -> Path:
    return _repo_root().joinpath(*parts)


def paths_data_dir() -> Path:
    rel = _require(get_config(), "paths.data_dir")
    return repo_path(str(rel))


def paths_prompt_path() -> Path:
    rel = _require(get_config(), "paths.prompt_path")
    return repo_path(str(rel))


def ui_default_backend() -> str:
    return str(_require(get_config(), "ui.default_backend"))


def resolve_gemini_settings(
    override_model: Optional[str] = None,
    override_temperature: Optional[float] = None,
    override_max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    cfg = _require(get_config(), "llm.gemini")

    # API key precedence: Streamlit secrets > env
    api_key: Optional[str] = None
    secrets_key = _require(cfg, "api_key.secrets_key")
    env_key = _require(cfg, "api_key.env")

    try:
        import streamlit as st  # type: ignore
        if hasattr(st, "secrets") and secrets_key in st.secrets:
            api_key = st.secrets[secrets_key]  # type: ignore
    except Exception:
        pass
    if not api_key:
        api_key = os.getenv(str(env_key))

    model = override_model if override_model is not None else _require(cfg, "model")
    temperature = override_temperature if override_temperature is not None else cfg.get("temperature", None)
    max_tokens = override_max_tokens if override_max_tokens is not None else cfg.get("max_tokens", None)

    return {
        "api_key": api_key,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def resolve_lmstudio_settings(
    override_model: Optional[str] = None,
    override_temperature: Optional[float] = None,
    override_top_p: Optional[float] = None,
    override_max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    cfg = _require(get_config(), "llm.lmstudio")

    base_url = _require(cfg, "base_url")
    # Allow env override for base_url if provided
    base_url = os.getenv("LMSTUDIO_BASE_URL", base_url)

    api_key_env = _require(cfg, "api_key.env")
    api_key_default = _require(cfg, "api_key.default")
    api_key = os.getenv(str(api_key_env), api_key_default)

    model_cfg = _require(cfg, "model")
    model = override_model if override_model is not None else os.getenv("LMSTUDIO_MODEL", model_cfg)

    sampling = _require(cfg, "sampling")
    temperature = override_temperature if override_temperature is not None else sampling.get("temperature", None)
    top_p = override_top_p if override_top_p is not None else sampling.get("top_p", None)
    max_tokens = override_max_tokens if override_max_tokens is not None else sampling.get("max_tokens", None)

    return {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "temperature": float(temperature) if temperature is not None else None,
        "top_p": float(top_p) if top_p is not None else None,
        "max_tokens": int(max_tokens) if max_tokens is not None else None,
    }
