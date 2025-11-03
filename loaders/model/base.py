"""
Shared base utilities for loader data models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class LoaderBaseModel:
    """
    Lightweight base dataclass that adds a dict helper.

    Child models inherit dataclass behaviour and gain a ``to_dict`` helper
    that recursively converts the dataclass tree into primitve
    serialisable structures.  This keeps the loader models decoupled from
    pydantic or other heavy dependencies while remaining ergonomic.
    """

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
