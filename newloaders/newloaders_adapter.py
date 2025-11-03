"""
Provides a backwards-compatible factory for the updated PDF loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from loaders.pdf_loader import PDFLoader


def create_compatible_loader(
    *,
    config_path: Optional[str | Path] = None,
    prefer_pdf_extract_kit: bool = True,
) -> PDFLoader:
    """
    Factory mirroring the previous `newloaders` API.

    Args:
        config_path: Optional override for the PDF-Extract-Kit configuration.
        prefer_pdf_extract_kit: Skip directly to the lightweight fallback when
            set to ``False``.
    """
    return PDFLoader(
        config_path=config_path,
        prefer_pdf_extract_kit=prefer_pdf_extract_kit,
    )
