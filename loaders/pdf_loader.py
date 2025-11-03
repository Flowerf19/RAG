"""
High level PDF loader that converts PDF-Extract-Kit outputs into the internal
`PDFDocument` structure used by the RAG pipeline.

The loader favours the enhanced extraction pipeline from `PDFLoaders`
but gracefully falls back to a lightweight PyMuPDF based extractor when the
heavier models are unavailable.  Blocks produced by the loader carry rich
metadata so downstream chunkers can reason about tables, figures, headings
and inline formulae.
"""

from __future__ import annotations

import copy
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .model import (
    Block,
    PDFDocument,
    PDFPage,
    TableBlock,
    TableCell,
    TableRow,
    TableSchema,
)

logger = logging.getLogger(__name__)


BBox = Tuple[float, float, float, float]


class PDFLoader:
    """
    Wraps the PDF-Extract-Kit pipeline and exposes a `load` method that returns
    a normalized `PDFDocument`.
    """

    CATEGORY_TYPE_MAP: Dict[str, str] = {
        "text": "text",
        "plain_text": "text",
        "paragraph": "text",
        "body": "text",
        "title": "heading",
        "subtitle": "heading",
        "heading": "heading",
        "section": "heading",
        "list": "list",
        "bullet_list": "list",
        "figure": "figure",
        "figure_caption": "caption",
        "table_caption": "caption",
        "table_footnote": "footnote",
        "caption": "caption",
        "inline": "formula",
        "isolated": "formula",
        "isolate_formula": "formula",
        "formula": "formula",
        "formula_caption": "caption",
        "footnote": "footnote",
        "table": "table",
    }

    def __init__(
        self,
        *,
        config_path: Optional[Path | str] = None,
        cache_dir: Optional[Path | str] = None,
        prefer_pdf_extract_kit: bool = True,
        use_cache_outputs: bool = False,
    ) -> None:
        self.workspace_root = Path(__file__).resolve().parents[1]
        self.pdf_kit_root = self.workspace_root / "PDFLoaders"

        default_config = self.pdf_kit_root / "project" / "pdf2markdown" / "configs" / "pdf2markdown.yaml"
        self.config_path = Path(config_path) if config_path else default_config if default_config.exists() else None

        self.cache_dir = Path(cache_dir) if cache_dir else (self.workspace_root / "data" / "pdf_loader_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.prefer_pdf_extract_kit = prefer_pdf_extract_kit
        self.use_cache_outputs = use_cache_outputs

        self._pdf_task = None
        self._task_instances: Optional[Dict[str, Any]] = None
        self._pdf_config: Optional[Dict[str, Any]] = None

    @classmethod
    def create_default(cls) -> "PDFLoader":
        return cls()

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def load(self, pdf_path: str | Path) -> PDFDocument:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if self.prefer_pdf_extract_kit:
            try:
                return self._extract_with_pdf_extract_kit(pdf_path)
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning(
                    "Falling back to PyMuPDF extraction for %s due to error: %s",
                    pdf_path.name,
                    exc,
                )

        return self._extract_with_pymupdf(pdf_path)

    def load_directory(self, directory: str | Path) -> List[PDFDocument]:
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        pdf_files = sorted(directory.glob("*.pdf"))
        return [self.load(pdf_file) for pdf_file in pdf_files]

    # --------------------------------------------------------------------- #
    # PDF-Extract-Kit integration
    # --------------------------------------------------------------------- #
    def _ensure_pdf_extract_pipeline(self) -> None:
        if self._pdf_task is not None:
            return
        if not self.config_path or not Path(self.config_path).exists():
            raise FileNotFoundError(
                "PDF-Extract-Kit configuration not found. "
                "Provide a valid `config_path` when constructing PDFLoader."
            )

        try:
            from PDFLoaders.pdf_extract_kit.utils.config_loader import (
                initialize_tasks_and_models,
                load_config,
            )
            from PDFLoaders.project.pdf2markdown.scripts.pdf2markdown import PDF2MARKDOWN
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "PDF-Extract-Kit dependencies are not available in this environment."
            ) from exc

        raw_config = load_config(str(self.config_path))
        if not raw_config or "tasks" not in raw_config:
            raise ValueError("Invalid PDF-Extract-Kit configuration.")

        patched_config = self._patch_config_paths(raw_config)
        task_instances = initialize_tasks_and_models(patched_config)

        layout_model = task_instances.get("layout_detection").model if "layout_detection" in task_instances else None
        formula_det_model = task_instances.get("formula_detection").model if "formula_detection" in task_instances else None
        formula_rec_model = task_instances.get("formula_recognition").model if "formula_recognition" in task_instances else None
        ocr_model = task_instances.get("ocr").model if "ocr" in task_instances else None

        self._pdf_task = PDF2MARKDOWN(layout_model, formula_det_model, formula_rec_model, ocr_model)
        self._task_instances = task_instances
        self._pdf_config = patched_config

    def _extract_with_pdf_extract_kit(self, pdf_path: Path) -> PDFDocument:
        self._ensure_pdf_extract_pipeline()
        assert self._pdf_task is not None  # for type-checkers

        save_dir = None
        if self.use_cache_outputs:
            save_dir = self.cache_dir / "pdf_extract_outputs"
            save_dir.mkdir(parents=True, exist_ok=True)

        results = self._pdf_task.process(
            str(pdf_path),
            save_dir=str(save_dir) if save_dir else None,
            visualize=False,
            merge2markdown=False,
        )

        return self._build_document(pdf_path, results, extraction_source="pdf_extract_kit")

    def _patch_config_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        patched = copy.deepcopy(config)
        base_dir = self.pdf_kit_root

        # Standardise output directory
        patched["outputs"] = str((self.cache_dir / "pdf_extract_outputs").resolve())

        path_like_keys = ("path", "dir", "cfg")
        for task_cfg in patched.get("tasks", {}).values():
            model_cfg = task_cfg.get("model_config", {})
            for key, value in list(model_cfg.items()):
                if isinstance(value, str) and any(snippet in key.lower() for snippet in path_like_keys):
                    candidate = Path(value)
                    if not candidate.is_absolute():
                        candidate = (base_dir / candidate).resolve()
                    model_cfg[key] = str(candidate)
        return patched

    # --------------------------------------------------------------------- #
    # Fallback extractor (PyMuPDF)
    # --------------------------------------------------------------------- #
    def _extract_with_pymupdf(self, pdf_path: Path) -> PDFDocument:
        try:
            import fitz  # type: ignore
        except ImportError as exc:  # pragma: no cover - fallback
            raise RuntimeError(
                "PyMuPDF (`fitz`) is required for the fallback extractor."
            ) from exc

        doc = fitz.open(str(pdf_path))
        pages: List[Dict[str, Any]] = []

        try:
            for page_index, page in enumerate(doc):
                blocks = page.get_text("blocks")
                layout: List[Dict[str, Any]] = []
                for block in blocks:
                    if len(block) < 5:
                        continue
                    x0, y0, x1, y1, text = block[:5]
                    text_value = (text or "").strip()
                    if not text_value:
                        continue
                    layout.append(
                        {
                            "category_type": "text",
                            "poly": [x0, y0, x1, y0, x1, y1, x0, y1],
                            "text": text_value,
                            "score": None,
                        }
                    )

                pages.append(
                    {
                        "layout_dets": layout,
                        "page_info": {
                            "page_no": page_index,
                            "width": page.rect.width,
                            "height": page.rect.height,
                        },
                    }
                )
        finally:
            doc.close()

        return self._build_document(pdf_path, pages, extraction_source="pymupdf")

    # --------------------------------------------------------------------- #
    # Document construction
    # --------------------------------------------------------------------- #
    def _build_document(
        self,
        pdf_path: Path,
        extraction_results: Sequence[Dict[str, Any]],
        *,
        extraction_source: str,
    ) -> PDFDocument:
        doc_id = self._make_doc_id(pdf_path)
        pages: List[PDFPage] = []
        warnings: List[str] = []

        for page_index, page_result in enumerate(extraction_results):
            page_info = page_result.get("page_info") or {}
            width = float(page_info.get("width", 0))
            height = float(page_info.get("height", 0))

            blocks, tables, figures, page_warnings = self._build_page_blocks(
                doc_id,
                pdf_path,
                page_index,
                page_result.get("layout_dets") or [],
                page_width=width,
                page_height=height,
            )
            page_text = "\n\n".join(block.text for block in blocks if block.text)

            page = PDFPage(
                page_number=page_index + 1,
                width=width,
                height=height,
                blocks=blocks,
                tables=tables,
                figures=figures,
                text=page_text,
                source={
                    "file_path": str(pdf_path),
                    "doc_id": doc_id,
                    "page_number": page_index + 1,
                    "page_width": width,
                    "page_height": height,
                    "extraction_source": extraction_source,
                },
                warnings=page_warnings,
            )

            pages.append(page)
            warnings.extend(page_warnings)

        meta = {
            "doc_id": doc_id,
            "file_name": pdf_path.name,
            "file_path": str(pdf_path),
            "file_extension": pdf_path.suffix.lower(),
            "file_size": pdf_path.stat().st_size if pdf_path.exists() else None,
            "page_count": len(pages),
            "extraction_source": extraction_source,
        }

        return PDFDocument(
            file_path=str(pdf_path),
            pages=pages,
            meta=meta,
            warnings=warnings,
        )

    def _build_page_blocks(
        self,
        doc_id: str,
        pdf_path: Path,
        page_index: int,
        layout_dets: Sequence[Dict[str, Any]],
        *,
        page_width: float,
        page_height: float,
    ) -> Tuple[List[Block], List[TableSchema], List[Block], List[str]]:
        entries: List[Dict[str, Any]] = []
        warnings: List[str] = []

        for idx, raw in enumerate(layout_dets):
            category = str(raw.get("category_type") or raw.get("type") or "text").lower()
            category = category.replace(" ", "_")
            bbox = self._poly_to_bbox(raw.get("poly"))
            text_value = raw.get("text") or raw.get("latex") or ""

            entry = {
                "index": idx,
                "category": category,
                "bbox": bbox,
                "text": text_value,
                "score": raw.get("score"),
                "center_x": (bbox[0] + bbox[2]) / 2 if bbox else 0.0,
                "center_y": (bbox[1] + bbox[3]) / 2 if bbox else 0.0,
                "raw": raw,
            }
            entries.append(entry)

        table_entries = [entry for entry in entries if entry["category"] == "table"]
        text_candidates = [entry for entry in entries if entry["category"] != "table"]

        used_text_indices: Set[int] = set()
        table_blocks: Dict[int, TableBlock] = {}
        table_schemas: List[TableSchema] = []

        for table_order, table_entry in enumerate(table_entries, start=1):
            table_block, table_schema, used_indices, table_warnings = self._create_table_block(
                doc_id,
                page_index,
                table_order,
                table_entry,
                text_candidates,
            )
            table_blocks[table_entry["index"]] = table_block
            if table_schema:
                table_schemas.append(table_schema)
            used_text_indices.update(used_indices)
            warnings.extend(table_warnings)

        blocks: List[Block] = []
        figures: List[Block] = []
        block_counter = 0

        for entry in entries:
            idx = entry["index"]
            category = entry["category"]

            if category == "table":
                block_counter += 1
                table_block = table_blocks.get(idx)
                if table_block is None:
                    warnings.append(
                        f"page {page_index+1}: table entry {idx} missing schema; skipped."
                    )
                    block_counter -= 1
                    continue

                block_id = self._make_block_id(doc_id, page_index, block_counter)
                table_block.block_id = block_id
                table_block.stable_id = block_id
                table_block.metadata["block_index"] = block_counter
                blocks.append(table_block)
                continue

            if idx in used_text_indices:
                continue

            block = self._create_text_block(
                doc_id,
                page_index,
                block_counter + 1,
                entry,
                page_width=page_width,
                page_height=page_height,
            )
            if not block:
                continue

            block_counter += 1
            block.block_id = self._make_block_id(doc_id, page_index, block_counter)
            block.stable_id = block.block_id
            block.metadata["block_index"] = block_counter
            blocks.append(block)

            if block.block_type == "figure":
                figures.append(block)

        return blocks, table_schemas, figures, warnings

    def _create_table_block(
        self,
        doc_id: str,
        page_index: int,
        table_order: int,
        table_entry: Dict[str, Any],
        text_candidates: Sequence[Dict[str, Any]],
    ) -> Tuple[TableBlock, Optional[TableSchema], Set[int], List[str]]:
        bbox = table_entry["bbox"]
        tolerance = max(6.0, (bbox[3] - bbox[1]) * 0.02) if bbox else 6.0

        cell_entries = [
            candidate
            for candidate in text_candidates
            if candidate["text"]
            and self._bbox_contains(bbox, candidate["bbox"], tolerance=tolerance)
        ]
        used_indices = {candidate["index"] for candidate in cell_entries}

        schema = self._build_table_schema(
            doc_id,
            page_index,
            table_order,
            table_entry,
            cell_entries,
        )

        cell_provenance = [
            {
                "row": cell.row_index,
                "col": cell.col_index,
                "value": cell.value,
                "bbox": cell.bbox,
            }
            for row in (schema.rows if schema else [])
            for cell in row.cells
        ]

        table_id = schema.id if schema else f"{doc_id}_p{page_index+1:04d}_table{table_order:03d}"

        metadata = {
            "block_type": "table",
            "category": table_entry["category"],
            "bbox": bbox,
            "score": table_entry.get("score"),
            "page_number": page_index + 1,
            "table_order": table_order,
            "table_id": table_id,
            "table_schema": schema,
            "table_payload": schema,
            "cell_provenance": cell_provenance,
        }

        embedding_text = ""
        if schema and not schema.is_empty():
            embedding_text = schema.embedding_text()
            metadata["embedding_text"] = embedding_text
            metadata["cells"] = schema.cell_count()
            metadata["header_included"] = bool(schema.header)

        table_block = TableBlock(
            block_id="",
            page_number=page_index + 1,
            text=embedding_text,
            bbox=bbox,
            block_type="table",
            category=table_entry["category"],
            score=table_entry.get("score"),
            metadata=metadata,
            text_source="pdf_extract_kit",
            table=schema,
        )

        warnings: List[str] = []
        if schema and schema.is_empty():
            warnings.append(f"page {page_index+1}: table {table_order} has no cell content.")

        return table_block, schema, used_indices, warnings

    def _build_table_schema(
        self,
        doc_id: str,
        page_index: int,
        table_order: int,
        table_entry: Dict[str, Any],
        cell_entries: Sequence[Dict[str, Any]],
    ) -> Optional[TableSchema]:
        table_id = f"{doc_id}_p{page_index+1:04d}_table{table_order:03d}"
        bbox = table_entry["bbox"]

        if not cell_entries:
            return TableSchema(
                id=table_id,
                page_number=page_index + 1,
                header=[],
                rows=[],
                bbox=bbox,
            )

        sorted_cells = sorted(cell_entries, key=lambda c: (c["center_y"], c["center_x"]))
        tolerance = max(6.0, (bbox[3] - bbox[1]) * 0.02) if bbox else 6.0

        row_groups: List[Dict[str, Any]] = []
        for cell in sorted_cells:
            assigned = False
            for group in row_groups:
                if abs(group["avg_y"] - cell["center_y"]) <= tolerance:
                    group["cells"].append(cell)
                    group["avg_y"] = (
                        (group["avg_y"] * (len(group["cells"]) - 1)) + cell["center_y"]
                    ) / len(group["cells"])
                    assigned = True
                    break
            if not assigned:
                row_groups.append({"avg_y": cell["center_y"], "cells": [cell]})

        row_groups.sort(key=lambda g: g["avg_y"])

        table_rows: List[TableRow] = []
        for row_idx, group in enumerate(row_groups):
            ordered_cells = sorted(group["cells"], key=lambda c: c["center_x"])
            table_cells: List[TableCell] = []
            for col_idx, cell in enumerate(ordered_cells):
                value = (cell["text"] or "").strip()
                table_cells.append(
                    TableCell(
                        row_index=row_idx,
                        col_index=col_idx,
                        value=value,
                        bbox=cell["bbox"],
                        confidence=cell.get("score"),
                    )
                )
            table_rows.append(TableRow(index=row_idx, cells=table_cells))

        if not table_rows:
            return TableSchema(
                id=table_id,
                page_number=page_index + 1,
                header=[],
                rows=[],
                bbox=bbox,
            )

        if len(table_rows) == 1:
            single_row = table_rows[0]
            data_rows = [
                TableRow(
                    index=0,
                    cells=[
                        TableCell(
                            row_index=0,
                            col_index=col_idx,
                            value=cell.value,
                            bbox=cell.bbox,
                            confidence=cell.confidence,
                        )
                        for col_idx, cell in enumerate(single_row.cells)
                    ],
                )
            ]
            schema = TableSchema(
                id=table_id,
                page_number=page_index + 1,
                header=[],
                rows=data_rows,
                bbox=bbox,
            )
            schema.build_markdown()
            return schema

        header = [cell.value for cell in table_rows[0].cells]
        data_rows: List[TableRow] = []
        for new_idx, row in enumerate(table_rows[1:], start=0):
            transformed_cells = [
                TableCell(
                    row_index=new_idx,
                    col_index=col_idx,
                    value=cell.value,
                    bbox=cell.bbox,
                    confidence=cell.confidence,
                )
                for col_idx, cell in enumerate(row.cells)
            ]
            data_rows.append(TableRow(index=new_idx, cells=transformed_cells))

        schema = TableSchema(
            id=table_id,
            page_number=page_index + 1,
            header=header,
            rows=data_rows,
            bbox=bbox,
        )
        schema.build_markdown()
        return schema

    def _create_text_block(
        self,
        doc_id: str,
        page_index: int,
        block_index: int,
        entry: Dict[str, Any],
        *,
        page_width: float,
        page_height: float,
    ) -> Optional[Block]:
        category = entry["category"]
        block_type = self.CATEGORY_TYPE_MAP.get(category, "text")
        bbox = entry["bbox"] or (0.0, 0.0, 0.0, 0.0)
        raw = entry.get("raw") or {}

        text_value = entry.get("text") or ""
        if block_type == "formula":
            text_value = raw.get("latex", text_value)
        if block_type == "figure" and not text_value:
            text_value = "[Figure]"

        text_value = (text_value or "").strip()
        if not text_value and block_type not in {"figure", "formula"}:
            return None

        metadata = {
            "block_type": block_type,
            "category": category,
            "bbox": bbox,
            "score": entry.get("score"),
            "page_number": page_index + 1,
            "page_width": page_width,
            "page_height": page_height,
            "source": "pdf_extract_kit",
        }
        if raw.get("latex"):
            metadata["latex"] = raw["latex"]

        block = Block(
            block_id="",
            page_number=page_index + 1,
            text=text_value,
            bbox=bbox,
            block_type=block_type,
            category=category,
            score=entry.get("score"),
            metadata=metadata,
            text_source="pdf_extract_kit",
        )

        return block

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _poly_to_bbox(self, poly: Optional[Sequence[float]]) -> BBox:
        if not poly or len(poly) < 4:
            return (0.0, 0.0, 0.0, 0.0)
        xs = poly[0::2]
        ys = poly[1::2]
        return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))

    def _bbox_contains(self, outer: Optional[BBox], inner: Optional[BBox], *, tolerance: float = 0.0) -> bool:
        if outer is None or inner is None:
            return False
        ox0, oy0, ox1, oy1 = outer
        ix0, iy0, ix1, iy1 = inner
        return (
            ix0 >= ox0 - tolerance
            and iy0 >= oy0 - tolerance
            and ix1 <= ox1 + tolerance
            and iy1 <= oy1 + tolerance
        )

    def _make_doc_id(self, pdf_path: Path) -> str:
        stat = pdf_path.stat()
        fingerprint = f"{pdf_path.name}|{stat.st_size}|{int(stat.st_mtime)}"
        return hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:16]

    def _make_block_id(self, doc_id: str, page_index: int, block_index: int) -> str:
        return f"{doc_id}_p{page_index+1:04d}_b{block_index:04d}"
