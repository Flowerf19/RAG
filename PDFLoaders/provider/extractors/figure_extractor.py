"""
Figure Extractor - Extract and group images with OCR and optional ML-based detection
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import logging
import os
import fitz

if TYPE_CHECKING:
    from .ocr_extractor import OCRExtractor

logger = logging.getLogger(__name__)

# Try to import PDF-Extract-Kit tasks (optional, for advanced figure/formula detection)
try:
    from PDFLoaders.pdf_extract_kit.tasks import (
        LayoutDetectionTask, 
        FormulaDetectionTask,
        _layout_available,
        _formula_available
    )
    LAYOUT_DETECTION_AVAILABLE = _layout_available
    FORMULA_DETECTION_AVAILABLE = _formula_available
except ImportError:
    LayoutDetectionTask = None
    FormulaDetectionTask = None
    LAYOUT_DETECTION_AVAILABLE = False
    FORMULA_DETECTION_AVAILABLE = False


class FigureExtractor:
    """Extract figures (images/diagrams) from PDF with OCR support and optional ML-based detection"""
    
    def __init__(self, ocr_extractor: Optional['OCRExtractor'] = None, use_ml_detection: bool = False):
        """
        Args:
            ocr_extractor: Optional OCR extractor for figure text extraction
            use_ml_detection: Use PDF-Extract-Kit's ML-based layout and formula detection
        """
        self.ocr_extractor = ocr_extractor
        self.use_ml_detection = use_ml_detection and (LAYOUT_DETECTION_AVAILABLE or FORMULA_DETECTION_AVAILABLE)
        self.layout_detector = None
        self.formula_detector = None
        
        if self.use_ml_detection:
            try:
                logger.info("ML-based figure detection available (layout + formula detection)")
                # LayoutDetectionTask and FormulaDetectionTask require model instances
                # For now, keep them optional and log availability
                logger.info("ML detection tasks available but not initialized (requires model configuration)")
                self.use_ml_detection = False  # Disable until models are configured
            except Exception as e:
                logger.warning(f"Could not initialize ML detectors: {e}")
                self.use_ml_detection = False
        
        if not self.use_ml_detection:
            logger.debug("Using spatial-grouping figure extraction (default)")
    
    def extract(self, fitz_doc, page, page_num: int) -> List[Dict[str, Any]]:
        """
        Extract and group images into figures (with optional OCR)
        
        Args:
            fitz_doc: PyMuPDF document object (for OCR extraction)
            page: PyMuPDF page object
            page_num: Page number (0-indexed)
            
        Returns:
            List of figure dictionaries with bbox, image_count, text (OCR), etc.
        """
        # Get filename for logging
        doc_name = Path(fitz_doc.name).name if fitz_doc.name else "unknown"
        
        try:
            # Get all images on page
            image_list = page.get_images()
            
            if not image_list:
                return []
            
            # Extract bbox for each image
            images_with_bbox = []
            for img_index, img in enumerate(image_list):
                try:
                    # Get image bbox from page
                    xref = img[0]
                    bbox_list = page.get_image_rects(xref)
                    
                    if bbox_list:
                        for bbox in bbox_list:
                            images_with_bbox.append({
                                'xref': xref,
                                'bbox': list(bbox),  # [x0, y0, x1, y1]
                                'image_index': img_index
                            })
                except Exception as e:
                    logger.debug(f"Could not get bbox for image {img_index}: {e}")
                    continue
            
            if not images_with_bbox:
                return []
            
            # Remove duplicate bboxes (same image referenced multiple times)
            unique_images = []
            seen_bboxes = set()
            for img in images_with_bbox:
                bbox_tuple = tuple(img['bbox'])
                if bbox_tuple not in seen_bboxes:
                    unique_images.append(img)
                    seen_bboxes.add(bbox_tuple)
            
            # Group nearby images into figures
            figures = self._group_images(unique_images, page)
            
            # Extract OCR text from each figure region (if OCR available)
            if self.ocr_extractor:
                for figure in figures:
                    # Use region-based OCR instead of individual image OCR
                    figure_bbox = figure['bbox']
                    text = self._extract_text_from_figure_region(fitz_doc, page, figure_bbox)
                    
                    figure['text'] = text.strip()
                    if figure['text']:
                        logger.debug(f"Figure OCR extracted {len(figure['text'])} chars from region {figure_bbox} - {doc_name}")
                    else:
                        # Only warn if figure is large enough to potentially contain text
                        figure_area = (figure_bbox[2] - figure_bbox[0]) * (figure_bbox[3] - figure_bbox[1])
                        if figure_area > 5000:  # Only warn for substantial figures
                            logger.warning(f"Figure on page {page_num+1} has no OCR text (area: {figure_area:.0f}, images: {figure.get('image_count', 0)}) - {doc_name}")
                        else:
                            logger.debug(f"Small figure on page {page_num+1} has no text (area: {figure_area:.0f}) - {doc_name}")
            else:
                # No OCR - set empty text
                for figure in figures:
                    figure['text'] = ""
                logger.debug(f"OCR extractor not available, skipping figure text extraction - {doc_name}")
            
            return figures
            
        except Exception as e:
            logger.warning(f"Figure extraction failed for page {page_num+1} in {doc_name}: {e}")
            return []
    
    def _extract_text_from_figure_region(self, fitz_doc, page, figure_bbox: list) -> str:
        """
        Extract OCR text from figure region (better for workflow diagrams)
        
        Args:
            fitz_doc: PyMuPDF document
            page: PyMuPDF page object
            figure_bbox: Figure bounding box [x0, y0, x1, y1]
            
        Returns:
            Extracted text from figure region using OCR
        """
        if not self.ocr_extractor or not self.ocr_extractor.is_available:
            return ""
        
        try:
            # Initialize OCR if needed
            self.ocr_extractor._init_ocr()
            if self.ocr_extractor.ocr_engine is None:
                return ""
            
            # Create clipped region from page
            # Add padding to ensure we capture text near figure boundaries
            padding = 10
            clip_rect = fitz.Rect(
                max(0, figure_bbox[0] - padding),
                max(0, figure_bbox[1] - padding), 
                min(page.rect.width, figure_bbox[2] + padding),
                min(page.rect.height, figure_bbox[3] + padding)
            )
            
            # Render region at 2x scale for better OCR
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat, clip=clip_rect)
            
            # Save temp region image for OCR
            cache_dir = Path("data/cache/ocr_temp")
            cache_dir.mkdir(parents=True, exist_ok=True)
            temp_path = cache_dir / f"figure_region_{hash(str(figure_bbox))}.png"
            pix.save(str(temp_path))
            
            try:
                # Run OCR on region
                result = self.ocr_extractor.ocr_engine.ocr(str(temp_path), cls=True)
                if result and result[0]:
                    # Convert OCR result to span format for bbox-based line detection
                    spans = []
                    for line in result[0]:
                        if line and len(line) >= 2:
                            bbox_points = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                            text_content = line[1][0]
                            confidence = line[1][1]
                            
                            # Skip low-confidence results
                            if confidence < 0.7:
                                continue
                            
                            # Convert to [x0, y0, x1, y1] format
                            x_coords = [p[0] for p in bbox_points]
                            y_coords = [p[1] for p in bbox_points]
                            bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                            
                            spans.append({
                                'bbox': bbox,
                                'content': text_content,
                                'type': 'text'
                            })
                    
                    if not spans:
                        return ""
                    
                    # Use PDF-Extract-Kit's merge logic for proper line detection
                    try:
                        from pdf_extract_kit.utils.merge_blocks_and_spans import (
                            merge_spans_to_line,
                            line_sort_spans_by_left_to_right
                        )
                        
                        # Merge spans into lines based on Y-axis overlap
                        lines = merge_spans_to_line(spans)
                        
                        # Sort spans within each line from left to right
                        line_objects = line_sort_spans_by_left_to_right(lines)
                        
                        # Build formatted text with proper line breaks
                        formatted_lines = []
                        for line_obj in line_objects:
                            line_text = ""
                            for span in line_obj['spans']:
                                line_text += span['content'].strip() + " "
                            formatted_lines.append(line_text.strip())
                        
                        text = "\n".join(formatted_lines)
                    except ImportError:
                        # Fallback: simple text extraction
                        text = " ".join([span['content'] for span in spans])
                    
                    if text.strip():
                        logger.debug(f"Region OCR extracted {len(text)} chars from figure region {figure_bbox}")
                    return text
                return ""
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
                    
        except Exception as e:
            logger.debug(f"Figure region OCR failed for bbox {figure_bbox}: {e}")
            return ""
    
    def _group_images(self, images: List[Dict], page) -> List[Dict[str, Any]]:
        """
        Group nearby images into single figures using spatial proximity
        
        Args:
            images: List of images with bbox
            page: PyMuPDF page for dimensions
            
        Returns:
            List of figure groups
        """
        if not images:
            return []
        
        # Region-based clustering: group images within same vertical region
        # This handles workflow diagrams and infographics better than pairwise proximity
        
        figures = []
        visited = set()
        
        # Sort by Y-coordinate (top to bottom)
        sorted_images = sorted(enumerate(images), key=lambda x: x[1]['bbox'][1])
        
        # Region parameters
        y_tolerance = 400  # Max vertical span for single figure (workflow diagrams ~355pt)
        x_overlap_ratio = 0.3  # Min horizontal overlap for grouping
        y_gap_threshold = 100  # Max gap between consecutive elements
        
        for idx, (i, img1) in enumerate(sorted_images):
            if i in visited:
                continue
            
            # Start new figure region
            figure_group = [img1]
            visited.add(i)
            
            bbox1 = img1['bbox']
            region_y_min = bbox1[1]
            region_y_max = bbox1[3]
            region_x_min = bbox1[0]
            region_x_max = bbox1[2]
            
            # Find all images in same vertical region
            for j, img2 in sorted_images[idx+1:]:
                if j in visited:
                    continue
                
                bbox2 = img2['bbox']
                y2_min, y2_max = bbox2[1], bbox2[3]
                x2_min, x2_max = bbox2[0], bbox2[2]
                
                # Check vertical span
                new_y_span = y2_max - region_y_min
                if new_y_span > y_tolerance:
                    continue
                
                # Check gap from region bottom to new image top
                y_gap = y2_min - region_y_max
                if y_gap > y_gap_threshold:
                    continue
                
                # Check horizontal overlap
                x_overlap = min(region_x_max, x2_max) - max(region_x_min, x2_min)
                region_width = region_x_max - region_x_min
                img2_width = x2_max - x2_min
                
                overlap_ratio1 = x_overlap / region_width if region_width > 0 else 0
                overlap_ratio2 = x_overlap / img2_width if img2_width > 0 else 0
                
                # Add if sufficient overlap or very close vertically
                if overlap_ratio1 >= x_overlap_ratio or overlap_ratio2 >= x_overlap_ratio or y_gap < 20:
                    figure_group.append(img2)
                    visited.add(j)
                    
                    # Expand region bounds
                    region_y_max = max(region_y_max, y2_max)
                    region_x_min = min(region_x_min, x2_min)
                    region_x_max = max(region_x_max, x2_max)
            
            # Create figure from group
            if len(figure_group) > 1:
                # Merge bboxes
                all_x0 = [img['bbox'][0] for img in figure_group]
                all_y0 = [img['bbox'][1] for img in figure_group]
                all_x1 = [img['bbox'][2] for img in figure_group]
                all_y1 = [img['bbox'][3] for img in figure_group]
                
                merged_bbox = [min(all_x0), min(all_y0), max(all_x1), max(all_y1)]
                
                figures.append({
                    'type': 'figure_group',
                    'bbox': merged_bbox,
                    'image_count': len(figure_group),
                    'images': figure_group
                })
            else:
                # Single image
                figures.append({
                    'type': 'single_image',
                    'bbox': img1['bbox'],
                    'image_count': 1,
                    'images': [img1]
                })
        
        return figures
