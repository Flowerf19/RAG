"""
Figure Extractor - Extract and group images with OCR
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import logging
import os
from PIL import Image
import io

if TYPE_CHECKING:
    from .ocr_extractor import OCRExtractor

logger = logging.getLogger(__name__)


class FigureExtractor:
    """Extract figures (images/diagrams) from PDF with OCR support"""
    
    def __init__(self, ocr_extractor: Optional['OCRExtractor'] = None):
        """
        Args:
            ocr_extractor: Optional OCR extractor for figure text extraction
        """
        self.ocr_extractor = ocr_extractor
    
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
            
            # Extract OCR text from each figure (if OCR available)
            if self.ocr_extractor:
                for figure in figures:
                    # Get unique xrefs from figure images
                    xrefs = list(set(img['xref'] for img in figure['images']))
                    
                    # Run OCR on each image and combine
                    ocr_texts = []
                    for xref in xrefs:
                        text = self._extract_text_from_image(fitz_doc, xref)
                        if text.strip():
                            ocr_texts.append(text)
                    
                    # Add combined text to figure
                    figure['text'] = " ".join(ocr_texts) if ocr_texts else ""
            else:
                # No OCR - set empty text
                for figure in figures:
                    figure['text'] = ""
            
            return figures
            
        except Exception as e:
            logger.warning(f"Figure extraction failed for page {page_num+1}: {e}")
            return []
    
    def _extract_text_from_image(self, fitz_doc, xref: int) -> str:
        """
        Extract OCR text FROM figure image
        
        Args:
            fitz_doc: PyMuPDF document
            xref: Image xref
            
        Returns:
            Extracted text from figure using OCR
        """
        if not self.ocr_extractor or not self.ocr_extractor.is_available:
            return ""
        
        try:
            # Initialize OCR if needed
            self.ocr_extractor._init_ocr()
            if self.ocr_extractor.ocr_engine is None:
                return ""
            
            # Extract image bytes
            base_image = fitz_doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Convert to PIL Image
            img = Image.open(io.BytesIO(image_bytes))
            
            # Save temp image for OCR
            cache_dir = Path("data/cache/ocr_temp")
            cache_dir.mkdir(parents=True, exist_ok=True)
            temp_path = cache_dir / f"figure_xref_{xref}.png"
            img.save(temp_path)
            
            try:
                # Run OCR
                result = self.ocr_extractor.ocr_engine.ocr(str(temp_path), cls=True)
                if result and result[0]:
                    # Convert OCR result to span format for bbox-based line detection
                    spans = []
                    for line in result[0]:
                        if line and len(line) >= 2:
                            bbox_points = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                            text_content = line[1][0]
                            
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
                    
                    if text.strip():
                        logger.debug(f"OCR extracted {len(text)} chars ({len(formatted_lines)} lines) from figure xref={xref}")
                    return text
                return ""
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
                    
        except Exception as e:
            logger.debug(f"Figure OCR failed for xref {xref}: {e}")
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
