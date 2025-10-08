"""
Test phân tích chất lượng block chunking và đề xuất cải thiện.
"""

import pytest
import os
from typing import List, Dict, Any
from datetime import datetime

from loaders.pdf_loader import PDFLoader


class TestBlockChunkingQuality:
    """
    Test class để phân tích chất lượng việc chia blocks và đề xuất cải thiện.
    """
    
    TEST_PDF_PATH = r"C:\Users\ENGUYEHWC\Downloads\RAG\RAG\data\pdf\Process_Risk Management.pdf"
    OUTPUT_DIR = r"C:\Users\ENGUYEHWC\Downloads\RAG\RAG\test_outputs"
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup cho mỗi test method."""
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        if not os.path.exists(self.TEST_PDF_PATH):
            pytest.skip(f"Test PDF file not found: {self.TEST_PDF_PATH}")
    
    def _analyze_block_fragmentation(self, blocks: List[Any]) -> Dict[str, Any]:
        """Phân tích mức độ phân mảnh của blocks."""
        analysis = {
            "total_blocks": len(blocks),
            "fragmented_sequences": [],
            "short_blocks": [],
            "incomplete_sentences": [],
            "list_items": [],
            "potential_merges": []
        }
        
        for i, block in enumerate(blocks):
            if not isinstance(block, (tuple, list)) or len(block) < 5:
                continue
                
            text = str(block[4]).strip()
            if not text:
                continue
            
            # Detect short blocks (< 50 chars)
            if len(text) < 50:
                analysis["short_blocks"].append({
                    "index": i,
                    "text": text,
                    "length": len(text)
                })
            
            # Detect incomplete sentences (không kết thúc bằng dấu câu)
            if not text.endswith(('.', '!', '?', ':', ';')) and len(text) > 10:
                analysis["incomplete_sentences"].append({
                    "index": i,
                    "text": text,
                    "length": len(text)
                })
            
            # Detect list items
            if text.startswith('•') or text.startswith('-') or text.startswith('○'):
                analysis["list_items"].append({
                    "index": i,
                    "text": text,
                    "length": len(text)
                })
        
        # Analyze potential merges (consecutive short blocks)
        for i in range(len(blocks) - 1):
            if not isinstance(blocks[i], (tuple, list)) or len(blocks[i]) < 5:
                continue
            if not isinstance(blocks[i+1], (tuple, list)) or len(blocks[i+1]) < 5:
                continue
                
            text1 = str(blocks[i][4]).strip()
            text2 = str(blocks[i+1][4]).strip()
            
            if text1 and text2:
                # Check if blocks should be merged
                should_merge = False
                
                # Case 1: First block không kết thúc câu và block thứ 2 không bắt đầu uppercase
                if (not text1.endswith(('.', '!', '?', ':', ';')) and 
                    text2 and not text2[0].isupper() and 
                    len(text1) < 100 and len(text2) < 100):
                    should_merge = True
                
                # Case 2: List continuation
                if (text1.startswith('•') and not text1.endswith('.') and
                    not text2.startswith('•') and len(text2) < 80):
                    should_merge = True
                
                if should_merge:
                    analysis["potential_merges"].append({
                        "block1_index": i,
                        "block2_index": i + 1,
                        "text1": text1,
                        "text2": text2,
                        "merged_text": text1 + " " + text2,
                        "reason": "incomplete_sentence" if not text1.endswith(('.', '!', '?', ':', ';')) else "list_continuation"
                    })
        
        return analysis
    
    def _suggest_block_improvements(self, analysis: Dict[str, Any]) -> List[str]:
        """Đề xuất cải thiện dựa trên phân tích."""
        suggestions = []
        
        if len(analysis["short_blocks"]) > analysis["total_blocks"] * 0.3:
            suggestions.append(
                f"🔧 HIGH PRIORITY: {len(analysis['short_blocks'])} blocks are very short (<50 chars). "
                f"Consider merging adjacent short blocks."
            )
        
        if len(analysis["incomplete_sentences"]) > 10:
            suggestions.append(
                f"📝 MEDIUM PRIORITY: {len(analysis['incomplete_sentences'])} blocks have incomplete sentences. "
                f"These may need merging with following blocks."
            )
        
        if len(analysis["potential_merges"]) > 5:
            suggestions.append(
                f"🔀 HIGH PRIORITY: Found {len(analysis['potential_merges'])} potential block merges. "
                f"Implement post-processing to merge related blocks."
            )
        
        if len(analysis["list_items"]) > 0:
            fragmented_lists = 0
            for item in analysis["list_items"]:
                if not item["text"].endswith('.'):
                    fragmented_lists += 1
            
            if fragmented_lists > 0:
                suggestions.append(
                    f"📋 MEDIUM PRIORITY: {fragmented_lists} list items appear fragmented. "
                    f"Implement list-aware block merging."
                )
        
        return suggestions
    
    def test_analyze_block_chunking_quality(self):
        """Phân tích chất lượng block chunking."""
        print(f"\n🔍 ANALYZING BLOCK CHUNKING QUALITY")
        
        loader = PDFLoader.create_default()
        document = loader.load(self.TEST_PDF_PATH)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.OUTPUT_DIR, f"{timestamp}_chunking_analysis.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("BLOCK CHUNKING QUALITY ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            
            overall_stats = {
                "total_blocks": 0,
                "total_short_blocks": 0,
                "total_incomplete_sentences": 0,
                "total_potential_merges": 0,
                "pages_with_issues": 0
            }
            
            problematic_pages = []
            
            for page in document.pages:
                if not page.blocks:
                    continue
                
                f.write(f"\n=== PAGE {page.page_number} ===\n")
                
                analysis = self._analyze_block_fragmentation(page.blocks)
                suggestions = self._suggest_block_improvements(analysis)
                
                # Update overall stats
                overall_stats["total_blocks"] += analysis["total_blocks"]
                overall_stats["total_short_blocks"] += len(analysis["short_blocks"])
                overall_stats["total_incomplete_sentences"] += len(analysis["incomplete_sentences"])
                overall_stats["total_potential_merges"] += len(analysis["potential_merges"])
                
                if len(analysis["potential_merges"]) > 0 or len(analysis["short_blocks"]) > 5:
                    overall_stats["pages_with_issues"] += 1
                    problematic_pages.append(page.page_number)
                
                f.write(f"Total blocks: {analysis['total_blocks']}\n")
                f.write(f"Short blocks (<50 chars): {len(analysis['short_blocks'])}\n")
                f.write(f"Incomplete sentences: {len(analysis['incomplete_sentences'])}\n")
                f.write(f"Potential merges: {len(analysis['potential_merges'])}\n")
                
                # Show examples of problematic blocks
                if analysis["potential_merges"]:
                    f.write(f"\nPOTENTIAL MERGES (first 3):\n")
                    for merge in analysis["potential_merges"][:3]:
                        f.write(f"  Block {merge['block1_index']}: \"{merge['text1']}\"\n")
                        f.write(f"  Block {merge['block2_index']}: \"{merge['text2']}\"\n")
                        f.write(f"  → Merged: \"{merge['merged_text']}\"\n")
                        f.write(f"  Reason: {merge['reason']}\n\n")
                
                # Show short blocks examples
                if analysis["short_blocks"] and len(analysis["short_blocks"]) > 3:
                    f.write(f"SHORT BLOCKS (first 3):\n")
                    for block in analysis["short_blocks"][:3]:
                        f.write(f"  Block {block['index']} ({block['length']} chars): \"{block['text']}\"\n")
                    f.write("\n")
                
                # Show suggestions
                if suggestions:
                    f.write("SUGGESTIONS:\n")
                    for suggestion in suggestions:
                        f.write(f"  {suggestion}\n")
                    f.write("\n")
            
            # Overall summary
            f.write("\n" + "=" * 50 + "\n")
            f.write("OVERALL ANALYSIS\n")
            f.write("=" * 50 + "\n")
            
            f.write(f"Total blocks: {overall_stats['total_blocks']}\n")
            f.write(f"Short blocks: {overall_stats['total_short_blocks']} ({overall_stats['total_short_blocks']/overall_stats['total_blocks']*100:.1f}%)\n")
            f.write(f"Incomplete sentences: {overall_stats['total_incomplete_sentences']} ({overall_stats['total_incomplete_sentences']/overall_stats['total_blocks']*100:.1f}%)\n")
            f.write(f"Potential merges: {overall_stats['total_potential_merges']}\n")
            f.write(f"Pages with issues: {overall_stats['pages_with_issues']}/16\n")
            
            # Calculate quality score
            short_block_penalty = min(50, overall_stats['total_short_blocks'] / overall_stats['total_blocks'] * 100)
            incomplete_penalty = min(30, overall_stats['total_incomplete_sentences'] / overall_stats['total_blocks'] * 100)
            merge_penalty = min(20, overall_stats['total_potential_merges'] / overall_stats['total_blocks'] * 100)
            
            quality_score = 100 - short_block_penalty - incomplete_penalty - merge_penalty
            
            f.write(f"\nCHUNKING QUALITY SCORE: {quality_score:.1f}/100\n")
            
            if quality_score >= 80:
                f.write("GRADE: A (Good chunking)\n")
            elif quality_score >= 70:
                f.write("GRADE: B (Acceptable with minor issues)\n") 
            elif quality_score >= 60:
                f.write("GRADE: C (Needs improvement)\n")
            else:
                f.write("GRADE: D (Poor chunking - major fixes needed)\n")
            
            f.write(f"\nProblematic pages: {problematic_pages}\n")
            
            # Top recommendations
            f.write("\nTOP RECOMMENDATIONS:\n")
            if overall_stats['total_potential_merges'] > 20:
                f.write("1. 🔥 CRITICAL: Implement automatic block merging for fragmented sentences\n")
            if overall_stats['total_short_blocks'] > overall_stats['total_blocks'] * 0.4:
                f.write("2. 🔧 HIGH: Too many short blocks - improve block boundary detection\n")
            if overall_stats['pages_with_issues'] > 8:
                f.write("3. 📋 MEDIUM: More than half pages have chunking issues - review PDF parsing logic\n")
            
            f.write("\nPOSSIBLE SOLUTIONS:\n")
            f.write("- Add post-processing step to merge incomplete sentences\n")
            f.write("- Implement paragraph-aware block detection\n")
            f.write("- Use text flow analysis to identify logical boundaries\n")
            f.write("- Add list item consolidation logic\n")
        
        print(f"📄 Chunking analysis saved to: {report_file}")
        
        # Print summary to console
        print(f"📊 CHUNKING QUALITY SUMMARY:")
        print(f"   Total blocks: {overall_stats['total_blocks']}")
        print(f"   Short blocks: {overall_stats['total_short_blocks']} ({overall_stats['total_short_blocks']/overall_stats['total_blocks']*100:.1f}%)")
        print(f"   Potential merges: {overall_stats['total_potential_merges']}")
        print(f"   Quality score: {quality_score:.1f}/100")
        print(f"   Pages with issues: {overall_stats['pages_with_issues']}/16")
        
        # Assert for test validation
        assert overall_stats['total_blocks'] > 0, "Should have found blocks"
        
        return report_file, quality_score
    
    def test_demonstrate_improved_chunking(self):
        """Demo cách cải thiện chunking bằng cách merge blocks."""
        print(f"\n🔧 DEMONSTRATING IMPROVED CHUNKING")
        
        loader = PDFLoader.create_default()
        document = loader.load(self.TEST_PDF_PATH)
        
        # Find the problematic page (around page 15-16 where the risk approaches are)
        target_page = None
        for page in document.pages:
            if page.blocks:
                for block in page.blocks:
                    if isinstance(block, (tuple, list)) and len(block) >= 5:
                        text = str(block[4])
                        if "different approaches to risk management" in text:
                            target_page = page
                            break
            if target_page:
                break
        
        if not target_page:
            print("Could not find the problematic text section")
            return
        
        print(f"Found problematic section on page {target_page.page_number}")
        
        # Extract the blocks containing the risk approaches section
        risk_blocks = []
        capturing = False
        
        for i, block in enumerate(target_page.blocks):
            if isinstance(block, (tuple, list)) and len(block) >= 5:
                text = str(block[4]).strip()
                
                if "different approaches to risk management" in text:
                    capturing = True
                
                if capturing:
                    risk_blocks.append((i, text))
                    
                if capturing and ("operations." in text or len(risk_blocks) > 10):
                    break
        
        # Show original vs improved
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        demo_file = os.path.join(self.OUTPUT_DIR, f"{timestamp}_chunking_improvement_demo.txt")
        
        with open(demo_file, 'w', encoding='utf-8') as f:
            f.write("CHUNKING IMPROVEMENT DEMONSTRATION\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("ORIGINAL (FRAGMENTED) BLOCKS:\n")
            f.write("-" * 30 + "\n")
            for i, text in risk_blocks:
                f.write(f"Block {i}: {text}\n")
            
            f.write(f"\nIMPROVED (MERGED) VERSION:\n")
            f.write("-" * 30 + "\n")
            
            # Simple merging logic demonstration
            merged_blocks = []
            current_block = ""
            
            for i, text in risk_blocks:
                if not current_block:
                    current_block = text
                elif (not current_block.endswith(('.', '!', '?', ':')) and 
                      text and not text[0].isupper() and not text.startswith('•')):
                    # Merge with previous
                    current_block += " " + text
                elif text.startswith('•') and not current_block.endswith('.'):
                    # Merge list item with previous incomplete sentence
                    current_block += "\n" + text
                else:
                    # Start new block
                    merged_blocks.append(current_block)
                    current_block = text
            
            if current_block:
                merged_blocks.append(current_block)
            
            for i, merged_text in enumerate(merged_blocks):
                f.write(f"Merged Block {i+1}:\n{merged_text}\n\n")
            
            f.write("IMPROVEMENT METRICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Original blocks: {len(risk_blocks)}\n")
            f.write(f"Merged blocks: {len(merged_blocks)}\n")
            f.write(f"Reduction: {len(risk_blocks) - len(merged_blocks)} blocks ({(len(risk_blocks) - len(merged_blocks))/len(risk_blocks)*100:.1f}%)\n")
            
        print(f"📄 Improvement demo saved to: {demo_file}")
        print(f"   Original blocks: {len(risk_blocks)}")
        print(f"   Merged blocks: {len(merged_blocks)}")
        print(f"   Improvement: {(len(risk_blocks) - len(merged_blocks))/len(risk_blocks)*100:.1f}% reduction")
        
        return demo_file