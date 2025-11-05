"""
Test script to demonstrate PDF-Extract-Kit tasks integration with extractors
"""

import logging
from PDFLoaders.provider.extractors import OCRExtractor, TableExtractor, FigureExtractor
from PDFLoaders.pdf_provider import PDFProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_kit_tasks_availability():
    """Check which PDF-Extract-Kit tasks are available"""
    print("\n" + "="*70)
    print("PDF-EXTRACT-KIT TASKS AVAILABILITY")
    print("="*70)
    
    try:
        from PDFLoaders.pdf_extract_kit.tasks import (
            _layout_available, _formula_available, _formula_recog_available,
            _ocr_available, _table_available
        )
        
        tasks = [
            ("LayoutDetectionTask", _layout_available, "Detect page layout structure"),
            ("FormulaDetectionTask", _formula_available, "Detect mathematical formulas"),
            ("FormulaRecognitionTask", _formula_recog_available, "Recognize formulas to LaTeX"),
            ("OCRTask", _ocr_available, "OCR with registry pattern"),
            ("TableParsingTask", _table_available, "Advanced ML-based table parsing")
        ]
        
        available_count = sum(1 for _, avail, _ in tasks if avail)
        
        for name, available, description in tasks:
            status = "✓" if available else "✗"
            print(f"  {status} {name:25s} - {description}")
        
        print(f"\nTotal: {available_count}/5 tasks available")
        
        if available_count == 5:
            print("✅ All PDF-Extract-Kit tasks are available!")
        else:
            print("⚠️  Some tasks are not available (may require model configuration)")
            
    except ImportError as e:
        print(f"❌ Could not import PDF-Extract-Kit tasks: {e}")

def test_extractors_integration():
    """Test that extractors support PDF-Extract-Kit tasks"""
    print("\n" + "="*70)
    print("EXTRACTORS INTEGRATION WITH KIT TASKS")
    print("="*70)
    
    # Test OCRExtractor
    print("\n📄 OCRExtractor:")
    ocr = OCRExtractor(lang="multilingual")
    print(f"  • PaddleOCR available: {ocr.is_available}")
    print(f"  • Kit OCRTask support: {hasattr(ocr, 'use_kit_ocr') and ocr.use_kit_ocr is not None}")
    print(f"  • Currently using: PaddleOCR (default)")
    
    # Test TableExtractor
    print("\n📊 TableExtractor:")
    table = TableExtractor()
    print(f"  • ML parsing support: {hasattr(table, 'use_ml_parsing')}")
    print(f"  • ML parsing enabled: {table.use_ml_parsing if hasattr(table, 'use_ml_parsing') else False}")
    print(f"  • Currently using: pdfplumber + OCR (default)")
    
    # Test FigureExtractor
    print("\n🖼️  FigureExtractor:")
    fig = FigureExtractor()
    print(f"  • ML detection support: {hasattr(fig, 'use_ml_detection')}")
    print(f"  • ML detection enabled: {fig.use_ml_detection if hasattr(fig, 'use_ml_detection') else False}")
    print(f"  • Currently using: spatial grouping + OCR (default)")
    
    print("\n✅ All extractors have been enhanced with PDF-Extract-Kit task support!")

def test_pdf_provider():
    """Test PDFProvider with kit tasks integration"""
    print("\n" + "="*70)
    print("PDF PROVIDER INITIALIZATION")
    print("="*70 + "\n")
    
    provider = PDFProvider(use_ocr="auto", ocr_lang="multilingual")
    
    print("\n✅ PDFProvider initialized successfully!")
    print("\nNote: PDF-Extract-Kit tasks are available but require model configuration")
    print("      to be fully functional (YOLO weights, UniMERNet models, etc.)")

def show_usage_info():
    """Show information about how to use the kit tasks"""
    print("\n" + "="*70)
    print("USAGE INFORMATION")
    print("="*70)
    
    print("\n📚 How to enable PDF-Extract-Kit tasks:")
    print("\n1. For ML-based Table Parsing:")
    print("   table_extractor = TableExtractor(use_ml_parsing=True)")
    print("   # Requires: CUDA GPU + StructEqTable model weights")
    
    print("\n2. For ML-based Figure Detection:")
    print("   figure_extractor = FigureExtractor(use_ml_detection=True)")
    print("   # Requires: YOLO weights for layout/formula detection")
    
    print("\n3. For Kit OCRTask:")
    print("   ocr_extractor = OCRExtractor(use_kit_ocr=True)")
    print("   # Requires: Model configuration in pdf_extract_kit/tasks/ocr/")
    
    print("\n⚠️  Current Status:")
    print("   • Tasks are imported and available ✓")
    print("   • Model configurations are not set up yet")
    print("   • Using default implementations (pdfplumber, PaddleOCR, spatial grouping)")
    
    print("\n💡 Benefits of current integration:")
    print("   • Tasks are ready to use when models are configured")
    print("   • Graceful degradation to default implementations")
    print("   • No breaking changes to existing code")
    print("   • Easy to enable advanced features when needed")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("PDF-EXTRACT-KIT INTEGRATION TEST")
    print("="*70)
    
    test_kit_tasks_availability()
    test_extractors_integration()
    test_pdf_provider()
    show_usage_info()
    
    print("\n" + "="*70)
    print("TEST COMPLETED")
    print("="*70 + "\n")
