#!/usr/bin/env python3
"""
Test script for pipeline cache logic
"""

print('=== Testing Pipeline Cache Logic ===')
try:
    from pipeline.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline()

    # Test cache detection
    is_processed, cached = pipeline._is_pdf_processed('data/pdf/Process_Risk Management.pdf')
    print(f'Cache check for Process_Risk Management.pdf: processed={is_processed}')

    if cached:
        print(f'Cached result keys: {list(cached.keys())}')

    # Test processing (should use cache)
    print('Testing process_pdf with cache...')
    result = pipeline.process_pdf('data/pdf/Process_Risk Management.pdf')
    print(f'Result: cached={result.get("cached", False)}, success={result.get("success", False)}')

    print('✅ Pipeline cache test completed successfully!')

except Exception as e:
    print(f'❌ Pipeline cache test failed: {e}')
    import traceback
    traceback.print_exc()