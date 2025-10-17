#!/usr/bin/env python3
"""
Debug script để test fetch_retrieval
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from pipeline.backend_connector import fetch_retrieval
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_retrieval():
    """Test hàm fetch_retrieval"""
    prompt_text = "What are RAG techniques?"

    print(f"Testing retrieval for: '{prompt_text}'")

    try:
        ret = fetch_retrieval(prompt_text, top_k=3, max_chars=1000)
        context = ret.get("context", "") or ""
        sources = ret.get("sources", [])

        print("Retrieval successful!")
        print(f"Context length: {len(context)} characters")
        print(f"Sources count: {len(sources)}")

        if context:
            print("Context preview:")
            print(context[:200] + "..." if len(context) > 200 else context)
        else:
            print("No context returned!")

        return True

    except Exception as e:
        print(f"Retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_retrieval()
    print(f"Test {'PASSED' if success else 'FAILED'}")