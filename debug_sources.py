#!/usr/bin/env python3
"""
Debug script để kiểm tra dữ liệu nguồn trong session state
"""
import sys
import os
sys.path.insert(0, '.')

from pipeline.backend_connector import fetch_retrieval

def test_retrieval():
    """Test retrieval function"""
    query = "RAG là gì"
    print(f"Testing retrieval with query: '{query}'")

    try:
        result = fetch_retrieval(query, top_k=5, max_chars=2000)
        print("Retrieval successful!")

        context = result.get("context", "")
        sources = result.get("sources", [])

        print(f"Context length: {len(context)} characters")
        print(f"Number of sources: {len(sources)}")

        if sources:
            print("\nAll sources details:")
            for i, src in enumerate(sources):
                print(f"\nSource {i+1}:")
                print(f"  Keys: {list(src.keys())}")
                for key, value in src.items():
                    if key == 'text' and value:
                        print(f"  {key}: {len(value)} chars - {value[:50]}...")
                    else:
                        print(f"  {key}: {value}")
        else:
            print("No sources returned!")

    except Exception as e:
        print(f"Error during retrieval: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_retrieval()