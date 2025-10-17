#!/usr/bin/env python3
"""
Test việc lưu sources vào session state
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

def test_session_state_simulation():
    """Simulate session state như trong Streamlit"""

    # Simulate session state
    session_state = {
        "messages": [],
        "is_generating": False,
        "pending_prompt": None,
        "last_sources": []
    }

    prompt_text = "What are RAG techniques?"

    print(f"Testing with prompt: '{prompt_text}'")

    try:
        print("Calling fetch_retrieval...")
        ret = fetch_retrieval(prompt_text, top_k=10, max_chars=8000)
        context = ret.get("context", "") or ""
        session_state["last_sources"] = ret.get("sources", [])

        print("✅ Retrieval successful!")
        print(f"📝 Context length: {len(context)}")
        print(f"📚 Sources count: {len(session_state['last_sources'])}")

        # Check if sources are properly stored
        if session_state["last_sources"]:
            print("✅ Sources stored in session state")
            for i, src in enumerate(session_state["last_sources"][:3], 1):
                title = src.get("title", "No title")
                score = src.get("similarity_score", 0)
                print(f"  {i}. {title} (score: {score})")
        else:
            print("❌ No sources stored in session state")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_session_state_simulation()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")