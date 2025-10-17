#!/usr/bin/env python3
"""
Test toàn bộ flow như trong LLM_FE.py
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from pipeline.backend_connector import fetch_retrieval
    from llm.chat_handler import build_messages
    from llm.LLM_API import call_gemini
    print("All imports successful")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_full_flow():
    """Test toàn bộ flow như trong LLM_FE.py"""
    prompt_text = "What are RAG techniques?"

    print(f"Testing full flow for: '{prompt_text}'")

    try:
        # Step 1: Get retrieval data (như trong LLM_FE.py)
        print("Step 1: Fetching retrieval data...")
        ret = fetch_retrieval(prompt_text, top_k=10, max_chars=8000)
        context = ret.get("context", "") or ""
        sources = ret.get("sources", [])

        print(f"  Context length: {len(context)}")
        print(f"  Sources count: {len(sources)}")

        # Step 2: Build messages (như trong LLM_FE.py)
        print("Step 2: Building messages...")
        messages = build_messages(
            query=prompt_text,
            context=context,
            history=[]  # Empty history for test
        )

        print(f"  Messages count: {len(messages)}")
        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content_preview = msg.get("content", "")[:100]
            print(f"    {i+1}. {role}: {content_preview}...")

        # Step 3: Test LLM call (như trong LLM_FE.py)
        print("Step 3: Testing LLM call...")
        try:
            reply = call_gemini(messages)
            print(f"  LLM response length: {len(reply)}")
            print(f"  Response preview: {reply[:200]}...")
            return True
        except Exception as e:
            print(f"  LLM call failed: {e}")
            return False

    except Exception as e:
        print(f"Flow failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_flow()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")