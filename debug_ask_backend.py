#!/usr/bin/env python3
"""
Test ask_backend function như trong LLM_FE.py
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

def ask_backend_test(prompt_text: str) -> str:
    """
    Test version của ask_backend từ LLM_FE.py
    """
    try:
        # TODO: Khi có retrieval system, lấy context ở đây
        context = ""  # Tạm thời để trống

        # Build messages bằng chat_handler
        # Lấy context từ Retrieval (nếu có) và lưu nguồn để hiển thị.
        try:
            print(f"DEBUG: Calling fetch_retrieval for prompt: {prompt_text[:50]}...")
            ret = fetch_retrieval(prompt_text, top_k=10, max_chars=8000)  # Tăng lên 8000
            context = ret.get("context", "") or ""
            last_sources = ret.get("sources", [])
            print(f"DEBUG: Retrieval successful - context: {len(context)} chars, sources: {len(last_sources)}")
        except Exception as e:
            print(f"DEBUG: Retrieval failed: {e}")
            import traceback
            traceback.print_exc()
            context = ""
            last_sources = []

        messages = build_messages(
            query=prompt_text,
            context=context,
            history=[]
        )

        # Gọi LLM
        print("DEBUG: Calling LLM...")
        reply = call_gemini(messages)
        print(f"DEBUG: LLM response: {len(reply)} chars")

        return reply

    except Exception as e:
        return f"[Error] {e}"

if __name__ == "__main__":
    test_query = "What are RAG techniques?"
    print(f"Testing ask_backend with: '{test_query}'")

    response = ask_backend_test(test_query)
    print(f"\nResponse: {response}")