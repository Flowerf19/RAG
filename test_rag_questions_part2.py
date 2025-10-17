#!/usr/bin/env python3
"""
Test hệ thống RAG với câu hỏi 11-15 (phần còn lại)
"""

import sys
import os
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from pipeline.backend_connector import fetch_retrieval
    from llm.chat_handler import build_messages
    from llm.LLM_API import call_gemini
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Câu hỏi 11-15 (phần còn lại)
TEST_QUESTIONS = [
    {
        "id": 11,
        "question": "Cache-Augmented Generation (CAG) hoạt động như thế nào để tối ưu hóa chi phí và độ trễ cho hệ thống RAG, và kỹ thuật này sử dụng những phương pháp nào để xác định và lưu trữ các câu trả lời có thể tái sử dụng?",
        "topic": "Kỹ thuật Cache-Augmented Generation (CAG)"
    },
    {
        "id": 12,
        "question": "Kỹ thuật Speculative Retrieval (Truy xuất Dự đoán) hoạt động như thế nào để cải thiện hiệu suất của hệ thống RAG, và nó sử dụng những phương pháp nào để dự đoán và mở rộng tập tài liệu liên quan?",
        "topic": "Kỹ thuật Speculative Retrieval"
    },
    {
        "id": 13,
        "question": "Kỹ thuật Step-Back Prompting giải quyết vấn đề gì của RAG cơ bản, và cách hoạt động của nó khác với các kỹ thuật prompting khác như Chain-of-Thought như thế nào?",
        "topic": "Kỹ thuật Step-Back Prompting"
    },
    {
        "id": 14,
        "question": "Kỹ thuật Sub-Question Query Engine (SQE) hoạt động như thế nào để cải thiện khả năng trả lời câu hỏi phức tạp trong hệ thống RAG, và nó sử dụng những chiến lược nào để phân tích và tổng hợp thông tin?",
        "topic": "Kỹ thuật Sub-Question Query Engine (SQE)"
    },
    {
        "id": 15,
        "question": "Kỹ thuật Iterative Retrieval-Generation (IRG) khác với kiến trúc RAG cơ bản như thế nào, và nó sử dụng những phương pháp nào để tinh chỉnh dần dần quá trình truy xuất và tạo ra câu trả lời?",
        "topic": "Kỹ thuật Iterative Retrieval-Generation (IRG)"
    }
]

def test_rag_system_part2():
    """Test hệ thống RAG với câu hỏi 11-15"""
    results = []

    print("🚀 Testing RAG System with Questions 11-15")
    print("=" * 60)

    for i, test_case in enumerate(TEST_QUESTIONS, 1):
        question_id = test_case["id"]
        question = test_case["question"]
        topic = test_case["topic"]

        print(f"\n📋 Question {question_id}: {topic}")
        print(f"❓ {question[:100]}{'...' if len(question) > 100 else ''}")

        try:
            # Step 1: Get retrieval data
            print("🔍 Retrieving context...")
            ret = fetch_retrieval(question, top_k=10, max_chars=8000)
            context = ret.get("context", "") or ""
            sources = ret.get("sources", [])

            print(f"   📄 Context: {len(context)} chars")
            print(f"   📚 Sources: {len(sources)} items")

            # Step 2: Build messages and call LLM
            print("🤖 Generating answer...")
            messages = build_messages(
                query=question,
                context=context,
                history=[]
            )

            response = call_gemini(messages)

            print(f"   💬 Response: {len(response)} chars")

            # Store result
            result = {
                "id": question_id,
                "topic": topic,
                "question": question,
                "context_length": len(context),
                "sources_count": len(sources),
                "response_length": len(response),
                "response_preview": response[:200] + "..." if len(response) > 200 else response,
                "top_sources": [
                    {
                        "file": src.get("file_name", ""),
                        "page": src.get("page_number", ""),
                        "score": src.get("similarity_score", 0)
                    } for src in sources[:3]  # Top 3 sources
                ]
            }

            results.append(result)

            print("   ✅ Success")

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                "id": question_id,
                "topic": topic,
                "question": question,
                "error": str(e)
            })

    return results

def save_results(results, filename="rag_test_results_part2.json"):
    """Save test results to JSON file"""
    output_path = Path(filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Results saved to {output_path}")

def print_summary(results):
    """Print test summary"""
    successful = len([r for r in results if "error" not in r])
    failed = len([r for r in results if "error" in r])

    print(f"\n📊 Test Summary (Part 2):")
    print(f"   ✅ Successful: {successful}/5")
    print(f"   ❌ Failed: {failed}/5")

    if successful > 0:
        avg_context = sum(r.get("context_length", 0) for r in results if "error" not in r) / successful
        avg_response = sum(r.get("response_length", 0) for r in results if "error" not in r) / successful
        avg_sources = sum(r.get("sources_count", 0) for r in results if "error" not in r) / successful

        print(f"   📄 Average context length: {avg_context:.0f} chars")
        print(f"   💬 Average response length: {avg_response:.0f} chars")
        print(f"   📚 Average sources count: {avg_sources:.1f}")

if __name__ == "__main__":
    results = test_rag_system_part2()
    save_results(results)
    print_summary(results)
    print("\n🎉 RAG System Testing Part 2 Complete!")