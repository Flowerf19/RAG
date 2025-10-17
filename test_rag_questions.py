#!/usr/bin/env python3
"""
Test hệ thống RAG với 15 câu hỏi về RAG và Reranker
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

# 15 câu hỏi test
TEST_QUESTIONS = [
    {
        "id": 1,
        "question": "Giải thích vai trò của Reranker trong quy trình truy xuất tăng cường thế hệ (RAG) hai giai đoạn. Tại sao giai đoạn reranking lại đặc biệt quan trọng đối với sự hài lòng của người dùng?",
        "topic": "Vai trò của Reranker trong Kiến trúc RAG"
    },
    {
        "id": 2,
        "question": "Mô tả sự khác biệt cốt lõi trong cách cross-encoder (như BGE Reranker hoặc ViRanker) và bi-encoder xử lý cặp truy vấn–tài liệu (query–document) để đánh giá mức độ liên quan.",
        "topic": "Sự khác biệt cốt lõi giữa Cross-Encoder và Bi-Encoder"
    },
    {
        "id": 3,
        "question": "Hai sửa đổi kiến trúc chính được áp dụng cho encoder nền tảng BGE-M3 để tạo ra ViRanker là gì, và sửa đổi nào được thêm vào để cải thiện hiệu suất tính toán và xử lý tài liệu dài?",
        "topic": "Kiến trúc cốt lõi của ViRanker"
    },
    {
        "id": 4,
        "question": "Kích thước của kho ngữ liệu tiếng Việt được tuyển chọn (curated corpus) được sử dụng để đào tạo ViRanker là bao nhiêu, và mô hình này đã sử dụng chiến lược Hybrid Hard-Negative Mining bao gồm những phương pháp truy xuất nào?",
        "topic": "Dữ liệu và Chiến lược Lấy mẫu Phủ định Cứng của ViRanker"
    },
    {
        "id": 5,
        "question": "Kỹ thuật Hybrid Retrieval (Truy xuất Lai) kết hợp những phương pháp tìm kiếm nào, và hai lợi ích chính mà nó mang lại cho hệ thống RAG là gì?",
        "topic": "Kỹ thuật Hybrid Retrieval trong RAG"
    },
    {
        "id": 6,
        "question": "ViRanker đạt được những điểm số NDCG@3 và MRR@3 nào trên bộ benchmark MMARCO-VI? Theo tài liệu, so với PhoRanker, ViRanker thể hiện ưu thế ở loại truy vấn nào, và PhoRanker giữ lợi thế ở loại truy vấn nào?",
        "topic": "Hiệu suất và So sánh giữa ViRanker và PhoRanker"
    },
    {
        "id": 7,
        "question": "Theo tài liệu, hai loại vấn đề/truy vấn chính thường dẫn đến lỗi (failures) cho ViRanker trên tập dữ liệu MMARCO-VI là gì?",
        "topic": "Phân tích lỗi của ViRanker"
    },
    {
        "id": 8,
        "question": "Jina Reranker v2 nổi bật so với các reranker khác (như BGE-reranker-v2-m3) ở khía cạnh nào liên quan đến tốc độ và khả năng xử lý ngữ cảnh dài (long context)?",
        "topic": "Ưu điểm nổi bật về Hiệu suất của Jina Reranker v2"
    },
    {
        "id": 9,
        "question": "Liệt kê ba lý do cốt lõi khiến kiến trúc RAG cơ bản ('chunk documents → embed them → store in a vector database → retrieve top-k similar chunks') thường thất bại trong các ứng dụng thực tế.",
        "topic": "Tại sao RAG Cơ bản thất bại"
    },
    {
        "id": 10,
        "question": "Kỹ thuật PageIndex giải quyết vấn đề gì của RAG cơ bản, và cách hoạt động của nó mô phỏng cách con người duyệt tài liệu như thế nào?",
        "topic": "Kỹ thuật PageIndex"
    },
    {
        "id": 11,
        "question": "Cache-Augmented Generation (CAG) hoạt động như thế nào để tối ưu hóa chi phí và độ trễ cho hệ thống RAG? Kỹ thuật này thường được dùng cho loại dữ liệu nào?",
        "topic": "Kỹ thuật Cache-Augmented Generation (CAG)"
    },
    {
        "id": 12,
        "question": "Kỹ thuật Self-Reasoning (Tự Lý luận) chuyển đổi hệ thống RAG từ một công cụ thụ động thành một tác nhân như thế nào, và lợi ích chính của nó đối với đầu ra của LLM là gì?",
        "topic": "Kỹ thuật Self-Reasoning"
    },
    {
        "id": 13,
        "question": "Mục đích của kỹ thuật Multivector Retrieval là gì, và nó giải quyết hạn chế nào của tìm kiếm vector truyền thống?",
        "topic": "Kỹ thuật Multivector Retrieval"
    },
    {
        "id": 14,
        "question": "Adaptive RAG (RAG Thích ứng) xử lý các truy vấn đơn giản và phức tạp khác nhau như thế nào, và lợi ích mà kỹ thuật này mang lại?",
        "topic": "Kỹ thuật Adaptive RAG (RAG Thích ứng)"
    },
    {
        "id": 15,
        "question": "Ngoài các số liệu xếp hạng truyền thống (như NDCG), hãy định nghĩa và giải thích ý nghĩa của hai số liệu quan trọng sau để đo lường chất lượng đầu ra của LLM trong RAG: Faithfulness (Tính Trung thực) và Context Precision (Độ chính xác Ngữ cảnh).",
        "topic": "Các Số liệu Đánh giá Chính trong RAG"
    }
]

def test_rag_system():
    """Test hệ thống RAG với 15 câu hỏi"""
    results = []

    print("🚀 Testing RAG System with 15 Questions")
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

def save_results(results, filename="rag_test_results.json"):
    """Save test results to JSON file"""
    output_path = Path(filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Results saved to {output_path}")

def print_summary(results):
    """Print test summary"""
    successful = len([r for r in results if "error" not in r])
    failed = len([r for r in results if "error" in r])

    print(f"\n📊 Test Summary:")
    print(f"   ✅ Successful: {successful}/15")
    print(f"   ❌ Failed: {failed}/15")

    if successful > 0:
        avg_context = sum(r.get("context_length", 0) for r in results if "error" not in r) / successful
        avg_response = sum(r.get("response_length", 0) for r in results if "error" not in r) / successful
        avg_sources = sum(r.get("sources_count", 0) for r in results if "error" not in r) / successful

        print(f"   📄 Average context length: {avg_context:.0f} chars")
        print(f"   💬 Average response length: {avg_response:.0f} chars")
        print(f"   📚 Average sources count: {avg_sources:.1f}")

if __name__ == "__main__":
    results = test_rag_system()
    save_results(results)
    print_summary(results)
    print("\n🎉 RAG System Testing Complete!")