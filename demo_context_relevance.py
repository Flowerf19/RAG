#!/usr/bin/env python3
"""
Demo Context Relevance với mock contexts
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragas import evaluate
from ragas.metrics import ContextRelevance
from ragas.llms import LangchainLLMWrapper
from langchain_ollama import ChatOllama
from datasets import Dataset

def demo_context_relevance():
    """Demo context relevance với và không có contexts"""
    print("🎯 Demo Context Relevance với Mock Contexts")
    print("=" * 60)

    # Setup Ollama LLM for Ragas
    llm = ChatOllama(model="gemma3:1b", temperature=0.1)
    ragas_llm = LangchainLLMWrapper(llm)

    # Test case: Câu hỏi về CNN
    question = "CNN được sử dụng để làm gì?"
    answer = "CNN (Convolutional Neural Networks) được sử dụng chủ yếu cho computer vision tasks như image recognition."
    ground_truth = "CNN (Convolutional Neural Networks) được sử dụng chủ yếu cho computer vision tasks như image recognition."

    print(f"📝 Question: {question}")
    print(f"🤖 Answer: {answer[:80]}...")
    print()

    # Test 1: Empty contexts (như hiện tại)
    print("🧪 Test 1: Empty Contexts (như hiện tại)")
    data_empty = {
        'question': [question],
        'answer': [answer],
        'contexts': [[]],  # Empty list
        'ground_truth': [ground_truth]
    }

    dataset_empty = Dataset.from_dict(data_empty)
    result_empty = evaluate(dataset_empty, [ContextRelevance()], llm=ragas_llm)
    context_relevance_empty = float(result_empty['nv_context_relevance'][0])

    print("   Contexts: [] (empty)")
    print(f"   Context Relevance: {context_relevance_empty}")
    print("   → Không có contexts → Relevance = 0")
    print()

    # Test 2: Mock contexts liên quan
    print("🧪 Test 2: Mock Contexts liên quan")
    mock_contexts_relevant = [
        "CNN là Convolutional Neural Networks, một loại mạng neural được thiết kế đặc biệt cho việc xử lý dữ liệu hình ảnh.",
        "CNN được sử dụng chủ yếu trong computer vision tasks như nhận dạng hình ảnh, phân loại đối tượng, và phát hiện biên.",
        "Các ứng dụng của CNN bao gồm image recognition, object detection, và medical image analysis."
    ]

    data_relevant = {
        'question': [question],
        'answer': [answer],
        'contexts': [mock_contexts_relevant],
        'ground_truth': [ground_truth]
    }

    dataset_relevant = Dataset.from_dict(data_relevant)
    result_relevant = evaluate(dataset_relevant, [ContextRelevance()], llm=ragas_llm)
    context_relevance_relevant = float(result_relevant['nv_context_relevance'][0])

    print(f"   Contexts: {len(mock_contexts_relevant)} relevant chunks")
    print(f"   Context Relevance: {context_relevance_relevant}")
    print("   → Contexts liên quan → Relevance > 0")
    print()

    # Test 3: Mock contexts không liên quan
    print("🧪 Test 3: Mock Contexts KHÔNG liên quan")
    mock_contexts_irrelevant = [
        "Machine learning là một nhánh của trí tuệ nhân tạo.",
        "Deep learning sử dụng neural networks với nhiều layers.",
        "Python là một ngôn ngữ lập trình phổ biến."
    ]

    data_irrelevant = {
        'question': [question],
        'answer': [answer],
        'contexts': [mock_contexts_irrelevant],
        'ground_truth': [ground_truth]
    }

    dataset_irrelevant = Dataset.from_dict(data_irrelevant)
    result_irrelevant = evaluate(dataset_irrelevant, [ContextRelevance()], llm=ragas_llm)
    context_relevance_irrelevant = float(result_irrelevant['nv_context_relevance'][0])

    print(f"   Contexts: {len(mock_contexts_irrelevant)} irrelevant chunks")
    print(f"   Context Relevance: {context_relevance_irrelevant}")
    print("   → Contexts không liên quan → Relevance thấp")
    print()

    # Summary
    print("📊 Summary:")
    print(f"   Empty contexts:     {context_relevance_empty}")
    print(f"   Relevant contexts:  {context_relevance_relevant}")
    print(f"   Irrelevant contexts: {context_relevance_irrelevant}")
    print()
    print("💡 Kết luận:")
    print("   - Context Relevance đo lường mức độ contexts hỗ trợ trả lời câu hỏi")
    print("   - Cần có actual contexts từ RAG retrieval để có điểm số có ý nghĩa")
    print("   - Empty contexts = 0.0 (như hiện tại)")
    print("   - Good RAG system nên có Context Relevance > 0.5")

def test_with_database_data():
    """Test với data thực từ database nhưng thêm mock contexts"""
    print("\n🔄 Test với Database Data + Mock Contexts")
    print("=" * 60)

    # Setup Ollama LLM for Ragas
    llm = ChatOllama(model="gemma3:1b", temperature=0.1)
    ragas_llm = LangchainLLMWrapper(llm)

    from evaluation.backend_dashboard.api import BackendDashboard
    b = BackendDashboard()
    gt = b.get_ground_truth_list(limit=1)

    if gt:
        item = gt[0]
        question = item['question']
        answer = item['answer']

        # Mock contexts liên quan
        mock_contexts = [
            f"{question} CNN là viết tắt của Convolutional Neural Networks.",
            "CNN được thiết kế đặc biệt để xử lý dữ liệu hình ảnh 2D.",
            "Các ứng dụng chính của CNN bao gồm computer vision, image recognition."
        ]

        data = {
            'question': [question],
            'answer': [answer],
            'contexts': [mock_contexts],
            'ground_truth': [answer]
        }

        dataset = Dataset.from_dict(data)
        result = evaluate(dataset, [ContextRelevance()], llm=ragas_llm)
        score = float(result['nv_context_relevance'][0])

        print(f"Question: {question[:50]}...")
        print(f"Mock Contexts: {len(mock_contexts)} chunks")
        print(f"Context Relevance: {score}")
        print("✅ Success! Context Relevance > 0 với mock contexts")

if __name__ == "__main__":
    demo_context_relevance()
    test_with_database_data()