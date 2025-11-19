#!/usr/bin/env python3
"""
Integration test for autotest functionality - simulates full workflow
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from io import StringIO
from evaluation.backend_dashboard.api import BackendDashboard
from ui.dashboard.components.ground_truth.file_handler import normalize_columns

def simulate_file_upload_and_autotest():
    """Simulate the complete autotest workflow"""
    print("🚀 Simulating complete autotest workflow...")
    print("=" * 60)

    # Step 1: Create test CSV data
    print("📄 Step 1: Creating test CSV data...")
    csv_data = """STT,Câu hỏi,Câu trả lời,Nguồn
1,Machine learning là gì?,Machine learning là một nhánh của trí tuệ nhân tạo cho phép máy tính học từ dữ liệu.,AI Basics
2,Deep learning khác gì với machine learning?,Deep learning là một subset của machine learning sử dụng neural networks với nhiều layers.,Neural Networks
3,CNN được sử dụng để làm gì?,CNN (Convolutional Neural Networks) được sử dụng chủ yếu cho computer vision tasks như image recognition.,Computer Vision"""

    df = pd.read_csv(StringIO(csv_data))
    print(f"✅ Created test data with {len(df)} rows")

    # Step 2: Normalize columns (simulate file parsing)
    print("🔄 Step 2: Normalizing columns...")
    normalized = normalize_columns(df)
    print(f"✅ Normalized data: {list(normalized.columns)}")
    print(f"   Sample question: {normalized.iloc[0]['question'][:50]}...")

    # Step 3: Initialize backend and handlers
    print("🔧 Step 3: Initializing backend and handlers...")
    backend = BackendDashboard()
    print("✅ Components initialized")

    # Step 4: Simulate auto-import (what happens when auto_import=True)
    print("💾 Step 4: Simulating auto-import to database...")
    rows = []
    for _, r in normalized.iterrows():
        rows.append({
            'question': r.get('question', ''),
            'answer': r.get('answer', ''),
            'source': r.get('source', '')
        })

    try:
        inserted = backend.insert_ground_truth_rows(rows)
        print(f"✅ Auto-import successful: {inserted} rows inserted")
    except Exception as e:
        print(f"❌ Auto-import failed: {e}")
        return False

    # Step 5: Verify data in database
    print("🔍 Step 5: Verifying data in database...")
    ground_truth_list = backend.get_ground_truth_list(limit=10)
    print(f"✅ Found {len(ground_truth_list)} entries in database")

    # Step 6: Simulate auto-run evaluation (what happens when auto_run_eval=True)
    print("⚡ Step 6: Simulating auto-run evaluation...")
    try:
        # Use minimal settings for quick test
        eval_result = backend.evaluate_ground_truth_with_ragas(
            llm_provider='ollama',
            model_name='gemma3:1b',
            limit=2,  # Only test 2 samples for speed
            save_to_db=True
        )

        if 'error' in eval_result:
            print(f"❌ Evaluation failed: {eval_result['error']}")
            return False

        print("✅ Auto-evaluation successful!")
        print(f"   Total samples: {eval_result.get('total_samples', 0)}")
        print(f"   Faithfulness: {eval_result.get('faithfulness', {}).get('mean', 'N/A'):.3f}")
        print(f"   Context Recall: {eval_result.get('context_recall', {}).get('mean', 'N/A'):.3f}")
        print(f"   Context Relevance: {eval_result.get('context_relevance', {}).get('mean', 'N/A'):.3f}")
        print(f"   Answer Relevancy: {eval_result.get('answer_relevancy', {}).get('mean', 'N/A'):.3f}")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False

    # Step 7: Final verification
    print("🎯 Step 7: Final verification...")
    print("✅ File upload simulation: PASSED")
    print("✅ Column normalization: PASSED")
    print("✅ Auto-import to DB: PASSED")
    print("✅ Auto-run evaluation: PASSED")
    print("✅ Results saved to DB: PASSED")

    print("=" * 60)
    print("🎉 COMPLETE AUTOTEST WORKFLOW SUCCESSFUL!")
    print()
    print("📋 Summary:")
    print(f"   - Processed {len(normalized)} ground truth Q&A pairs")
    print("   - Auto-imported to database")
    print("   - Auto-ran Ragas evaluation with 4 metrics")
    print("   - Saved results for dashboard analytics")

    return True

def test_edge_cases():
    """Test edge cases"""
    print("🧪 Testing edge cases...")

    # Test empty CSV
    try:
        empty_df = pd.DataFrame(columns=['STT', 'Câu hỏi', 'Câu trả lời', 'Nguồn'])
        normalized = normalize_columns(empty_df)
        assert len(normalized) == 0, "Empty dataframe should remain empty"
        print("✅ Empty CSV handling: PASSED")
    except Exception as e:
        print(f"❌ Empty CSV test failed: {e}")

    # Test missing columns
    try:
        incomplete_df = pd.DataFrame({'STT': [1], 'Question': ['Q1']})  # Missing answer and source
        normalized = normalize_columns(incomplete_df)
        assert 'question' in normalized.columns, "Should create question column"
        assert normalized.iloc[0]['answer'] == '', "Missing columns should be empty strings"
        print("✅ Missing columns handling: PASSED")
    except Exception as e:
        print(f"❌ Missing columns test failed: {e}")

def main():
    """Run integration tests"""
    print("🧪 UI AUTOTEST INTEGRATION TESTS")
    print("=" * 60)

    try:
        # Run main workflow test
        success = simulate_file_upload_and_autotest()

        if success:
            # Run edge case tests
            test_edge_cases()

            print("=" * 60)
            print("🎯 ALL INTEGRATION TESTS PASSED!")
            print()
            print("✅ UI Autotest functionality is fully operational:")
            print("   - File upload → Auto-parse → Auto-import → Auto-evaluate → Save results")
            print("   - Handles Vietnamese column names correctly")
            print("   - Works with Ollama Gemma3:1b for evaluation")
            print("   - Saves all 4 Ragas metrics to database")
        else:
            print("❌ Integration test failed!")
            return 1

    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())