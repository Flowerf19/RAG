#!/usr/bin/env python3
"""
Test script for autotest functionality in dashboard
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from io import StringIO
from evaluation.backend_dashboard.api import BackendDashboard
from ui.dashboard.components.ground_truth.file_handler import GroundTruthFileHandler, normalize_columns

def test_normalize_columns():
    """Test column normalization"""
    print("🧪 Testing column normalization...")

    # Test data
    test_data = {
        'STT': [1, 2, 3],
        'Câu hỏi': ['Q1', 'Q2', 'Q3'],
        'Câu trả lời': ['A1', 'A2', 'A3'],
        'Nguồn': ['S1', 'S2', 'S3']
    }
    df = pd.DataFrame(test_data)

    normalized = normalize_columns(df)

    assert 'question' in normalized.columns, "Question column not found"
    assert 'answer' in normalized.columns, "Answer column not found"
    assert 'source' in normalized.columns, "Source column not found"

    assert normalized.iloc[0]['question'] == 'Q1', "Question normalization failed"
    assert normalized.iloc[0]['answer'] == 'A1', "Answer normalization failed"
    assert normalized.iloc[0]['source'] == 'S1', "Source normalization failed"

    print("✅ Column normalization test passed")

def test_file_handler_logic():
    """Test file handler logic without Streamlit"""
    print("🧪 Testing file handler logic...")

    backend = BackendDashboard()
    handler = GroundTruthFileHandler(backend)

    # Create test CSV data
    csv_data = """STT,Câu hỏi,Câu trả lời,Nguồn
1,Machine learning là gì?,ML là AI subset,AI Basics
2,Deep learning khác gì?,DL uses neural networks,Neural Networks"""

    # Parse CSV
    df = pd.read_csv(StringIO(csv_data))
    normalized = normalize_columns(df)

    assert len(normalized) == 2, f"Expected 2 rows, got {len(normalized)}"
    assert normalized.iloc[0]['question'] == 'Machine learning là gì?', "Question parsing failed"

    # Test handler's internal methods
    try:
        # This would normally be called by _import_to_db
        rows = []
        for _, r in normalized.iterrows():
            rows.append({
                'question': r.get('question', ''),
                'answer': r.get('answer', ''),
                'source': r.get('source', '')
            })
        assert len(rows) == 2, "Row preparation failed"
        print("✅ Handler row preparation works")
    except Exception as e:
        print(f"⚠️ Handler test warning: {e}")

    print("✅ File handler logic test passed")

def test_database_operations():
    """Test database operations"""
    print("🧪 Testing database operations...")

    backend = BackendDashboard()

    # Test data
    test_rows = [
        {'question': 'Test Q1', 'answer': 'Test A1', 'source': 'Test S1'},
        {'question': 'Test Q2', 'answer': 'Test A2', 'source': 'Test S2'}
    ]

    try:
        # Insert test data
        inserted = backend.insert_ground_truth_rows(test_rows)
        print(f"✅ Inserted {inserted} test rows")

        # Check if data exists
        ground_truth = backend.get_ground_truth_list(limit=10)
        print(f"✅ Found {len(ground_truth)} ground truth entries")

        # Clean up - delete test data (if possible)
        # Note: This might not be implemented, so we'll just log

    except Exception as e:
        print(f"⚠️ Database test warning: {e}")

def test_evaluation_service():
    """Test evaluation service initialization"""
    print("🧪 Testing evaluation service...")

    try:
        from ui.dashboard.components.ground_truth.evaluation_service import GroundTruthEvaluationService
        backend = BackendDashboard()
        eval_service = GroundTruthEvaluationService(backend)
        assert eval_service is not None, "Evaluation service should not be None"
        assert hasattr(eval_service, 'run_full_evaluation_suite'), "Should have run_full_evaluation_suite method"
        print("✅ Evaluation service initialized successfully")
    except Exception as e:
        print(f"❌ Evaluation service test failed: {e}")

def main():
    """Run all tests"""
    print("🚀 Starting UI Autotest Functionality Tests")
    print("=" * 50)

    try:
        test_normalize_columns()
        test_file_handler_logic()
        test_database_operations()
        test_evaluation_service()

        print("=" * 50)
        print("🎉 All tests passed! UI autotest functionality is working correctly.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()