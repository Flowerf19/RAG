#!/usr/bin/env python3
"""
Test Ragas Evaluation Integration
=================================

This script tests the new Ragas-based evaluation system to ensure
it works correctly before removing the old evaluation modules.
"""

import os
import logging
from unittest.mock import patch
import importlib.util

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ragas_imports():
    """Test that Ragas modules can be imported correctly."""
    logger.info("Testing Ragas module imports...")

    try:
        # Test imports one by one to identify issues
        import ragas
        from ragas.metrics import faithfulness
        from langchain_google_genai import ChatGoogleGenerativeAI

        logger.info("✅ All Ragas imports successful")
        assert True
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        assert False, f"Import failed: {e}"


def test_ragas_evaluator_initialization():
    """Test RagasEvaluator initialization (without API key)."""
    logger.info("Testing RagasEvaluator initialization...")

    try:
        from evaluation.backend_dashboard.ragas_evaluator import RagasEvaluator

        # Test that it requires API key
        with patch.dict(os.environ, {}, clear=True):
            try:
                RagasEvaluator()
                assert False, "Should have failed without API key"
            except ValueError as e:
                if "GOOGLE_API_KEY" in str(e):
                    logger.info("✅ Correctly requires API key")
                else:
                    assert False, f"Unexpected error: {e}"

        # Test with mock API key (won't actually connect)
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key'}):
            try:
                RagasEvaluator()
                logger.info("✅ RagasEvaluator initializes with API key")
                assert True
            except Exception as e:
                logger.error(f"Failed to initialize with API key: {e}")
                assert False, f"Failed to initialize with API key: {e}"

    except Exception as e:
        logger.error(f"Evaluator initialization test failed: {e}")
        assert False, f"Evaluator initialization test failed: {e}"


def test_backend_dashboard_integration():
    """Test that BackendDashboard has the new Ragas method."""
    logger.info("Testing BackendDashboard integration...")

    try:
        from evaluation.backend_dashboard.api import BackendDashboard

        dashboard = BackendDashboard()

        # Check that the method exists
        if not hasattr(dashboard, 'evaluate_ground_truth_with_ragas'):
            logger.error("BackendDashboard missing evaluate_ground_truth_with_ragas method")
            assert False, "BackendDashboard missing evaluate_ground_truth_with_ragas method"

        logger.info("✅ BackendDashboard has Ragas evaluation method")
        assert True

    except Exception as e:
        logger.error(f"BackendDashboard integration test failed: {e}")
        assert False, f"BackendDashboard integration test failed: {e}"


def test_ground_truth_data_format():
    """Test that ground truth data is in correct format."""
    logger.info("Testing ground truth data format...")

    try:
        import json

        if not os.path.exists('ground_truth_ragas.json'):
            logger.error("Ground truth file not found")
            assert False, "Ground truth file not found"

        with open('ground_truth_ragas.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            logger.error("Ground truth data should be a list")
            assert False, "Ground truth data should be a list"

        if len(data) == 0:
            logger.error("Ground truth data is empty")
            assert False, "Ground truth data is empty"

        # Check first item structure
        item = data[0]
        required_fields = ['question', 'answer', 'contexts', 'ground_truth']

        for field in required_fields:
            if field not in item:
                logger.error(f"Missing required field: {field}")
                assert False, f"Missing required field: {field}"

        if not isinstance(item['contexts'], list):
            logger.error("contexts field should be a list")
            assert False, "contexts field should be a list"

        logger.info(f"✅ Ground truth data format valid ({len(data)} samples)")
        assert True

    except Exception as e:
        logger.error(f"Ground truth data format test failed: {e}")
        assert False, f"Ground truth data format test failed: {e}"


def run_all_tests():
    """Run all integration tests."""
    tests = [
        test_ragas_imports,
        test_ragas_evaluator_initialization,
        test_backend_dashboard_integration,
        test_ground_truth_data_format
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            logger.error(f"Test {test.__name__} crashed: {e}")
            results.append(False)

    return results


if __name__ == "__main__":
    print("🧪 Testing Ragas Evaluation Integration (Mock Tests)")
    print("=" * 55)

    test_results = run_all_tests()

    passed = sum(test_results)
    total = len(test_results)

    if passed == total:
        print(f"\n✅ All {total} tests passed! Ragas evaluation integration is ready.")
        print("📋 Next steps:")
        print("   1. Set GOOGLE_API_KEY environment variable")
        print("   2. Run actual evaluation: dashboard.evaluate_ground_truth_with_ragas()")
        print("   3. ✅ Old evaluation modules removed (semantic.py, recall.py, faithfulness.py)")
        print("   4. Update UI to use Ragas evaluation methods")
    else:
        print(f"\n❌ {total - passed} out of {total} tests failed!")
        print("Please check the logs above and fix issues before proceeding.")
        exit(1)