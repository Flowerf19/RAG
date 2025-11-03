"""
PDF Extract Kit - Simplified module for our RAG system
Only exports the components we actually use.
"""

import os
import sys

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Only import what we actually use in our system
try:
    from .registry import TASK_REGISTRY, MODEL_REGISTRY
    from .utils.config_loader import get_config
    __all__ = ['TASK_REGISTRY', 'MODEL_REGISTRY', 'get_config']
except ImportError as e:
    # Graceful degradation if components not available
    print(f"Warning: Some PDF-Extract-Kit components not available: {e}")
    TASK_REGISTRY = None
    MODEL_REGISTRY = None
    get_config = None
    __all__ = []