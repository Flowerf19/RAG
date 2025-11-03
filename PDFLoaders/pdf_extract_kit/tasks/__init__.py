"""
Tasks module - Only load what we actually use
"""

# Only import tasks we actually use in our system
try:
    from .layout_detection.task import LayoutDetectionTask
    _layout_available = True
except (ImportError, RuntimeError) as e:
    if "numpy" in str(e).lower():
        print("Warning: LayoutDetectionTask not available due to numpy version conflict")
    else:
        print(f"Warning: LayoutDetectionTask not available: {e}")
    LayoutDetectionTask = None
    _layout_available = False

try:
    from .formula_detection.task import FormulaDetectionTask
    _formula_available = True
except (ImportError, RuntimeError) as e:
    if "numpy" in str(e).lower():
        print("Warning: FormulaDetectionTask not available due to numpy version conflict")
    else:
        print(f"Warning: FormulaDetectionTask not available: {e}")
    FormulaDetectionTask = None
    _formula_available = False

try:
    from .formula_recognition.task import FormulaRecognitionTask
    _formula_recog_available = True
except (ImportError, RuntimeError) as e:
    if "numpy" in str(e).lower():
        print("Warning: FormulaRecognitionTask not available due to numpy version conflict")
    else:
        print(f"Warning: FormulaRecognitionTask not available: {e}")
    FormulaRecognitionTask = None
    _formula_recog_available = False

try:
    from .ocr.task import OCRTask
    _ocr_available = True
except (ImportError, RuntimeError) as e:
    if "numpy" in str(e).lower():
        print("Warning: OCRTask not available due to numpy version conflict")
    else:
        print(f"Warning: OCRTask not available: {e}")
    OCRTask = None
    _ocr_available = False

try:
    from .table_parsing.task import TableParsingTask
    _table_available = True
except (ImportError, RuntimeError) as e:
    if "numpy" in str(e).lower():
        print("Warning: TableParsingTask not available due to numpy version conflict")
    else:
        print(f"Warning: TableParsingTask not available: {e}")
    TableParsingTask = None
    _table_available = False

# Import registry for registration
try:
    from ..registry.registry import TASK_REGISTRY
except ImportError:
    TASK_REGISTRY = None

# Only export what we use
__all__ = []
if _layout_available:
    __all__.append("LayoutDetectionTask")
if _formula_available:
    __all__.append("FormulaDetectionTask")
if _formula_recog_available:
    __all__.append("FormulaRecognitionTask")
if _ocr_available:
    __all__.append("OCRTask")
if _table_available:
    __all__.append("TableParsingTask")

def load_task(name, cfg=None):
    """
    Load a task by name with optional config.
    Only supports tasks we actually use.
    """
    if TASK_REGISTRY is None:
        raise RuntimeError("Task registry not available")

    try:
        task_class = TASK_REGISTRY.get(name)
        return task_class(cfg) if cfg else task_class()
    except ValueError as e:
        raise ValueError(f"Task '{name}' not found or not supported. Available: {TASK_REGISTRY.list_items()}") from e
    """
    Example

    >>> task = load_task("formula_detection", cfg=None)
    """
    task_class = TASK_REGISTRY.get(name)
    task_instance = task_class(cfg)

    return task_instance
