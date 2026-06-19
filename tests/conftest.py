"""Pytest config: ensure project root is on sys.path so `app.*` imports work."""
import sys
from pathlib import Path

# Project root is the parent of tests/.
_ROOT = Path(__file__).resolve().parent.parent

# Strip other Karsa-adjacent paths that may shadow our `app` package.
sys.path[:] = [
    p for p in sys.path
    if "rag-chatbot" not in p and "streak-reminder" not in p
]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))