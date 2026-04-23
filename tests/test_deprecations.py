"""Guardrails against reintroducing deprecated pandas/streamlit APIs."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = [REPO_ROOT / "stockiq"]


def _iter_py_files():
    for root in SCAN_DIRS:
        yield from root.rglob("*.py")


def test_no_pandas_fillna_method_kwarg():
    """pandas >= 2.2 removed fillna(method=...) -- use .ffill()/.bfill()."""
    offenders = []
    for path in _iter_py_files():
        text = path.read_text()
        if "fillna(method=" in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"fillna(method=...) reintroduced in: {offenders}"


def test_no_streamlit_use_container_width():
    """Streamlit deprecated use_container_width; use width='stretch'/'content'."""
    offenders = []
    for path in _iter_py_files():
        text = path.read_text()
        if "use_container_width" in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"use_container_width reintroduced in: {offenders}"
