from __future__ import annotations

import pytest

from optulus_anchor import enable_persistent_tracelog


@pytest.fixture(autouse=True)
def _tracelog_test_isolation(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a per-test trace root and keep programmatic enable as the default."""
    monkeypatch.setenv("OPTULUS_ANCHOR_TRACE_DIR", str(tmp_path))
    monkeypatch.delenv("OPTULUS_ANCHOR_NO_TRACE", raising=False)
    enable_persistent_tracelog()
