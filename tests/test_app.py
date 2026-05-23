"""Phase 7 -- smoke test the Streamlit dashboard runs end to end.

Uses Streamlit's headless AppTest harness to execute app.py and assert it renders
without raising. Slow-ish (runs a full year + the frontier), so it's a single
integration check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_dashboard_runs_without_exception():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, f"dashboard raised: {at.exception}"
    # KPI cards rendered
    assert len(at.metric) == 8
    labels = [m.label for m in at.metric]
    assert "Same-day access" in labels
    assert "GP utilisation" in labels


def test_continuity_slider_trades_off_responsiveness():
    """Driving continuity up should reduce same-day access in the live app."""
    at = AppTest.from_file(APP, default_timeout=180).run()
    # find the continuity slider and the same-day metric at low vs high continuity
    cont = [s for s in at.slider if "Continuity" in s.label][0]
    cont.set_value(0.0).run()
    low = [m.value for m in at.metric if m.label == "Same-day access"][0]
    cont.set_value(1.0).run()
    high = [m.value for m in at.metric if m.label == "Same-day access"][0]
    assert int(high.strip("%")) <= int(low.strip("%"))
