"""Phase 5 tests -- grouped metrics and scenario/frontier machinery."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpsim.config import load_config  # noqa: E402
from gpsim.metrics import (  # noqa: E402
    build_world, compute_metrics, frontier, intraday, run_scenario, sweep_lever,
)


def _world(list_size=6000):
    cfg = load_config()
    cfg["population"]["list_size"] = list_size
    return build_world(cfg, seed=42)


def test_metrics_has_all_groups():
    world = _world()
    mk = run_scenario(world, seed=42)["metrics"]
    for group in ("responsiveness", "continuity", "capacity", "safety", "efficiency", "levers"):
        assert group in mk
    assert 0.0 <= mk["responsiveness"]["pct_same_day"] <= 1.0
    assert mk["efficiency"]["cost_per_patient_year"] > 0


def test_frontier_returns_points():
    world = _world()
    scenarios = [
        ("A", {"triage_model": "traditional", "continuity_policy_strength": 0.0}),
        ("B", {"triage_model": "traditional", "continuity_policy_strength": 1.0}),
    ]
    fr = frontier(world, scenarios, seed=42)
    assert len(fr) == 2
    for col in ("responsiveness_same_day", "continuity_chronic_usual", "upc_high_need"):
        assert col in fr.columns
    # the trade-off: more continuity -> higher continuity, lower responsiveness
    a = fr[fr.scenario == "A"].iloc[0]
    b = fr[fr.scenario == "B"].iloc[0]
    assert b["continuity_chronic_usual"] > a["continuity_chronic_usual"]
    assert b["responsiveness_same_day"] < a["responsiveness_same_day"]


def test_sweep_lever_monotone_continuity():
    world = _world()
    sw = sweep_lever(world, "continuity_policy_strength", [0.0, 0.5, 1.0],
                     base_levers={"triage_model": "traditional"}, seed=42)
    assert sw["pct_chronic_with_usual"].is_monotonic_increasing


def test_ai_scribe_frees_clinician_time():
    world = _world()
    base = run_scenario(world, {"ai_scribe_adoption": 0.0}, seed=42)["metrics"]
    boosted = run_scenario(world, {"ai_scribe_adoption": 1.0}, seed=42)["metrics"]
    assert base["efficiency"]["clinician_hours_freed_per_day"] == 0.0
    assert boosted["efficiency"]["clinician_hours_freed_per_day"] > 0.0


def test_intraday_arrivals_peak_at_8am():
    world = _world()
    sim = run_scenario(world, seed=42)["sim"]
    df = intraday(sim)
    assert len(df) == 24
    assert int(df.loc[df["arrivals"].idxmax(), "hour"]) == 8


def test_appropriate_triage_only_in_total_triage():
    world = _world()
    trad = run_scenario(world, {"triage_model": "traditional"}, seed=42)["metrics"]
    tri = run_scenario(world, {"triage_model": "total_triage"}, seed=42)["metrics"]
    assert trad["safety"]["appropriate_triage_rate"] is None
    assert tri["safety"]["appropriate_triage_rate"] is not None
    assert 0.0 <= tri["safety"]["appropriate_triage_rate"] <= 1.0
