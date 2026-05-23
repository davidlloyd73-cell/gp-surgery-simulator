"""Phase 6 tests -- stress scenarios."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpsim.config import load_config  # noqa: E402
from gpsim.scenarios import (  # noqa: E402
    apply_scenario, compare_scenarios, list_scenarios, run_with_scenario,
)


def _small(cfg):
    cfg["population"]["list_size"] = 6000
    return cfg


def test_list_includes_baseline_and_configured():
    names = list_scenarios()
    assert "baseline" in names
    assert "winter_surge" in names and "gp_vacancy" in names


def test_apply_scenario_does_not_mutate_base():
    cfg = load_config()
    base_list = cfg["population"]["list_size"]
    base_gp = cfg["practice"]["roles"]["salaried_gp"]["clinicians"]
    apply_scenario(cfg, "list_growth")
    apply_scenario(cfg, "gp_vacancy")
    assert cfg["population"]["list_size"] == base_list
    assert cfg["practice"]["roles"]["salaried_gp"]["clinicians"] == base_gp


def test_gp_vacancy_removes_a_gp():
    cfg = load_config()
    n0 = cfg["practice"]["roles"]["salaried_gp"]["clinicians"]
    mod = apply_scenario(cfg, "gp_vacancy")
    assert mod["practice"]["roles"]["salaried_gp"]["clinicians"] == n0 - 1


def test_winter_surge_increases_pressure():
    cfg = _small(load_config())
    base = run_with_scenario("baseline", cfg, seed=42)["metrics"]
    winter = run_with_scenario("winter_surge", cfg, seed=42)["metrics"]
    assert winter["responsiveness"]["pct_same_day"] < base["responsiveness"]["pct_same_day"]
    assert winter["safety"]["escalations_to_111_ae"] > base["safety"]["escalations_to_111_ae"]


def test_pandemic_shifts_channels_online():
    cfg = _small(load_config())
    base = run_with_scenario("baseline", cfg, seed=42)["sim"]["contacts"]
    pan = run_with_scenario("pandemic_spike", cfg, seed=42)["sim"]["contacts"]
    base_online = (base["channel"] == "online").mean()
    pan_online = (pan["channel"] == "online").mean()
    assert pan_online > base_online


def test_compare_returns_row_per_scenario():
    cfg = _small(load_config())
    df = compare_scenarios(["baseline", "gp_vacancy", "list_growth"], cfg, seed=42)
    assert list(df["scenario"]) == ["baseline", "gp_vacancy", "list_growth"]
    assert "pct_same_day" in df.columns
