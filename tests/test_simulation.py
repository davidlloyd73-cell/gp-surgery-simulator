"""Smoke + sanity tests for the Phase 3 appointment-book simulation."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpsim.config import load_config  # noqa: E402
from gpsim.demand import generate_demand  # noqa: E402
from gpsim.population import generate_population  # noqa: E402
from gpsim.simulation import simulate  # noqa: E402


def _run(mode="traditional", enforce=False, list_size=6000):
    cfg = load_config()
    cfg["population"]["list_size"] = list_size
    cfg["practice"]["enforce_safe_limits"] = enforce
    pop = generate_population(cfg, seed=42)
    con = generate_demand(pop, cfg, seed=42)
    return cfg, simulate(pop, con, cfg, seed=42, routing_mode=mode)


def test_runs_and_returns_expected_keys():
    _, res = _run()
    for key in ("contacts", "daily", "role_util", "kpis", "clinicians"):
        assert key in res
    assert len(res["contacts"]) > 0


def test_reproducible():
    cfg = load_config()
    cfg["population"]["list_size"] = 5000
    pop = generate_population(cfg, seed=5)
    con = generate_demand(pop, cfg, seed=5)
    a = simulate(pop, con, cfg, seed=5)
    b = simulate(pop, con, cfg, seed=5)
    assert a["contacts"]["status"].equals(b["contacts"]["status"])
    assert a["kpis"] == b["kpis"]


def test_outcomes_are_sane():
    _, res = _run()
    k = res["kpis"]
    assert k["seen_rate"] > 0.9
    assert k["escalation_rate"] < 0.1
    assert k["unmet_rate"] < 0.05
    assert 0.0 <= k["overall_utilisation"] <= 1.0
    assert 0.0 <= k["gp_utilisation"] <= 1.0


def test_traditional_is_gp_heavy_triage_diverts():
    _, trad = _run("traditional")
    _, tri = _run("triage")
    # traditional concentrates work on GPs; triage spreads it out
    assert trad["kpis"]["gp_utilisation"] > tri["kpis"]["gp_utilisation"]
    ru_t = trad["role_util"].set_index("role")["utilisation"]
    ru_g = tri["role_util"].set_index("role")["utilisation"]
    # FCP / pharmacist are essentially unused in traditional but used in triage
    assert ru_g["fcp_physio"] > ru_t["fcp_physio"]
    assert ru_g["clinical_pharmacist"] > ru_t["clinical_pharmacist"]


def test_wait_days_consistency():
    _, res = _run()
    c = res["contacts"]
    seen = c[c.status == "seen"]
    assert (seen["wait_days"] == (seen["day_seen"] - seen["day"])).all()
    assert (seen["same_day"] == (seen["wait_days"] == 0)).all()
    not_seen = c[c.status != "seen"]
    assert (not_seen["wait_days"] == -1).all()


def test_safe_limits_cap_load_when_enforced():
    _, res = _run(enforce=True)
    # no clinician-day should exceed its safe daily contact limit
    assert res["kpis"]["max_load_vs_safe"] <= 1.0
