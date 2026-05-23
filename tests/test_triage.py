"""Phase 4 tests -- triage model, AI levers and continuity."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpsim.config import load_config  # noqa: E402
from gpsim.demand import generate_demand  # noqa: E402
from gpsim.population import generate_population  # noqa: E402
from gpsim.simulation import simulate  # noqa: E402


def _setup(list_size=6000):
    cfg = load_config()
    cfg["population"]["list_size"] = list_size
    pop = generate_population(cfg, seed=42)
    con = generate_demand(pop, cfg, seed=42)
    return cfg, pop, con


def _kpis(cfg, pop, con, **levers):
    return simulate(pop, con, cfg, seed=42, levers=levers)["kpis"]


def test_total_triage_diverts_and_frees_gps():
    cfg, pop, con = _setup()
    trad = _kpis(cfg, pop, con, triage_model="traditional")
    tri = _kpis(cfg, pop, con, triage_model="total_triage")
    assert tri["diversion_rate"] > 0.05            # demand diverted/signposted
    assert tri["gp_utilisation"] < trad["gp_utilisation"]   # GPs offloaded
    assert tri["triage_capacity_burden_slots_per_day"] > 0  # triage costs time


def test_continuity_policy_raises_continuity():
    cfg, pop, con = _setup()
    lo = _kpis(cfg, pop, con, triage_model="traditional", continuity_policy_strength=0.0)
    hi = _kpis(cfg, pop, con, triage_model="traditional", continuity_policy_strength=1.0)
    assert hi["pct_chronic_with_usual"] > lo["pct_chronic_with_usual"]
    assert hi["continuity_upc_high_need"] >= lo["continuity_upc_high_need"]


def test_continuity_costs_responsiveness():
    """The designed-in tension: more continuity -> less same-day access."""
    cfg, pop, con = _setup()
    lo = _kpis(cfg, pop, con, triage_model="traditional", continuity_policy_strength=0.0)
    hi = _kpis(cfg, pop, con, triage_model="traditional", continuity_policy_strength=1.0)
    assert hi["same_day_rate"] < lo["same_day_rate"]


def test_ai_scribe_raises_capacity():
    cfg, pop, con = _setup()
    base = simulate(pop, con, cfg, seed=42, levers={"ai_scribe_adoption": 0.0})
    boosted = simulate(pop, con, cfg, seed=42, levers={"ai_scribe_adoption": 1.0})
    gp = ["gp_partner", "salaried_gp", "locum_gp"]
    cap0 = base["role_util"].set_index("role").loc[gp, "capacity_per_day"].sum()
    cap1 = boosted["role_util"].set_index("role").loc[gp, "capacity_per_day"].sum()
    assert cap1 > cap0


def test_ai_admin_resolves_admin_contacts():
    cfg, pop, con = _setup()
    k0 = _kpis(cfg, pop, con, ai_admin_adoption=0.0)
    k1 = _kpis(cfg, pop, con, ai_admin_adoption=1.0)
    assert k0["ai_admin_resolved_rate"] == 0.0
    assert k1["ai_admin_resolved_rate"] > 0.0


def test_ai_triage_accuracy_drives_safety():
    """Lower AI-triage accuracy must increase under-triaged urgent cases."""
    cfg, pop, con = _setup()
    good = _kpis(cfg, pop, con, triage_model="total_triage",
                 ai_triage_adoption=1.0, ai_triage_accuracy=0.99)
    bad = _kpis(cfg, pop, con, triage_model="total_triage",
                ai_triage_adoption=1.0, ai_triage_accuracy=0.60)
    assert bad["under_triage_count"] > good["under_triage_count"]


def test_levers_reproducible():
    cfg, pop, con = _setup(5000)
    lev = {"triage_model": "total_triage", "ai_scribe_adoption": 0.5,
           "ai_triage_adoption": 0.5, "continuity_policy_strength": 0.7}
    a = simulate(pop, con, cfg, seed=9, levers=lev)["kpis"]
    b = simulate(pop, con, cfg, seed=9, levers=lev)["kpis"]
    assert a == b
