"""Phase 6 -- stress scenarios.

Each scenario is a named modifier applied on top of the baseline config (a deep
copy, so the baseline is never mutated). Scenarios can change demand (volume,
seasonality, channel mix) and/or supply (staff). The dashboard and the
comparison helper run baseline vs scenario with the *same* levers to show the
before/after on identical KPIs.
"""
from __future__ import annotations

import copy

import pandas as pd

from .config import load_config
from .metrics import build_world, run_scenario


def list_scenarios(config: dict | None = None) -> dict:
    """Return {name: description}, including the implicit 'baseline'."""
    config = config or load_config()
    out = {"baseline": "No shock -- the practice as configured"}
    for name, spec in config.get("scenarios", {}).items():
        out[name] = spec.get("description", name)
    return out


def apply_scenario(config: dict, name: str | None) -> dict:
    """Return a new config with the named scenario's modifiers applied."""
    cfg = copy.deepcopy(config)
    if not name or name == "baseline":
        return cfg
    spec = config["scenarios"][name]
    dem = cfg["demand"]
    roles = cfg["practice"]["roles"]

    if "demand_multiplier" in spec:
        dem["demand_multiplier"] = dem.get("demand_multiplier", 1.0) * spec["demand_multiplier"]
    if "winter_month_extra" in spec:
        for mi in (11, 0, 1):                       # Dec, Jan, Feb
            dem["month_multipliers"][mi] *= spec["winter_month_extra"]
    if "monday_extra" in spec:
        dem["weekday_multipliers"][0] *= spec["monday_extra"]
    if "post_holiday_extra" in spec:
        dem["post_holiday_surge"] *= spec["post_holiday_extra"]
    if "online_shift" in spec:
        s = spec["online_shift"]
        for grp in dem["channels_by_age"].values():
            move = min(grp["walk_in"], s)
            grp["walk_in"] -= move
            grp["online"] += move * 0.6
            grp["phone"] += move * 0.4
    if "staff_slot_factor" in spec:
        f = spec["staff_slot_factor"]
        for sp in roles.values():
            sp["slots_per_day"] = max(1, round(sp["slots_per_day"] * f))
    if "remove_clinicians" in spec:
        for role, n in spec["remove_clinicians"].items():
            roles[role]["clinicians"] = max(0, roles[role]["clinicians"] - n)
    if "list_size_factor" in spec:
        cfg["population"]["list_size"] = int(cfg["population"]["list_size"] * spec["list_size_factor"])
    return cfg


def run_with_scenario(name: str | None, base_config: dict | None = None,
                      levers: dict | None = None, seed: int | None = None) -> dict:
    """Build a world under the scenario's config and run it for the given levers."""
    base_config = base_config or load_config()
    cfg = apply_scenario(base_config, name)
    world = build_world(cfg, seed=seed)
    res = run_scenario(world, levers=levers, seed=seed)
    res["scenario"] = name or "baseline"
    return res


def _key_kpis(metrics: dict) -> dict:
    r, c, cap, s, e = (metrics["responsiveness"], metrics["continuity"],
                       metrics["capacity"], metrics["safety"], metrics["efficiency"])
    return {
        "pct_same_day": r["pct_same_day"],
        "mean_wait_days": r["mean_routine_wait_days"],
        "did_not_wait_rate": r["did_not_wait_rate"],
        "unmet_rate": r["unmet_rate"],
        "gp_utilisation": cap["gp_utilisation"],
        "max_load_vs_safe": cap["max_load_vs_safe"],
        "upc_high_need": c["upc_high_need"],
        "escalations_111_ae": s["escalations_to_111_ae"],
        "missed_urgent": s["missed_urgent_total"],
        "cost_per_patient": e["cost_per_patient_year"],
    }


def compare_scenarios(names: list[str], base_config: dict | None = None,
                      levers: dict | None = None, seed: int | None = None) -> pd.DataFrame:
    """Run baseline + each named scenario; return a KPI table (one row each)."""
    base_config = base_config or load_config()
    rows = []
    for name in names:
        res = run_with_scenario(name, base_config, levers, seed)
        rows.append({"scenario": res["scenario"], **_key_kpis(res["metrics"])})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = compare_scenarios(["baseline"] + list(load_config()["scenarios"].keys()))
    print(df.to_string(index=False))
