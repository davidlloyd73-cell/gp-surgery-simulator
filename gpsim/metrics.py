"""Phase 5 -- metrics, outputs and the scenario / frontier machinery.

Turns a raw simulation result into the grouped metric set the brief asks for
(responsiveness, continuity, capacity/workload, safety, efficiency), and provides
the helpers the dashboard and scenario comparisons build on:

* build_world()  -- generate population + demand once (cache across lever changes)
* run_scenario() -- run the book for a set of levers and return sim + metrics
* run_full()     -- build_world + run_scenario in one call
* frontier()     -- run several scenarios and return continuity-vs-responsiveness points
* sweep_lever()  -- vary one lever and return a metric table
* intraday()     -- per-hour arrivals / seen / queue for one day (the 8am rush)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_config
from .demand import generate_demand
from .population import generate_population
from .simulation import simulate

# canonical roles in display order
GP_ROLES = ("gp_partner", "salaried_gp", "locum_gp")


def _clinical_hours_freed(sim: dict, config: dict) -> float:
    """Clinician hours/working day freed by AI scribe + AI admin."""
    levers = sim["levers"]
    aie = config["ai_effects"]
    a_scribe = float(levers.get("ai_scribe_adoption", 0.0))
    a_admin = float(levers.get("ai_admin_adoption", 0.0))
    scribe_roles = set(aie["scribe_roles"])
    gain = aie["scribe_throughput_gain_at_full"] * a_scribe + aie["admin_clinician_time_saved_at_full"] * a_admin
    minutes = 0.0
    for c in sim["clinicians"]:
        if c.role in scribe_roles:
            minutes += c.slots_per_day * gain * c.appt_minutes
    # plus admin contacts AI fully handled -> their would-be appointment time
    ai_resolved = (sim["contacts"]["status"] == "ai_resolved").sum()
    n_working = int(sim["daily"]["working"].sum())
    nav_appt = config["practice"]["roles"]["care_navigator"]["appt_minutes"]
    minutes += (ai_resolved / max(n_working, 1)) * nav_appt
    return round(minutes / 60.0, 1)


def _cost(sim: dict, config: dict) -> dict:
    """Simple cost-style proxy for the year."""
    cc = config["costing"]["cost_per_contact"]
    c = sim["contacts"]
    seen = c[c["status"] == "seen"]
    appt_cost = float(seen["seen_role"].map(cc).fillna(0).sum())
    esc = int((c["status"] == "escalated").sum()) * config["costing"]["escalation_cost"]
    div = int(((c["status"] == "diverted") | (c["status"] == "ai_resolved")).sum()) * config["costing"]["diverted_cost"]
    total = appt_cost + esc + div
    list_size = int(config["population"]["list_size"])
    return {
        "total_cost_year": round(total),
        "cost_per_patient_year": round(total / max(list_size, 1), 2),
        "appointment_cost_year": round(appt_cost),
        "escalation_cost_year": round(esc),
    }


def compute_metrics(sim: dict, population: pd.DataFrame, config: dict) -> dict:
    """Group the KPIs into the brief's five families (+ levers echo)."""
    k = sim["kpis"]
    c = sim["contacts"]
    seen = c[c["status"] == "seen"]
    m = k["total_contacts"]

    # time-to-first-contact across all seen (same-day = 0 days)
    ttc = seen["wait_days"]
    dealt_same_day = float(((c["same_day"] & (c["status"] == "seen"))
                            | c["status"].isin(["diverted", "ai_resolved"])).sum() / m)

    # safety: were genuinely urgent (high/emergency acuity) contacts dealt with?
    high = c["acuity"].isin(["high", "emergency"])
    high_seen = float((high & (c["status"] == "seen")).sum() / max(high.sum(), 1))
    n_tri = k.get("n_triaged", 0)
    appropriate_triage = (round(1.0 - (k["under_triage_count"] + k["over_triage_count"]) / n_tri, 3)
                          if n_tri > 0 else None)

    # capacity: contacts per GP per day vs safe limit
    fp = sim["focus_profile"]
    peak_queue = int(fp["qlen"].max()) if fp is not None else 0

    responsiveness = {
        "mean_time_to_contact_days": round(float(ttc.mean()) if len(ttc) else 0.0, 2),
        "median_time_to_contact_days": float(ttc.median()) if len(ttc) else 0.0,
        "pct_same_day": k["same_day_rate"],
        "pct_dealt_same_day": round(dealt_same_day, 3),
        "pct_within_2_days": k["pct_seen_within_2_days"],
        "pct_within_7_days": round(float(((c["wait_days"] >= 0) & (c["wait_days"] <= 7)).sum() / m), 3),
        "mean_routine_wait_days": k["mean_routine_wait_days"],
        "mean_sameday_wait_min": k["mean_sameday_wait_min"],
        "peak_intraday_queue": peak_queue,
        "did_not_wait_rate": k["escalation_rate"],
        "unmet_rate": k["unmet_rate"],
    }
    continuity = {
        "upc_overall": k["continuity_upc"],
        "upc_high_need": k["continuity_upc_high_need"],
        "pct_chronic_with_usual": k["pct_chronic_with_usual"],
    }
    capacity = {
        "overall_utilisation": k["overall_utilisation"],
        "gp_utilisation": k["gp_utilisation"],
        "locum_share_of_appts": k["locum_share_of_appts"],
        "safe_limit_breach_rate": k["safe_limit_breach_rate"],
        "max_load_vs_safe": k["max_load_vs_safe"],
        "unmet_count": int((c["status"] == "unmet").sum()),
        "role_utilisation": dict(zip(sim["role_util"]["role"], sim["role_util"]["utilisation"])),
    }
    safety = {
        "escalations_to_111_ae": int((c["status"] == "escalated").sum()),
        "escalation_rate": k["escalation_rate"],
        "under_triaged_urgent": k["under_triage_count"],
        "missed_urgent_total": k["missed_urgent_count"],
        "appropriate_triage_rate": appropriate_triage,
        "pct_high_acuity_seen": round(high_seen, 3),
    }
    efficiency = {
        "clinician_hours_freed_per_day": _clinical_hours_freed(sim, config),
        "resolved_no_appt_rate": k["resolved_no_appt_rate"],
        "diversion_rate": k["diversion_rate"],
        "ai_admin_resolved_rate": k["ai_admin_resolved_rate"],
        "triage_capacity_burden_slots_per_day": k["triage_capacity_burden_slots_per_day"],
        **_cost(sim, config),
    }
    return {
        "levers": {kk: k[kk] for kk in (
            "triage_model", "ai_scribe_adoption", "ai_triage_adoption",
            "ai_triage_accuracy", "ai_admin_adoption", "continuity_policy_strength")},
        "responsiveness": responsiveness,
        "continuity": continuity,
        "capacity": capacity,
        "safety": safety,
        "efficiency": efficiency,
    }


# ---------------------------------------------------------------------------
# scenario machinery
# ---------------------------------------------------------------------------
def build_world(config: dict | None = None, seed: int | None = None) -> dict:
    """Generate population + demand once; reuse across lever changes."""
    if config is None:
        config = load_config()
    pop = generate_population(config, seed=seed)
    con = generate_demand(pop, config, seed=seed)
    return {"population": pop, "contacts": con, "config": config}


def run_scenario(world: dict, levers: dict | None = None, seed: int | None = None,
                 focus_day: int | None = None) -> dict:
    """Run the appointment book for a set of levers using a prebuilt world."""
    config = world["config"]
    sim = simulate(world["population"], world["contacts"], config,
                   seed=seed, levers=levers, focus_day=focus_day)
    metrics = compute_metrics(sim, world["population"], config)
    return {"sim": sim, "metrics": metrics}


def run_full(config: dict | None = None, levers: dict | None = None,
             seed: int | None = None) -> dict:
    return run_scenario(build_world(config, seed), levers, seed)


def frontier(world: dict, scenarios: list[tuple[str, dict]],
             seed: int | None = None) -> pd.DataFrame:
    """Run several named scenarios; return continuity-vs-responsiveness points."""
    rows = []
    for name, levers in scenarios:
        mk = run_scenario(world, levers, seed)["metrics"]
        rows.append({
            "scenario": name,
            "responsiveness_same_day": mk["responsiveness"]["pct_same_day"],
            "continuity_chronic_usual": mk["continuity"]["pct_chronic_with_usual"],
            "upc_high_need": mk["continuity"]["upc_high_need"],
            "gp_utilisation": mk["capacity"]["gp_utilisation"],
            "mean_wait_days": mk["responsiveness"]["mean_routine_wait_days"],
            "missed_urgent": mk["safety"]["missed_urgent_total"],
            "cost_per_patient": mk["efficiency"]["cost_per_patient_year"],
        })
    return pd.DataFrame(rows)


def sweep_lever(world: dict, lever: str, values: list, base_levers: dict | None = None,
                seed: int | None = None) -> pd.DataFrame:
    """Vary one lever across `values`; return a flat metric table for plotting."""
    base = dict(base_levers or {})
    rows = []
    for v in values:
        lev = {**base, lever: v}
        mk = run_scenario(world, lev, seed)["metrics"]
        rows.append({
            lever: v,
            "pct_same_day": mk["responsiveness"]["pct_same_day"],
            "mean_wait_days": mk["responsiveness"]["mean_routine_wait_days"],
            "upc_high_need": mk["continuity"]["upc_high_need"],
            "pct_chronic_with_usual": mk["continuity"]["pct_chronic_with_usual"],
            "gp_utilisation": mk["capacity"]["gp_utilisation"],
            "escalation_rate": mk["safety"]["escalation_rate"],
            "missed_urgent": mk["safety"]["missed_urgent_total"],
            "clinician_hours_freed_per_day": mk["efficiency"]["clinician_hours_freed_per_day"],
            "cost_per_patient": mk["efficiency"]["cost_per_patient_year"],
        })
    return pd.DataFrame(rows)


def intraday(sim: dict, day: int | None = None) -> pd.DataFrame:
    """Per-hour arrivals, same-day seen and mean queue length for one day."""
    fp = sim["focus_profile"]
    if day is None and fp is not None:
        day = fp["day"]
    c = sim["contacts"]
    sub = c[c["day"] == day]
    hours = np.arange(24)
    arrivals = np.array([(sub["arrival_hour"].astype(int) == h).sum() for h in hours])
    seen_sd = np.array([((sub["arrival_hour"].astype(int) == h) & sub["same_day"]).sum() for h in hours])
    queue = np.zeros(24)
    if fp is not None and fp["day"] == day:
        mins, qlen = fp["minutes"], fp["qlen"]
        for h in hours:
            mask = (mins >= h * 60) & (mins < (h + 1) * 60)
            queue[h] = qlen[mask].mean() if mask.any() else 0.0
    return pd.DataFrame({"hour": hours, "arrivals": arrivals,
                         "seen_same_day": seen_sd, "mean_queue": np.round(queue, 2)})


if __name__ == "__main__":
    world = build_world()
    res = run_scenario(world)
    import json
    print(json.dumps(res["metrics"], indent=2, default=str))
