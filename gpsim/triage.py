"""Phase 4 -- triage model, AI levers and continuity helpers.

Pure functions that the simulation calls to:
  * resolve the lever settings,
  * apply AI admin (auto-resolve some admin contacts),
  * run total triage (route / divert / detect under-triage) and cost its time,
  * convert AI adoption into effective clinician capacity,
  * set up continuity (usual clinician + per-contact attempt probability).

Keeping these here keeps the discrete-event engine in simulation.py readable.
"""
from __future__ import annotations

import numpy as np

from .practice import Clinician


def resolve_levers(config: dict, overrides: dict | None = None) -> dict:
    """Merge config['levers'] with any dashboard/scenario overrides."""
    levers = dict(config.get("levers", {}))
    if overrides:
        levers.update({k: v for k, v in overrides.items() if v is not None})
    return levers


# ---------------------------------------------------------------------------
# AI admin -- absorbs a share of admin contacts entirely (no appointment).
# ---------------------------------------------------------------------------
def apply_ai_admin(c_type: np.ndarray, status: np.ndarray, levers: dict,
                   config: dict, rng: np.random.Generator) -> None:
    """Mutate `status`: mark AI-resolved admin contacts as 'ai_resolved'."""
    a = float(levers.get("ai_admin_adoption", 0.0))
    if a <= 0:
        return
    frac = config["ai_effects"]["admin_autoresolve_at_full"] * a
    admin = c_type == "admin"
    draw = rng.random(c_type.shape[0]) < frac
    status[admin & draw] = "ai_resolved"


# ---------------------------------------------------------------------------
# Total triage -- assess every active contact, then route or divert.
# ---------------------------------------------------------------------------
def apply_total_triage(c_type: np.ndarray, c_acuity: np.ndarray,
                       status: np.ndarray, levers: dict, config: dict,
                       rng: np.random.Generator) -> dict:
    """Mutate `status` for triage outcomes; return diagnostics.

    Outcomes written to status (only where currently 'active'):
      * 'diverted'      -- correctly signposted to pharmacy/self-care/etc.
      * 'under_triaged' -- a high/emergency contact wrongly diverted (SAFETY)
      * (left 'active') -- routed onward to the appointment book; over-triaged
                           routed contacts are flagged in the returned 'over_triaged'.
    """
    n = c_type.shape[0]
    tcfg = config["triage"]
    aie = config["ai_effects"]
    active = status == "active"

    divf = np.array([tcfg["diversion_fraction"].get(t, 0.0) for t in c_type])
    low_acuity = c_acuity == "low"
    high_acuity = np.isin(c_acuity, ["high", "emergency"])

    # the "correct" triage decision: divert only genuinely low-value low-acuity work
    correct_divert = low_acuity & (rng.random(n) < divf)

    # who assessed it -- AI takes a share off humans at high adoption
    ai_frac = aie["triage_human_offload_at_full"] * float(levers.get("ai_triage_adoption", 0.0))
    ai_triaged = rng.random(n) < ai_frac
    acc = float(levers.get("ai_triage_accuracy", 0.92))
    human_acc = tcfg["human_triage_accuracy"]
    error = rng.random(n) < np.where(ai_triaged, 1.0 - acc, 1.0 - human_acc)

    # apply the decision, flipping it on a triage error
    final_divert = np.where(error, ~correct_divert, correct_divert)
    over_triaged = error & correct_divert            # should've diverted, routed instead
    became_diverted = error & (~correct_divert)      # should've routed, diverted instead
    under = final_divert & high_acuity               # a genuinely urgent case diverted

    final_divert &= active
    under &= active
    over_triaged &= active

    status[final_divert & ~under] = "diverted"
    status[under] = "under_triaged"

    n_triaged = int(active.sum())
    return {
        "n_triaged": n_triaged,
        "diverted": int((status == "diverted").sum()),
        "under_triaged": int(under.sum()),
        "over_triaged": int(over_triaged.sum()),
        "ai_triaged": int((ai_triaged & active).sum()),
    }


# ---------------------------------------------------------------------------
# Effective capacity -- AI scribe/admin raise it; triage burden lowers it.
# ---------------------------------------------------------------------------
def effective_slots(clinicians: list[Clinician], levers: dict, config: dict,
                    n_triaged_per_working_day: float,
                    total_triage: bool) -> np.ndarray:
    """Per-clinician effective daily slots after AI gains and triage cost."""
    aie = config["ai_effects"]
    a_scribe = float(levers.get("ai_scribe_adoption", 0.0))
    a_admin = float(levers.get("ai_admin_adoption", 0.0))
    a_triage = float(levers.get("ai_triage_adoption", 0.0))
    scribe_roles = set(aie["scribe_roles"])

    gain = (aie["scribe_throughput_gain_at_full"] * a_scribe
            + aie["admin_clinician_time_saved_at_full"] * a_admin)
    eff = np.array([c.slots_per_day * (1.0 + gain) if c.role in scribe_roles
                    else float(c.slots_per_day) for c in clinicians])

    if total_triage and n_triaged_per_working_day > 0:
        tcfg = config["triage"]
        human_frac = 1.0 - aie["triage_human_offload_at_full"] * a_triage
        human_tri = n_triaged_per_working_day * human_frac
        gp_tri = human_tri * tcfg["duty_gp_triage_share"]
        nav_tri = human_tri * (1.0 - tcfg["duty_gp_triage_share"])

        gp_idx = [i for i, c in enumerate(clinicians) if c.is_gp]
        nav_idx = [i for i, c in enumerate(clinicians) if c.role == "care_navigator"]
        if gp_idx:
            gp_appt = np.mean([clinicians[i].appt_minutes for i in gp_idx])
            per_gp = (gp_tri * tcfg["triage_minutes_clinical"] / gp_appt) / len(gp_idx)
            for i in gp_idx:
                eff[i] -= per_gp
        if nav_idx:
            nav_appt = np.mean([clinicians[i].appt_minutes for i in nav_idx])
            per_nav = (nav_tri * tcfg["triage_minutes_navigator"] / nav_appt) / len(nav_idx)
            for i in nav_idx:
                eff[i] -= per_nav

    return np.maximum(np.round(eff), 1).astype(int)


def triage_capacity_burden(clinicians: list[Clinician], levers: dict, config: dict,
                           n_triaged_per_working_day: float,
                           total_triage: bool) -> float:
    """Slot-equivalents per working day consumed by human triage (the cost)."""
    if not total_triage or n_triaged_per_working_day <= 0:
        return 0.0
    tcfg = config["triage"]
    aie = config["ai_effects"]
    a_triage = float(levers.get("ai_triage_adoption", 0.0))
    human_frac = 1.0 - aie["triage_human_offload_at_full"] * a_triage
    human_tri = n_triaged_per_working_day * human_frac
    gp_tri = human_tri * tcfg["duty_gp_triage_share"]
    nav_tri = human_tri * (1.0 - tcfg["duty_gp_triage_share"])
    gp_idx = [i for i, c in enumerate(clinicians) if c.is_gp]
    nav_idx = [i for i, c in enumerate(clinicians) if c.role == "care_navigator"]
    burden = 0.0
    if gp_idx:
        gp_appt = np.mean([clinicians[i].appt_minutes for i in gp_idx])
        burden += gp_tri * tcfg["triage_minutes_clinical"] / gp_appt
    if nav_idx:
        nav_appt = np.mean([clinicians[i].appt_minutes for i in nav_idx])
        burden += nav_tri * tcfg["triage_minutes_navigator"] / nav_appt
    return float(burden)


# ---------------------------------------------------------------------------
# Continuity -- assign a usual clinician and per-contact attempt probability.
# ---------------------------------------------------------------------------
def assign_usual_clinician(population, clinicians: list[Clinician], config: dict,
                           rng: np.random.Generator) -> np.ndarray:
    """Assign each patient (by position) a usual GP, load-balanced across GPs."""
    roles = set(config["continuity"]["usual_clinician_roles"])
    gp_ids = [c.cid for c in clinicians if c.role in roles]
    if not gp_ids:
        gp_ids = [c.cid for c in clinicians if c.is_gp]
    n = len(population)
    # weight by each GP's slots so busier-capacity GPs hold proportionally more
    weights = np.array([clinicians[i].slots_per_day for i in gp_ids], dtype=float)
    weights /= weights.sum()
    return np.array(gp_ids)[rng.choice(len(gp_ids), size=n, p=weights)]


def continuity_attempt(c_type: np.ndarray, cont_pref: np.ndarray, levers: dict,
                       config: dict, rng: np.random.Generator) -> np.ndarray:
    """Boolean per contact: should we try to book the patient's usual clinician?"""
    strength = float(levers.get("continuity_policy_strength", 0.0))
    tw = config["continuity"]["type_weight"]
    type_w = np.array([tw.get(t, 0.0) for t in c_type])
    p = strength * cont_pref * type_w
    return rng.random(c_type.shape[0]) < p
