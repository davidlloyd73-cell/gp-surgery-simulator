"""Phase 3 -- the appointment-book simulation engine.

Two coupled mechanisms, each where it fits best:

* SAME-DAY access (urgent / emergency clinical demand) is simulated as a queue
  with **SimPy**: clinicians are resources, contacts are entities that arrive
  through the day (the 8am rush), pick the least-loaded eligible clinician, wait,
  and are either seen or -- if no reserved capacity is left or the clinic closes
  first -- escalate (a proxy for 111 / A&E / out-of-hours).

* ROUTINE demand is booked forward against each clinician's remaining daily
  capacity (a ledger), so the wait-in-days grows when demand outstrips supply,
  and requests waiting longer than the patience limit count as unmet.

Both honour scope-of-practice and the routing preference (traditional vs triage).
DNAs waste a booked slot. An optional BMA safe-working cap limits contacts per
clinician per day. Everything is reproducible from the global seed.

This is a deliberate simplification of a real appointment book (no part-time
rota detail, no within-day session boundaries beyond open/close, routine
appointments assumed to flow smoothly once booked). It is built to expose system
*dynamics and trade-offs*, not to reproduce any real practice's diary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import simpy

from .config import load_config, make_rng
from .demand import build_calendar
from .practice import (
    Clinician, build_clinicians, eligible_clinician_indices, expand_role_token,
    routing_for,
)
from .triage import (
    apply_ai_admin, apply_total_triage, assign_usual_clinician,
    continuity_attempt, effective_slots, resolve_levers, triage_capacity_burden,
)

# Clinical contact types whose urgent/emergency contacts contend for same-day
# access. Admin / self-care are always handled through the routine ledger.
SAMEDAY_TYPES = frozenset({"acute_minor", "acute_serious",
                           "chronic_review", "mental_health"})


def _sameday_eligibility(routing: dict, clinicians: list[Clinician]) -> dict:
    """type -> urgent-capable eligible clinician ids, in preference order."""
    out = {}
    for ctype in SAMEDAY_TYPES:
        cids = eligible_clinician_indices(ctype, routing, clinicians)
        out[ctype] = [c for c in cids if clinicians[c].urgent_capable]
    return out


def _simulate_sameday_day(day_rows, clinicians, sameday_cap, elig_map,
                          open_min, close_min):
    """SimPy queue for one day's same-day contacts.

    day_rows: list of (contact_index, ctype, arrival_min).
    sameday_cap: dict cid -> remaining same-day slots (mutated).
    Returns {contact_index: (status, cid, wait_min, start_min)} and a list of
    (time, +1/-1) queue events for building an intra-day profile.
    """
    env = simpy.Environment(initial_time=open_min)
    resources = {cid: simpy.Resource(env, capacity=1) for cid in sameday_cap}
    results: dict = {}
    q_events: list = []

    def patient(idx, ctype, arrival_min):
        arr = max(arrival_min, open_min)
        if env.now < arr:
            yield env.timeout(arr - env.now)
        elig = [cid for cid in elig_map.get(ctype, []) if sameday_cap.get(cid, 0) > 0]
        if not elig:
            results[idx] = ("escalated", -1, np.nan, np.nan)
            return
        cid = min(elig, key=lambda x: len(resources[x].queue) + len(resources[x].users))
        res = resources[cid]
        appt = clinicians[cid].appt_minutes
        q_events.append((env.now, 1))
        with res.request() as req:
            yield req
            q_events.append((env.now, -1))
            start = env.now
            if sameday_cap.get(cid, 0) <= 0 or start + appt > close_min:
                results[idx] = ("escalated", -1, np.nan, np.nan)
                return
            sameday_cap[cid] -= 1
            wait = start - arr
            yield env.timeout(appt)
            results[idx] = ("seen", cid, wait, start)

    for idx, ctype, arrival_min in day_rows:
        env.process(patient(idx, ctype, arrival_min))
    env.run(until=close_min)
    for idx, _, _ in day_rows:
        results.setdefault(idx, ("escalated", -1, np.nan, np.nan))
    return results, q_events


def _queue_profile(q_events, open_min, close_min):
    """Turn (+1/-1) events into a per-minute queue-length array over the day."""
    minutes = np.arange(int(open_min), int(close_min) + 1)
    qlen = np.zeros(minutes.shape[0], dtype=int)
    if q_events:
        ev = sorted(q_events)
        cur = 0
        j = 0
        for i, t in enumerate(minutes):
            while j < len(ev) and ev[j][0] <= t:
                cur += ev[j][1]
                j += 1
            qlen[i] = max(cur, 0)
    return minutes, qlen


def simulate(population: pd.DataFrame, contacts: pd.DataFrame,
             config: dict | None = None, seed: int | None = None,
             levers: dict | None = None, routing_mode: str | None = None,
             focus_day: int | None = None) -> dict:
    """Run the appointment-book simulation over the full year, with Phase-4 levers.

    Returns a dict with the augmented contacts, daily series, per-role
    utilisation, a focus-day intra-day profile, and headline KPIs (access,
    continuity, safety, capacity).
    """
    if config is None:
        config = load_config()
    pr = config["practice"]
    rng = make_rng(config.get("seed", 42) if seed is None else seed, "simulation")

    levers = resolve_levers(config, levers)
    total_triage = levers.get("triage_model") == "total_triage"
    if routing_mode is None:
        routing_mode = "triage" if total_triage else "traditional"

    cal = build_calendar(config)
    n_days = len(cal)
    working = (~cal["weekend"].to_numpy()) & (~cal["is_bank_holiday"].to_numpy())
    n_working = int(working.sum())

    clinicians = build_clinicians(config)
    n_clin = len(clinicians)
    routing = routing_for(config, routing_mode)
    elig_sameday = _sameday_eligibility(routing, clinicians)
    reserve = pr["urgent_reserved_fraction"]

    m = len(contacts)
    c_type = contacts["type"].to_numpy()
    c_urg = contacts["urgency"].to_numpy()
    c_acuity = contacts["acuity"].to_numpy()
    c_day = contacts["day"].to_numpy()
    c_hour = contacts["arrival_hour"].to_numpy()
    c_pos = contacts["patient_id"].to_numpy() - 1     # positional index into population

    # ---- Phase 4 pre-processing: AI admin, then total triage --------------
    status = np.array(["active"] * m, dtype=object)
    apply_ai_admin(c_type, status, levers, config, rng)
    triage_diag = {"n_triaged": 0, "diverted": 0, "under_triaged": 0,
                   "over_triaged": 0, "ai_triaged": 0}
    if total_triage:
        triage_diag = apply_total_triage(c_type, c_acuity, status, levers, config, rng)

    # triage burden falls only on contacts needing human triage (online contacts
    # self-triage via the eConsult-style form, so they cost no staff time)
    c_channel = contacts["channel"].to_numpy()
    needs_human_triage = (status == "active")
    if config["triage"].get("online_self_triage", True):
        needs_human_triage = needs_human_triage & (c_channel != "online")
    n_human_triaged_per_wd = int(needs_human_triage.sum()) / max(n_working, 1)

    # ---- effective capacity after AI gains and triage burden --------------
    eff = effective_slots(clinicians, levers, config, n_human_triaged_per_wd, total_triage)
    sameday_slots = np.array([round(eff[i] * reserve) if c.urgent_capable else 0
                              for i, c in enumerate(clinicians)])
    routine_slots = eff - sameday_slots
    clin_routine = np.zeros((n_days, n_clin), dtype=int)
    clin_routine[working] = routine_slots
    appts_used = np.zeros((n_days, n_clin), dtype=int)

    # ---- continuity setup -------------------------------------------------
    usual_per_patient = assign_usual_clinician(population, clinicians, config, rng)
    usual_cid = usual_per_patient[c_pos]
    cont_pref = population["continuity_preference"].to_numpy()[c_pos]
    cont_attempt = continuity_attempt(c_type, cont_pref, levers, config, rng)

    seen_cid = np.full(m, -1, dtype=int)
    day_seen = np.full(m, -1, dtype=int)
    wait_days = np.full(m, -1, dtype=int)
    intraday_wait = np.full(m, np.nan)
    same_day = np.zeros(m, dtype=bool)
    dna = np.zeros(m, dtype=bool)
    cont_booked = np.zeros(m, dtype=bool)

    active = status == "active"
    is_sameday = active & np.isin(c_type, list(SAMEDAY_TYPES)) & np.isin(c_urg, ["urgent", "emergency"])

    open_min = pr["open_hour"] * 60.0
    close_min = pr["close_hour"] * 60.0

    # ---- Pass A: same-day access via SimPy, day by day ---------------------
    focus_profile = None
    if focus_day is None:
        sd_per_day = np.bincount(c_day[is_sameday], minlength=n_days)
        focus_day = int(np.argmax(np.where(working, sd_per_day, 0)))

    sd_idx_by_day: dict[int, list] = {}
    for idx in np.nonzero(is_sameday)[0]:
        sd_idx_by_day.setdefault(int(c_day[idx]), []).append(idx)

    for d in range(n_days):
        rows = sd_idx_by_day.get(d)
        if not rows:
            continue
        if not working[d]:
            for idx in rows:
                status[idx] = "escalated"
            continue
        sameday_cap = {c.cid: int(sameday_slots[c.cid]) for c in clinicians
                       if sameday_slots[c.cid] > 0}
        day_rows = [(idx, c_type[idx], c_hour[idx] * 60.0) for idx in rows]
        results, q_events = _simulate_sameday_day(
            day_rows, clinicians, sameday_cap, elig_sameday, open_min, close_min)
        for idx, (st, cid, wait, _start) in results.items():
            if st == "seen":
                status[idx] = "seen"
                seen_cid[idx] = cid
                day_seen[idx] = d
                wait_days[idx] = 0
                same_day[idx] = True
                intraday_wait[idx] = wait
                appts_used[d, cid] += 1
            else:
                status[idx] = "escalated"
        if d == focus_day:
            minutes, qlen = _queue_profile(q_events, open_min, close_min)
            focus_profile = {"minutes": minutes, "qlen": qlen, "day": d}

    # ---- Pass B: routine booking (continuity-aware) -----------------------
    role_to_cids: dict[str, list[int]] = {}
    for c in clinicians:
        role_to_cids.setdefault(c.role, []).append(c.cid)
    type_pref: dict[str, list[list[int]]] = {}
    for ctype in pd.unique(c_type):
        groups = []
        for token in routing.get(ctype, []):
            for role in expand_role_token(token):
                cids = [cid for cid in role_to_cids.get(role, [])
                        if ctype in clinicians[cid].scope]
                if cids:
                    groups.append(cids)
        type_pref[ctype] = groups

    max_wait = int(pr["max_routine_wait_days"])
    dna_rate = pr["dna_rate"]
    routine_idx = np.nonzero(active & ~is_sameday)[0]
    order = routine_idx[np.lexsort((c_hour[routine_idx], c_day[routine_idx]))]
    for idx in order:
        a = int(c_day[idx])
        dmax = min(a + max_wait, n_days - 1)
        assigned = False
        booked_cont = False
        # 1) try the patient's usual clinician (continuity), accepting a longer wait
        if cont_attempt[idx]:
            u = int(usual_cid[idx])
            if c_type[idx] in clinicians[u].scope:
                for d in range(a, dmax + 1):
                    if working[d] and clin_routine[d, u] > 0:
                        cid = u
                        booked_cont = True
                        assigned = True
                        break
        # 2) otherwise book the best-matched available clinician
        if not assigned:
            groups = type_pref.get(c_type[idx], [])
            for d in range(a, dmax + 1):
                if not working[d]:
                    continue
                for cids in groups:
                    avail = [cid for cid in cids if clin_routine[d, cid] > 0]
                    if avail:
                        cid = max(avail, key=lambda x: clin_routine[d, x])
                        assigned = True
                        break
                if assigned:
                    break
        if assigned:
            clin_routine[d, cid] -= 1
            appts_used[d, cid] += 1
            status[idx] = "seen"
            seen_cid[idx] = cid
            day_seen[idx] = d
            wait_days[idx] = d - a
            same_day[idx] = (d == a)
            cont_booked[idx] = booked_cont
            dna[idx] = rng.random() < dna_rate
        else:
            status[idx] = "unmet"

    contacts_out = contacts.copy()
    contacts_out["status"] = status
    contacts_out["seen_clinician"] = [clinicians[c].name if c >= 0 else None for c in seen_cid]
    contacts_out["seen_role"] = [clinicians[c].role if c >= 0 else None for c in seen_cid]
    contacts_out["seen_cid"] = seen_cid
    contacts_out["day_seen"] = day_seen
    contacts_out["wait_days"] = wait_days
    contacts_out["same_day"] = same_day
    contacts_out["intraday_wait_min"] = np.round(intraday_wait, 2)
    contacts_out["continuity_booked"] = cont_booked
    contacts_out["dna"] = dna

    # ---- daily series & utilisation --------------------------------------
    roles = list(config["practice"]["roles"].keys())
    role_of = np.array([c.role for c in clinicians])
    daily_cap = np.where(working[:, None], eff[None, :], 0)

    used_by_role = {r: appts_used[:, role_of == r].sum(axis=1) for r in roles}
    cap_by_role = {r: daily_cap[:, role_of == r].sum(axis=1) for r in roles}
    total_used = appts_used.sum(axis=1)
    total_cap = daily_cap.sum(axis=1)

    demand_new = np.bincount(c_day, minlength=n_days)
    seen_same_day = np.bincount(c_day[same_day], minlength=n_days) if same_day.any() else np.zeros(n_days, int)
    esc_mask = status == "escalated"
    escalations = np.bincount(c_day[esc_mask], minlength=n_days) if esc_mask.any() else np.zeros(n_days, int)

    daily = pd.DataFrame({
        "day": np.arange(n_days),
        "date": cal["date"].to_numpy(),
        "weekday": cal["weekday"].to_numpy(),
        "working": working,
        "demand": demand_new,
        "appts_delivered": total_used,
        "capacity": total_cap,
        "utilisation": np.divide(total_used, total_cap, out=np.zeros(n_days), where=total_cap > 0),
        "seen_same_day": seen_same_day,
        "escalations": escalations,
    })

    wd = working
    role_util = pd.DataFrame([{
        "role": r,
        "capacity_per_day": int(cap_by_role[r][wd].mean()) if wd.any() else 0,
        "used_per_day": round(float(used_by_role[r][wd].mean()), 1) if wd.any() else 0.0,
        "utilisation": round(float(np.divide(used_by_role[r][wd].sum(),
                       cap_by_role[r][wd].sum())) if cap_by_role[r][wd].sum() > 0 else 0.0, 3),
    } for r in roles])

    safe_limit = np.array([c.safe_daily_contacts for c in clinicians])
    load = appts_used[working]
    over = load > safe_limit[None, :]
    safe_breach_rate = float(over.mean()) if over.size else 0.0
    max_load_ratio = float((load / safe_limit[None, :]).max()) if load.size else 0.0

    # ---- continuity (UPC) & safety metrics -------------------------------
    upc_overall, upc_high_need, pct_chronic_usual = _continuity_metrics(
        contacts_out, population, config)

    seen_mask = status == "seen"
    n_seen = int(seen_mask.sum())
    routine_seen = seen_mask & (~same_day) & (wait_days >= 0)
    high_acuity = np.isin(c_acuity, ["high", "emergency"])
    missed_urgent = int(((status == "under_triaged") | ((status == "unmet") & high_acuity)).sum())

    gp_roles = {"gp_partner", "salaried_gp", "locum_gp"}
    gp_used = sum(used_by_role[r][wd].sum() for r in roles if r in gp_roles)
    gp_cap = max(sum(cap_by_role[r][wd].sum() for r in roles if r in gp_roles), 1)

    kpis = {
        "triage_model": levers.get("triage_model"),
        "routing_mode": routing_mode,
        "ai_scribe_adoption": levers.get("ai_scribe_adoption", 0.0),
        "ai_triage_adoption": levers.get("ai_triage_adoption", 0.0),
        "ai_triage_accuracy": levers.get("ai_triage_accuracy"),
        "ai_admin_adoption": levers.get("ai_admin_adoption", 0.0),
        "continuity_policy_strength": levers.get("continuity_policy_strength", 0.0),
        "total_contacts": m,
        "seen": n_seen,
        "n_triaged": triage_diag["n_triaged"],
        "same_day_rate": round(float(same_day.sum() / m), 3),
        "seen_rate": round(float(n_seen / m), 3),
        "resolved_no_appt_rate": round(float(((status == "diverted") | (status == "ai_resolved")).sum() / m), 3),
        "escalation_rate": round(float(esc_mask.mean()), 3),
        "unmet_rate": round(float((status == "unmet").mean()), 3),
        "mean_routine_wait_days": round(float(wait_days[routine_seen].mean()) if routine_seen.any() else 0.0, 2),
        "median_routine_wait_days": float(np.median(wait_days[routine_seen])) if routine_seen.any() else 0.0,
        "pct_seen_within_2_days": round(float(((wait_days >= 0) & (wait_days <= 2)).sum() / m), 3),
        "mean_sameday_wait_min": round(float(np.nanmean(intraday_wait)) if same_day.any() else 0.0, 1),
        "dna_rate_realised": round(float(dna.sum() / max(n_seen, 1)), 3),
        "overall_utilisation": round(float(total_used[wd].sum() / total_cap[wd].sum()) if total_cap[wd].sum() > 0 else 0.0, 3),
        "gp_utilisation": round(float(gp_used / gp_cap), 3),
        "locum_share_of_appts": round(float(used_by_role.get("locum_gp", np.zeros(1)).sum() / max(total_used.sum(), 1)), 3),
        "safe_limit_breach_rate": round(safe_breach_rate, 3),
        "max_load_vs_safe": round(max_load_ratio, 2),
        # continuity
        "continuity_upc": round(upc_overall, 3),
        "continuity_upc_high_need": round(upc_high_need, 3),
        "pct_chronic_with_usual": round(pct_chronic_usual, 3),
        # triage / safety
        "diversion_rate": round(float((status == "diverted").sum() / m), 3),
        "ai_admin_resolved_rate": round(float((status == "ai_resolved").sum() / m), 3),
        "under_triage_count": triage_diag["under_triaged"],
        "under_triage_rate": round(float(triage_diag["under_triaged"] / m), 4),
        "over_triage_count": triage_diag["over_triaged"],
        "missed_urgent_count": missed_urgent,
        "triage_capacity_burden_slots_per_day": round(
            triage_capacity_burden(clinicians, levers, config, n_human_triaged_per_wd, total_triage), 1),
    }

    return {
        "contacts": contacts_out,
        "daily": daily,
        "role_util": role_util,
        "focus_profile": focus_profile,
        "kpis": kpis,
        "clinicians": clinicians,
        "levers": levers,
    }


def _continuity_metrics(contacts_out: pd.DataFrame, population: pd.DataFrame,
                        config: dict) -> tuple[float, float, float]:
    """Usual-Provider-of-Care index overall, for high-need patients, and the
    share of chronic-disease reviews seen by the patient's usual clinician."""
    seen = contacts_out[contacts_out["status"] == "seen"]
    if seen.empty:
        return 0.0, 0.0, 0.0
    counts = seen.groupby(["patient_id", "seen_cid"]).size()
    total = counts.groupby("patient_id").sum()
    modal = counts.groupby("patient_id").max()
    eligible = total[total >= 2].index
    upc = (modal[eligible] / total[eligible])
    upc_overall = float(upc.mean()) if len(upc) else 0.0

    thr = config["continuity"]["high_need_min_conditions"]
    high_need_ids = set(population.loc[population["n_conditions"] >= thr, "patient_id"])
    hn = upc[upc.index.isin(high_need_ids)]
    upc_high = float(hn.mean()) if len(hn) else upc_overall

    # modal clinician per patient -> % of chronic reviews with that clinician
    modal_cid = counts.groupby("patient_id").idxmax().map(lambda t: t[1])
    chronic = seen[seen["type"] == "chronic_review"].copy()
    if chronic.empty:
        return upc_overall, upc_high, 0.0
    chronic["modal"] = chronic["patient_id"].map(modal_cid)
    pct_chronic_usual = float((chronic["seen_cid"] == chronic["modal"]).mean())
    return upc_overall, upc_high, pct_chronic_usual


if __name__ == "__main__":
    from .population import generate_population
    from .demand import generate_demand
    cfg = load_config()
    pop = generate_population(cfg)
    con = generate_demand(pop, cfg)
    res = simulate(pop, con, cfg)
    for k, v in res["kpis"].items():
        print(f"{k:28} {v}")
