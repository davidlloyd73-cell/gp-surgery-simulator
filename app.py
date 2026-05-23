"""Synthetic GP Surgery Simulator -- interactive dashboard (Phase 7).

Launch with:  streamlit run app.py

ALL DATA IS SYNTHETIC. This is an illustrative sandbox, not a clinical or
commissioning decision tool.
"""
from __future__ import annotations

import copy

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from gpsim.config import load_config
from gpsim.demand import generate_demand
from gpsim.metrics import compute_metrics, frontier, intraday
from gpsim.population import generate_population
from gpsim.scenarios import apply_scenario, list_scenarios
from gpsim.simulation import simulate

st.set_page_config(page_title="Synthetic GP Surgery Simulator", layout="wide")
BASE = load_config()

ROLE_LABELS = {
    "gp_partner": "GP partners", "salaried_gp": "Salaried GPs", "locum_gp": "Locum GPs",
    "anp": "ANP / ACP", "practice_nurse": "Practice nurses", "hca": "HCAs",
    "clinical_pharmacist": "Clinical pharmacists", "fcp_physio": "FCP physios",
    "paramedic": "Paramedics", "care_navigator": "Care navigators",
    "social_prescriber": "Social prescribers", "mental_health_practitioner": "MH practitioners",
}


# ---------------------------------------------------------------------------
# config assembly + cached runs
# ---------------------------------------------------------------------------
def assemble_config(scenario, list_size, demand_mult, staff):
    cfg = copy.deepcopy(BASE)
    cfg["population"]["list_size"] = int(list_size)
    cfg["demand"]["demand_multiplier"] = float(demand_mult)
    for role, (clin, slots) in staff.items():
        cfg["practice"]["roles"][role]["clinicians"] = int(clin)
        cfg["practice"]["roles"][role]["slots_per_day"] = int(slots)
    return apply_scenario(cfg, scenario)


@st.cache_data(show_spinner=False)
def get_pop_demand(scenario, list_size, demand_mult, seed):
    cfg = assemble_config(scenario, list_size, demand_mult, {})
    pop = generate_population(cfg, seed=seed)
    con = generate_demand(pop, cfg, seed=seed)
    return pop, con


@st.cache_data(show_spinner=False)
def run_sim(scenario, list_size, demand_mult, seed, staff_items, levers_items):
    staff = {r: (c, s) for r, c, s in staff_items}
    cfg = assemble_config(scenario, list_size, demand_mult, staff)
    pop, con = get_pop_demand(scenario, list_size, demand_mult, seed)
    sim = simulate(pop, con, cfg, seed=seed, levers=dict(levers_items))
    metrics = compute_metrics(sim, pop, cfg)
    return sim, metrics


@st.cache_data(show_spinner=False)
def get_frontier(scenario, list_size, demand_mult, seed, staff_items):
    staff = {r: (c, s) for r, c, s in staff_items}
    cfg = assemble_config(scenario, list_size, demand_mult, staff)
    pop, con = get_pop_demand(scenario, list_size, demand_mult, seed)
    world = {"population": pop, "contacts": con, "config": cfg}
    presets = [
        ("Traditional, continuity off", {"triage_model": "traditional", "continuity_policy_strength": 0.0}),
        ("Traditional, continuity max", {"triage_model": "traditional", "continuity_policy_strength": 1.0}),
        ("Total triage", {"triage_model": "total_triage", "continuity_policy_strength": 0.5}),
        ("Total triage + full AI", {"triage_model": "total_triage", "continuity_policy_strength": 0.5,
                                    "ai_scribe_adoption": 1.0, "ai_admin_adoption": 1.0, "ai_triage_adoption": 1.0}),
    ]
    return frontier(world, presets, seed)


# ---------------------------------------------------------------------------
# sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.title("Controls")
st.sidebar.caption("Move the sliders, then watch the KPIs and charts update. "
                   "Everything is synthetic.")

if st.sidebar.button("Re-run (clear cache)"):
    st.cache_data.clear()

seed = st.sidebar.number_input("Random seed", value=42, step=1,
                               help="Same seed reproduces the same run exactly.")

st.sidebar.subheader("Demand")
list_size = st.sidebar.slider("Registered list size", 6000, 15000,
                              int(BASE["population"]["list_size"]), step=500)
demand_mult = st.sidebar.slider("Demand multiplier", 0.5, 2.0, 1.0, step=0.05,
                                help="Scales the whole list's contact volume.")

st.sidebar.subheader("Triage model")
triage_model = st.sidebar.radio(
    "How patients reach care", ["traditional", "total_triage"],
    format_func=lambda x: "Traditional (direct booking)" if x == "traditional"
    else "Total triage (Modern General Practice)",
    help="Total triage assesses every contact first, then routes or diverts -- "
         "better matching, but the triage step costs time.")

st.sidebar.subheader("AI adoption (0-100%)")
ai_scribe = st.sidebar.slider("AI ambient scribe", 0, 100, 0,
                              help="Cuts documentation time -> raises clinician throughput.") / 100
ai_admin = st.sidebar.slider("AI admin (letters/coding/results)", 0, 100, 0,
                             help="Absorbs admin contacts that then need no appointment.") / 100
ai_triage = st.sidebar.slider("AI triage / symptom assessment", 0, 100, 0,
                              help="Speeds up triage (total triage only).") / 100
ai_triage_acc = st.sidebar.slider("AI triage accuracy", 50, 100, 92,
                                  help="Lower accuracy -> more under-triaged urgent cases (the safety risk).") / 100

st.sidebar.subheader("Continuity")
continuity = st.sidebar.slider("Continuity-policy strength", 0.0, 1.0, 0.5, step=0.05,
                               help="How hard the practice books patients with their usual clinician. "
                                    "Higher continuity tends to cost same-day access.")

st.sidebar.subheader("Stress scenario")
scenarios = list_scenarios(BASE)
scenario = st.sidebar.selectbox("Inject a shock", list(scenarios.keys()),
                                format_func=lambda x: x.replace("_", " ").title())
st.sidebar.caption(scenarios[scenario])

with st.sidebar.expander("Staff roster (advanced)"):
    st.caption("Clinicians and daily slots per role. Scenario shocks apply on top.")
    staff = {}
    for role, spec in BASE["practice"]["roles"].items():
        c1, c2 = st.columns(2)
        clin = c1.number_input(ROLE_LABELS.get(role, role), min_value=0, max_value=20,
                               value=int(spec["clinicians"]), key=f"{role}_n")
        slots = c2.number_input("slots/day", min_value=1, max_value=60,
                                value=int(spec["slots_per_day"]), key=f"{role}_s")
        staff[role] = (clin, slots)

levers = {
    "triage_model": triage_model,
    "ai_scribe_adoption": ai_scribe,
    "ai_admin_adoption": ai_admin,
    "ai_triage_adoption": ai_triage,
    "ai_triage_accuracy": ai_triage_acc,
    "continuity_policy_strength": continuity,
}
staff_items = tuple((r, c, s) for r, (c, s) in staff.items())
levers_items = tuple(sorted(levers.items()))

# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
with st.spinner("Simulating a year of general practice..."):
    sim, M = run_sim(scenario, list_size, demand_mult, seed, staff_items, levers_items)
    fr = get_frontier(scenario, list_size, demand_mult, seed, staff_items)

# ---------------------------------------------------------------------------
# header
# ---------------------------------------------------------------------------
st.title("Synthetic GP Surgery Simulator")
st.warning("**All data is synthetic.** Every patient is invented; figures are typical "
           "published NHS/ONS-style anchors. This is an illustrative sandbox — **not** a "
           "clinical or commissioning decision tool.")

with st.expander("What this is showing  (plain English)", expanded=False):
    st.markdown(
        "This models one English GP practice (loosely North West London / Harrow) for a "
        "whole year. You set the **staff**, the **triage model**, **AI adoption** and the "
        "**continuity policy**, and the model simulates ~80,000 patient contacts flowing "
        "through the appointment book.\n\n"
        "The central trade-off is **continuity vs responsiveness**: a practice can optimise "
        "for *speed of access* (same-day) **or** for patients seeing *their own clinician* — "
        "rarely both. Push the **continuity** slider up and watch same-day access fall; that "
        "tension is the whole point. **Total triage** frees GP time by diverting work, but "
        "the triage step costs time and — with **AI triage** at lower accuracy — risks "
        "under-triaging genuinely urgent cases. The **AI scribe** raises throughput without "
        "those safety risks.")

# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------
r, c, cap, s, e = M["responsiveness"], M["continuity"], M["capacity"], M["safety"], M["efficiency"]
st.subheader("Headline KPIs")
row1 = st.columns(4)
row1[0].metric("Same-day access", f"{r['pct_same_day']*100:.0f}%")
row1[1].metric("Mean wait (routine)", f"{r['mean_routine_wait_days']:.1f} d")
row1[2].metric("GP utilisation", f"{cap['gp_utilisation']*100:.0f}%")
row1[3].metric("Continuity (UPC, high-need)", f"{c['upc_high_need']:.2f}")
row2 = st.columns(4)
row2[0].metric("Chronic reviews w/ usual GP", f"{c['pct_chronic_with_usual']*100:.0f}%")
row2[1].metric("Escalations to 111/A&E", f"{s['escalations_to_111_ae']:,}/yr")
row2[2].metric("Missed urgent (safety)", f"{s['missed_urgent_total']:,}/yr")
row2[3].metric("Cost proxy", f"£{e['cost_per_patient_year']:.0f}/patient")

# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------
col_a, col_b = st.columns(2)

# 1) demand vs capacity time series
daily = sim["daily"]
wk = daily[daily["working"]].copy()
roll = lambda x: x.rolling(15, min_periods=1, center=True).mean()
fig_ts = go.Figure()
fig_ts.add_trace(go.Scatter(x=wk["date"], y=roll(wk["demand"]), name="Demand", line=dict(color="#d62728")))
fig_ts.add_trace(go.Scatter(x=wk["date"], y=roll(wk["capacity"]), name="Capacity", line=dict(color="#2ca02c")))
fig_ts.add_trace(go.Scatter(x=wk["date"], y=roll(wk["appts_delivered"]), name="Delivered", line=dict(color="#1f77b4")))
fig_ts.update_layout(title="Demand vs capacity over the year (15-day smoothed)",
                     height=360, margin=dict(t=40, b=10), legend=dict(orientation="h", y=-0.2),
                     yaxis_title="contacts / working day")
col_a.plotly_chart(fig_ts, width="stretch")

# 2) continuity-vs-responsiveness frontier
fig_fr = go.Figure()
fig_fr.add_trace(go.Scatter(
    x=fr["responsiveness_same_day"], y=fr["continuity_chronic_usual"],
    mode="markers+text", text=fr["scenario"], textposition="top center",
    marker=dict(size=12, color="#1f77b4"), name="preset scenarios"))
fig_fr.add_trace(go.Scatter(
    x=[r["pct_same_day"]], y=[c["pct_chronic_with_usual"]],
    mode="markers", marker=dict(size=20, color="#d62728", symbol="star"),
    name="your current settings"))
fig_fr.update_layout(title="Continuity-vs-responsiveness frontier",
                     xaxis_title="responsiveness  (same-day access %)",
                     yaxis_title="continuity  (chronic reviews w/ usual GP)",
                     height=360, margin=dict(t=40, b=10), legend=dict(orientation="h", y=-0.25))
col_b.plotly_chart(fig_fr, width="stretch")

col_c, col_d = st.columns(2)

# 3) staff utilisation bars
ru = sim["role_util"].copy()
ru["label"] = ru["role"].map(ROLE_LABELS).fillna(ru["role"])
ru = ru.sort_values("utilisation", ascending=True)
colors = ["#d62728" if u >= 0.9 else "#2ca02c" if u < 0.7 else "#ff7f0e" for u in ru["utilisation"]]
fig_u = go.Figure(go.Bar(x=ru["utilisation"] * 100, y=ru["label"], orientation="h",
                         marker_color=colors, text=[f"{u*100:.0f}%" for u in ru["utilisation"]],
                         textposition="outside"))
fig_u.update_layout(title="Staff utilisation by role (red = ≥90%, stretched)",
                    xaxis_title="% of capacity used", height=380, margin=dict(t=40, b=10),
                    xaxis_range=[0, 110])
col_c.plotly_chart(fig_u, width="stretch")

# 4) intraday queue / 8am rush
idf = intraday(sim)
busy = idf[(idf["hour"] >= 7) & (idf["hour"] <= 19)]
fig_id = go.Figure()
fig_id.add_trace(go.Bar(x=busy["hour"], y=busy["arrivals"], name="arrivals", marker_color="#9ecae1"))
fig_id.add_trace(go.Scatter(x=busy["hour"], y=busy["mean_queue"], name="avg queue",
                            yaxis="y2", line=dict(color="#d62728")))
fday = sim["focus_profile"]["day"] if sim["focus_profile"] else None
fdate = str(daily.loc[fday, "date"])[:10] if fday is not None else ""
fig_id.update_layout(title=f"Intra-day same-day demand & queue (busiest day {fdate})",
                     xaxis_title="hour of day", yaxis_title="arrivals",
                     yaxis2=dict(title="avg queue", overlaying="y", side="right"),
                     height=380, margin=dict(t=40, b=10), legend=dict(orientation="h", y=-0.25))
col_d.plotly_chart(fig_id, width="stretch")

# ---------------------------------------------------------------------------
# detail + assumptions
# ---------------------------------------------------------------------------
with st.expander("All metrics (responsiveness / continuity / capacity / safety / efficiency)"):
    cols = st.columns(5)
    for col, group in zip(cols, ("responsiveness", "continuity", "capacity", "safety", "efficiency")):
        col.markdown(f"**{group.title()}**")
        for k, v in M[group].items():
            if k == "role_utilisation":
                continue
            col.caption(f"{k}: {v}")

with st.expander("Key assumptions & sources"):
    st.markdown(
        "- **Population**: age/sex from an ONS-style Harrow pyramid (default an *older* list, "
        "over-65s ~27%); ethnicity from ONS Census 2021 Harrow (large South Asian community); "
        "disease registers from **QOF 2022/23**, age/sex/ethnicity/deprivation-conditioned with "
        "correlated multimorbidity; T2DM/IHD elevated in South Asian patients.\n"
        "- **Demand**: ~5.5 contacts/patient/year (NHS Digital Appointments), winter seasonality, "
        "Monday & post-bank-holiday peaks, the 8am rush.\n"
        "- **Supply**: a plausible roster for a ~15k list with scope-of-practice; same-day access is "
        "a SimPy queue, routine demand is booked forward; DNAs ~4%; optional BMA safe-working cap.\n"
        "- **Triage/AI/continuity**: total triage diverts low-acuity work but costs triager time "
        "(online eConsults self-triage free); AI scribe/admin raise effective capacity; AI triage "
        "accuracy drives under-triage risk; continuity routes patients to their usual GP at the cost "
        "of speed.\n"
        "- **Everything is in `config/config.yaml`** with `# source:` / `# ASSUMPTION:` comments.\n\n"
        "**Limits:** an illustrative model of *system dynamics and trade-offs*, not a prediction of "
        "any real practice. Simplifications are documented in the README and code.")

st.caption("Synthetic GP Surgery Simulator · all data synthetic · illustrative only · "
           "tune anything in config/config.yaml")
