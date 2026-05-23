# Synthetic GP Surgery Simulator

An interactive sandbox for exploring how an English GP practice behaves as you
change its **staff mix**, **triage model**, **use of AI**, and **demand
pressures**. Loosely modelled on a representative North West London / Harrow
practice ("Ridgway Surgery, North Harrow").

The central thing it makes visible is the **continuity-versus-responsiveness
tension**: a practice can optimise for *speed of access* OR for patients seeing
*their own clinician*, and this model lets you see and tune that trade-off.

**▶ Live demo: <https://gp-surgery-simulator.streamlit.app/>** — open it in a
browser, no install needed. (To run it locally instead, see *Quick start* below.)

> ## ⚠️ All data is synthetic
> Every patient in this model is invented. No real patient data and no real
> practice records are used. The figures are typical published NHS/ONS-style
> anchors. **This is an illustrative sandbox, not a clinical or commissioning
> decision tool.**

---

## Quick start

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py     # <- the interactive dashboard (one command)
```

Then move the sidebar sliders and watch the KPIs and charts update.

> **`streamlit: command not found`?** Use the `python -m streamlit ...` form
> above rather than a bare `streamlit run`. `pip` installs the Streamlit
> launcher into a per-user scripts directory (e.g. `~/Library/Python/3.x/bin`
> on macOS) that is often not on your `PATH`; invoking it via `python -m`
> sidesteps that entirely.
>
> **On a Mac, `python` / `pip` may not exist** (`zsh: command not found:
> python`) — modern macOS ships only `python3`. Use `python3` and `pip3`
> everywhere below (Python 3.10+ recommended):
>
> ```bash
> python3 -m pip install -r requirements.txt
> python3 -m streamlit run app.py
> ```

## Status — complete

Built phase by phase:

- [x] **Phase 1 — Synthetic patient list generator** (with validation)
- [x] **Phase 2 — Demand model** (year-long contact stream)
- [x] **Phase 3 — Practice / supply model** (SimPy appointment book)
- [x] **Phase 4 — Triage, flow, continuity & AI** (the heart of it)
- [x] **Phase 5 — Metrics & outputs** (+ scenario / frontier machinery)
- [x] **Phase 6 — Stress scenarios**
- [x] **Phase 7 — Streamlit dashboard**

### Command-line views (each phase also runs standalone)

```bash
python scripts/validate_population.py   # Phase 1: list profile + validation
python scripts/show_demand.py           # Phase 2: a year of demand
python scripts/show_practice.py         # Phase 3: appointment-book profile
python scripts/show_levers.py           # Phase 4: triage / AI / continuity levers
python scripts/show_metrics.py          # Phase 5: grouped metrics + frontier
python scripts/show_scenarios.py        # Phase 6: stress scenarios before/after
python -m pytest -q                     # 41 tests across all phases
```

## What Phase 1 produces

`gpsim.population.generate_population()` returns a pandas DataFrame, one row per
synthetic patient, with:

- **Demographics** calibrated to Harrow: age/sex from an ONS-style pyramid,
  ethnicity reflecting the borough's large Indian / South Asian community, and
  an IMD deprivation decile.
- **Disease registers** sampled from QOF prevalence but **age-, sex-, ethnicity-
  and deprivation-conditioned** — not flat. Type 2 diabetes and IHD are elevated
  in South Asian patients, consistent with the literature.
- **Correlated multimorbidity**: dependent conditions (CKD, IHD, heart failure,
  AF, stroke, dementia) carry comorbidity multipliers, so co-occurrence is
  realistic rather than independent.
- A **frailty** index/flag for older patients (eFI-style).
- **Behavioural attributes** for the demand model: a per-patient
  `consultation_propensity` and a `continuity_preference`.

Disease *per-stratum* rates are calibrated (by bisection) against a fixed
**reference population** (the true Harrow age pyramid) so the reference
reproduces the QOF crude targets; those fixed rates are then applied to whatever
list you configure. This means crude prevalence correctly **floats with
case-mix** — the default older list shows hypertension ~21% (1.5× national),
diabetes ~11%, CKD ~7% and ~15% of over-65s frail, rather than every list being
forced to the same national crude rate.

## What Phase 2 produces

`gpsim.demand.generate_demand(population, config)` returns a DataFrame with one
row per **contact** over a full simulated year (~82k contacts for the default
list). Volume per patient is `Poisson(base_rate × consultation_propensity)`, so
the list mean lands on the configured ~5.5 contacts/patient/year. Each contact
has:

- **When** — day, calendar date, month, weekday and an `arrival_hour`. Layered
  weighting reproduces **winter seasonality** (~1.3× summer), the **Monday
  peak** (~1.3× midweek), the **post-bank-holiday rebound** (the busiest day of
  the year is the Monday after Christmas), and the **8am rush** (~20% of a day's
  contacts in one hour).
- **What** — `type` (acute minor / acute serious / chronic review / mental
  health / admin / self-care), `channel` (phone / online / walk-in, age-tilted),
  `urgency` (what's *requested*) and a true `acuity` (the ground truth). The gap
  between urgency and acuity is what makes **under-triage** measurable later.

## What Phase 3 produces

`gpsim.simulation.simulate(population, contacts, config)` runs the appointment
book over the year and returns each contact's outcome plus daily series, per-role
utilisation and headline KPIs. Two coupled mechanisms:

- **Same-day access** (urgent/emergency clinical demand) is a **SimPy** queue —
  clinicians are resources, contacts arrive through the day, pick the least-loaded
  eligible clinician, and are seen or *escalate* (a 111/A&E proxy) if reserved
  capacity runs out or the clinic closes.
- **Routine demand** is **booked forward** against each clinician's remaining
  daily capacity, so the wait-in-days grows under pressure; requests waiting past
  the patience limit count as unmet.

Both honour **scope-of-practice** and a **routing preference**. The default
roster is deliberately *stretched* (GPs ~97% utilised). The traditional
direct-booking routing concentrates work on GPs (FCP/pharmacist/social-prescriber
near-idle); switching to triage routing diverts work to those roles and drops GP
utilisation from ~98% to ~44% — the lever Phase 4 builds on. Also models DNAs
(~4%), urgent-slot reservation, and an optional BMA safe-working cap.

**Honest about limits:** the appointment book is simplified — no part-time rota
detail or within-day session boundaries beyond open/close, and routine
appointments are assumed to flow smoothly once booked. SimPy drives the same-day
queue (where queueing physics matter); routine is a capacity ledger. It exists to
show *dynamics and trade-offs*, not to reproduce a real practice's diary.

## What Phase 4 produces (the heart of it)

`simulate(...)` takes a `levers` dict and exposes the interesting choices:

- **Triage model** — `traditional` direct-booking vs `total_triage`. Total triage
  diverts ~26% of demand (signposting/pharmacy/self-care) and drops GP utilisation
  from ~97% to ~46%, **but the triage step itself consumes capacity** (online
  eConsult contacts self-triage for free; phone/walk-in cost staff time).
- **Three AI sliders (0–100%)** — **ambient scribe** raises effective clinician
  throughput (GP capacity 146→166 slots/day at full); **AI admin** absorbs admin
  contacts (~9% resolved with no appointment at full); **AI triage** makes triage
  fast/cheap (burden 81→12 slots) but carries a tunable **accuracy** knob — drop it
  and under-triaged urgent cases climb (80 → 1,600+), so you can explore the *risk*
  side, not just the upside.
- **Continuity** — each patient has a usual clinician; a `continuity_policy_strength`
  lever governs how hard the practice books patients with them. A UPC (Usual
  Provider of Care) index is tracked overall and for high-need patients.

**The designed-in tension:** pushing continuity trades away responsiveness. As the
continuity lever goes 0→1, chronic reviews with the usual clinician rise 20%→60%
and high-need UPC rises 0.26→0.37, while same-day access falls 0.91→0.71. The
model exposes this **continuity-vs-responsiveness frontier** rather than hiding it.

## What Phase 5 produces

`gpsim.metrics` consolidates everything into the brief's five metric families —
**responsiveness** (time-to-contact, % same-day, intraday queue, did-not-wait,
unmet), **continuity** (UPC overall + high-need, % chronic reviews with usual
clinician), **capacity** (utilisation by role, locum reliance, contacts vs safe
limit), **safety** (escalations to 111/A&E, under-triaged urgent, appropriate-
triage rate, % high-acuity seen) and **efficiency** (clinician hours freed by AI,
a £/patient cost proxy) — plus the scenario machinery the dashboard runs on:

- `build_world()` generates population + demand once (cached across lever changes);
- `run_scenario()` / `run_full()` run the book for a lever set and return metrics;
- `frontier()` returns continuity-vs-responsiveness points for the frontier plot;
- `sweep_lever()` varies one lever; `intraday()` gives the per-hour 8am-rush view.

The frontier makes the trade-offs explicit — e.g. total triage is cheaper
(~£153 vs £193/patient) and frees GPs, but carries a real under-triage safety
cost that worsens as AI-triage accuracy falls.

## What Phase 6 produces

`gpsim.scenarios` adds named, toggleable shocks applied on top of the baseline
config (the baseline is never mutated): **winter surge** (demand spike + staff
sickness), **pandemic spike** (sharp demand + shift to remote channels), **GP
vacancy**, **staff shortage**, **Monday/post-bank-holiday pile-up**, and **list
growth + demand inflation**. `compare_scenarios()` shows before/after on the same
KPIs. For example, a winter surge collapses same-day access 85%→42% with 10-day
waits — and the model shows AI scribe + admin can claw most of that back
(→64%, ~2-day waits), while relying on AI triage to do it trades safety.

## What Phase 7 produces — the dashboard

`streamlit run app.py` launches the interactive sandbox for a non-developer:

- **Sidebar controls** — list size & demand multiplier; triage-model toggle;
  three AI sliders + an AI-triage-accuracy slider; continuity-policy strength;
  a stress-scenario selector; the full staff roster (clinicians & slots per role,
  in an expander); and a random seed + "re-run" button.
- **Main panel** — eight KPI cards (responsiveness, continuity, capacity, safety,
  cost) followed by four interactive Plotly charts: demand-vs-capacity over the
  year, the **continuity-vs-responsiveness frontier** (preset scenarios plus your
  current settings as a star), staff utilisation by role, and the intra-day 8am
  queue. Plus plain-English "what this is showing" and "key assumptions" panels.

Population + demand are cached so moving the AI/continuity/triage sliders re-runs
only the appointment book — fast enough to feel live. A prominent banner states
that all data is synthetic and the model is illustrative.

## Configuration & sources

Everything is in [`config/config.yaml`](config/config.yaml). Every non-obvious
number carries a `# source:` (QOF, ONS Census 2021, MHCLG IMD 2019, NHS Digital
appointments data, GP Patient Survey) or a `# ASSUMPTION:` comment, so the model
stays auditable. Change any value and re-run.

Reproducible: set `seed` in the config (or pass `seed=` to the generator) — the
same seed reproduces the same list exactly.

## Honest about limits

- QOF registers have condition-specific age denominators (e.g. depression/CKD
  are 18+). Here they are treated as whole-list rates with the age gradient
  doing the work — fine for an illustrative model, not for epidemiology.
- Multimorbidity uses the 15 QOF registers only; broader condition lists (40+
  conditions, as in Barnett et al. 2012) yield higher 2+ rates. On the default
  (older) list ~24% of adults have 2+ LTCs, rising steeply with age — the age
  gradient is the point.
- The per-stratum calibration standardises against *age* (via the reference
  pyramid) but shares the ethnicity/IMD mix; QOF England targets are treated as
  the reference list's crude rates. An anchor, not epidemiology.
- The model simplifies clinical reality throughout; it is for exploring
  *system dynamics and trade-offs*, not predicting any real practice.
