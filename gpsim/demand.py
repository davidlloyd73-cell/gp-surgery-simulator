"""Phase 2 -- demand model.

Turns a synthetic registered list into a year-long stream of contacts (demand
events) at daily resolution, with an intra-day arrival time so a single busy day
can be inspected too.

Each contact carries:
  * when:    day-of-year, calendar date, month, weekday, arrival_hour
  * who:     patient_id (+ the patient's attributes are available via the list)
  * what:    type     (acute_minor / acute_serious / chronic_review /
                        mental_health / admin / self_care)
             channel  (phone / online / walk_in)
             urgency  (emergency / urgent / routine)  -- what is *requested*
             acuity   (low / moderate / high / emergency) -- the ground truth

Demand volume per patient is Poisson(base_rate * consultation_propensity), so
the whole-list mean lands on the configured contacts-per-patient-per-year.
Seasonality, the Monday peak, post-bank-holiday rebound and the 8am rush are all
applied through day/hour weighting. Everything is config-driven and reproducible.
"""
from __future__ import annotations

import calendar
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import load_config, make_rng

CONTACT_TYPES = ["acute_minor", "acute_serious", "chronic_review",
                 "mental_health", "admin", "self_care"]
CHANNELS = ["phone", "online", "walk_in"]
URGENCIES = ["emergency", "urgent", "routine"]
ACUITIES = ["low", "moderate", "high", "emergency"]


def _sample_categorical(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vectorised per-row categorical draw. weights: (M, K) non-negative.

    Returns an (M,) array of column indices, each row sampled from its own
    (normalised) weight distribution.
    """
    cdf = np.cumsum(weights, axis=1)
    cdf = cdf / cdf[:, -1:]
    u = rng.random(weights.shape[0])
    return (u[:, None] >= cdf).sum(axis=1).clip(0, weights.shape[1] - 1)


def build_calendar(config: dict) -> pd.DataFrame:
    """One row per day of the sim year with its demand weight and flags."""
    d = config["demand"]
    year = d["sim_year"]
    n_days = 366 if calendar.isleap(year) else 365
    dates = [date(year, 1, 1) + timedelta(days=i) for i in range(n_days)]
    months = np.array([dt.month for dt in dates])
    weekdays = np.array([dt.weekday() for dt in dates])   # 0 = Monday

    month_mult = np.array(d["month_multipliers"])[months - 1]
    weekday_mult = np.array(d["weekday_multipliers"])[weekdays]

    holidays = {date(year, m, day) for m, day in d["bank_holidays"]}
    is_holiday = np.array([dt in holidays for dt in dates])
    weekend = weekdays >= 5

    weight = month_mult * weekday_mult
    weight = np.where(is_holiday, weight * d["bank_holiday_multiplier"], weight)

    # Post-holiday rebound: surge the next working (non-weekend, non-holiday) day.
    post_holiday = np.zeros(n_days, dtype=bool)
    for i, dt in enumerate(dates):
        if dt in holidays:
            j = i + 1
            while j < n_days and (weekdays[j] >= 5 or dates[j] in holidays):
                j += 1
            if j < n_days:
                post_holiday[j] = True
    weight = np.where(post_holiday, weight * d["post_holiday_surge"], weight)

    return pd.DataFrame({
        "day": np.arange(n_days),
        "date": pd.to_datetime(dates),
        "month": months,
        "weekday": weekdays,
        "weekend": weekend,
        "is_bank_holiday": is_holiday,
        "post_holiday": post_holiday,
        "weight": weight,
    })


def _patient_type_weights(pop: pd.DataFrame, cfg_types: dict) -> np.ndarray:
    """(N, 6) contact-type weights per patient."""
    n = len(pop)
    w = np.zeros((n, len(CONTACT_TYPES)))
    for k, t in enumerate(CONTACT_TYPES):
        w[:, k] = cfg_types[t]["base"]
    ci = CONTACT_TYPES.index("chronic_review")
    w[:, ci] += cfg_types["chronic_review"].get("per_condition", 0.0) * pop["n_conditions"].to_numpy()
    mi = CONTACT_TYPES.index("mental_health")
    mh = (pop["depression"].to_numpy() | pop["smi"].to_numpy())
    w[:, mi] += np.where(mh, cfg_types["mental_health"].get("depression_smi_boost", 0.0), 0.0)
    return w


def _patient_channel_weights(pop: pd.DataFrame, cfg_ch: dict) -> np.ndarray:
    """(N, 3) channel weights per patient, by age group."""
    n = len(pop)
    age = pop["age"].to_numpy()
    groups = np.where(age < 40, 0, np.where(age < 65, 1, 2))
    keys = ["under_40", "age_40_64", "age_65_plus"]
    table = np.array([[cfg_ch[k][c] for c in CHANNELS] for k in keys])  # (3,3)
    return table[groups]


def generate_demand(population: pd.DataFrame, config: dict | None = None,
                    seed: int | None = None) -> pd.DataFrame:
    """Generate the year-long contact stream for a given population."""
    if config is None:
        config = load_config()
    d = config["demand"]
    rng = make_rng(config.get("seed", 42) if seed is None else seed, "demand")

    cal = build_calendar(config)
    n_days = len(cal)
    day_p = cal["weight"].to_numpy()
    day_p = day_p / day_p.sum()

    n = len(population)
    base_rate = d["contacts_per_patient_per_year"] * d.get("demand_multiplier", 1.0)
    annual_rate = base_rate * population["consultation_propensity"].to_numpy()
    n_contacts = rng.poisson(annual_rate)
    pidx = np.repeat(np.arange(n), n_contacts)   # positional patient index
    m = pidx.shape[0]
    if m == 0:
        return pd.DataFrame()

    # when
    day_idx = rng.choice(n_days, size=m, p=day_p)
    hour_p = np.array(d["hour_weights"], dtype=float)
    hour_p = hour_p / hour_p.sum()
    hour = rng.choice(24, size=m, p=hour_p) + rng.random(m)

    # what -- type (per patient), then urgency & acuity (per type)
    type_w = _patient_type_weights(population, d["contact_types"])[pidx]
    type_idx = _sample_categorical(type_w, rng)

    chan_w = _patient_channel_weights(population, d["channels_by_age"])[pidx]
    chan_idx = _sample_categorical(chan_w, rng)

    urg_table = np.array([[d["urgency_by_type"][t][u] for u in URGENCIES]
                          for t in CONTACT_TYPES])
    urg_idx = _sample_categorical(urg_table[type_idx], rng)

    acu_table = np.array([[d["acuity_by_type"][t][a] for a in ACUITIES]
                          for t in CONTACT_TYPES])
    acu_idx = _sample_categorical(acu_table[type_idx], rng)

    cal_day = cal.iloc[day_idx].reset_index(drop=True)
    contacts = pd.DataFrame({
        "contact_id": np.arange(1, m + 1),
        "patient_id": population["patient_id"].to_numpy()[pidx],
        "day": day_idx,
        "date": cal_day["date"].to_numpy(),
        "month": cal_day["month"].to_numpy(),
        "weekday": cal_day["weekday"].to_numpy(),
        "weekend": cal_day["weekend"].to_numpy(),
        "is_bank_holiday": cal_day["is_bank_holiday"].to_numpy(),
        "post_holiday": cal_day["post_holiday"].to_numpy(),
        "arrival_hour": np.round(hour, 3),
        "type": np.array(CONTACT_TYPES)[type_idx],
        "channel": np.array(CHANNELS)[chan_idx],
        "urgency": np.array(URGENCIES)[urg_idx],
        "acuity": np.array(ACUITIES)[acu_idx],
    })
    return contacts.sort_values(["day", "arrival_hour"]).reset_index(drop=True)


def daily_demand(contacts: pd.DataFrame, n_days: int) -> np.ndarray:
    """Contacts per day across the year (length n_days)."""
    counts = np.zeros(n_days, dtype=int)
    vc = contacts["day"].value_counts()
    counts[vc.index.to_numpy()] = vc.to_numpy()
    return counts


def intraday_profile(contacts: pd.DataFrame, day: int) -> np.ndarray:
    """Contacts per hour (length 24) for a single day."""
    sub = contacts[contacts["day"] == day]
    counts = np.zeros(24, dtype=int)
    hrs, c = np.unique(sub["arrival_hour"].astype(int), return_counts=True)
    counts[hrs] = c
    return counts


if __name__ == "__main__":
    from .population import generate_population
    cfg = load_config()
    pop = generate_population(cfg)
    con = generate_demand(pop, cfg)
    print(con.head())
    print(f"\n{len(con):,} contacts  ({len(con)/len(pop):.2f} per patient/yr)")
