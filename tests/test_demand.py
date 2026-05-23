"""Smoke + sanity tests for the Phase 2 demand model."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from gpsim.config import load_config  # noqa: E402
from gpsim.demand import (  # noqa: E402
    CONTACT_TYPES, build_calendar, daily_demand, generate_demand,
)
from gpsim.population import generate_population  # noqa: E402


def _setup(list_size=6000):
    cfg = load_config()
    cfg["population"]["list_size"] = list_size
    pop = generate_population(cfg, seed=42)
    con = generate_demand(pop, cfg, seed=42)
    return cfg, pop, con


def test_volume_matches_target_rate():
    cfg, pop, con = _setup()
    rate = len(con) / len(pop)
    target = cfg["demand"]["contacts_per_patient_per_year"]
    assert abs(rate - target) / target < 0.1   # within 10%


def test_reproducible_with_seed():
    cfg = load_config()
    cfg["population"]["list_size"] = 5000
    pop = generate_population(cfg, seed=3)
    a = generate_demand(pop, cfg, seed=3)
    b = generate_demand(pop, cfg, seed=3)
    assert a.equals(b)


def test_seasonality_winter_higher_than_summer():
    _, _, con = _setup()
    by_month = con.groupby("month").size()
    winter = by_month[[12, 1, 2]].mean()
    summer = by_month[[6, 7, 8]].mean()
    assert winter > summer


def test_monday_busier_than_midweek():
    cfg, _, con = _setup()
    dd = daily_demand(con, len(build_calendar(cfg)))
    wk = build_calendar(cfg)["weekday"].to_numpy()
    assert dd[wk == 0].mean() > dd[wk == 2].mean()    # Mon > Wed


def test_eight_am_is_modal_hour():
    _, _, con = _setup()
    hour_counts = con["arrival_hour"].astype(int).value_counts()
    assert hour_counts.idxmax() == 8


def test_categories_valid_and_complete():
    _, _, con = _setup()
    assert set(con["type"]).issubset(set(CONTACT_TYPES))
    assert con["channel"].isin(["phone", "online", "walk_in"]).all()
    assert con["urgency"].isin(["emergency", "urgent", "routine"]).all()
    assert con["acuity"].isin(["low", "moderate", "high", "emergency"]).all()


def test_multimorbid_get_more_chronic_reviews():
    _, pop, con = _setup()
    merged = con.merge(pop[["patient_id", "n_conditions"]], on="patient_id")
    chronic = merged[merged["type"] == "chronic_review"]
    # chronic reviews concentrate in patients with conditions
    assert chronic["n_conditions"].mean() > merged["n_conditions"].mean()


def test_older_patients_use_less_online():
    _, pop, con = _setup()
    merged = con.merge(pop[["patient_id", "age"]], on="patient_id")
    young_online = (merged[merged.age < 40]["channel"] == "online").mean()
    old_online = (merged[merged.age >= 65]["channel"] == "online").mean()
    assert young_online > old_online
