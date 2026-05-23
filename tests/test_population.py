"""Smoke + sanity tests for the Phase 1 population generator."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from gpsim.config import load_config  # noqa: E402
from gpsim.population import (  # noqa: E402
    LTC_COLUMNS, generate_calibration_reference, generate_population,
)


def test_generates_requested_size():
    cfg = load_config()
    cfg["population"]["list_size"] = 4000
    df = generate_population(cfg, seed=1)
    assert len(df) == 4000


def test_reproducible_with_seed():
    cfg = load_config()
    a = generate_population(cfg, seed=7)
    b = generate_population(cfg, seed=7)
    assert a.equals(b)
    c = generate_population(cfg, seed=8)
    assert not a.equals(c)


def test_reference_prevalences_match_qof_targets():
    """The reference population must reproduce the QOF crude targets."""
    cfg = load_config()
    ref = generate_calibration_reference(cfg, seed=42)
    abs_tol = cfg["validation"]["prevalence_abs_tol"]
    rel_tol = cfg["validation"]["prevalence_rel_tol"]
    for cond in LTC_COLUMNS:
        tgt = cfg["population"]["conditions"][cond]["target_prevalence"]
        act = ref[cond].mean()
        assert (abs(act - tgt) <= abs_tol) or (abs(act - tgt) / tgt <= rel_tol), (
            f"{cond}: target {tgt}, reference {act}")


def test_older_list_raises_burden():
    """Ageing the list must INCREASE crude prevalence & frailty (not dilute it)."""
    cfg = load_config()
    young = {**cfg["population"], "age_sex_distribution":
             cfg["population"]["calibration_reference"]["age_sex_distribution"]}
    cfg_young = {**cfg, "population": young}
    df_young = generate_population(cfg_young, seed=42)
    df_actual = generate_population(cfg, seed=42)  # the configured (older) list
    assert df_actual["age"].mean() > df_young["age"].mean()
    assert df_actual["hypertension"].mean() > df_young["hypertension"].mean()
    assert df_actual[df_actual.age >= 65]["frail"].mean() >= \
        df_young[df_young.age >= 65]["frail"].mean()


def test_south_asian_diabetes_elevated():
    cfg = load_config()
    df = generate_population(cfg, seed=42)
    sa = df[df["south_asian"]]["diabetes"].mean()
    rest = df[~df["south_asian"]]["diabetes"].mean()
    assert sa / rest >= cfg["validation"]["south_asian_diabetes_min_ratio"]


def test_behavioural_attributes_sane():
    cfg = load_config()
    df = generate_population(cfg, seed=42)
    assert abs(df["consultation_propensity"].mean() - 1.0) < 0.05
    assert df["continuity_preference"].between(0, 1).all()
    # high-need patients value continuity more than the young & healthy
    hi = df[df["n_conditions"] >= 3]["continuity_preference"].mean()
    lo = df[(df["age"] < 40) & (df["n_conditions"] == 0)]["continuity_preference"].mean()
    assert hi > lo
