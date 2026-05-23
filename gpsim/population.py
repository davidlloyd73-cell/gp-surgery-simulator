"""Phase 1 -- synthetic registered-list generator.

Produces a plausible registered patient list for one English general practice,
calibrated to North West London / Harrow demographics. Output is a pandas
DataFrame, one row per synthetic patient.

Key modelling decision -- per-stratum rates, not fixed crude rates
-----------------------------------------------------------------
QOF target prevalences are England *crude* (whole-list) figures. If we forced
every list to reproduce that crude figure, an OLDER list would dilute each
elderly patient's risk (counter-factually lowering multimorbidity and frailty).
Instead we calibrate disease *per-stratum* rates against a fixed REFERENCE
population (the baseline Harrow pyramid) so the reference reproduces the QOF
targets, then APPLY those fixed rates to whatever list you configure. An older
list then correctly shows higher crude prevalence, multimorbidity and frailty;
a younger list, lower.

Other notes
-----------
* Multimorbidity is correlated, not independent: "dependent" conditions (CKD,
  IHD, heart failure, AF, stroke, dementia) carry comorbidity multipliers
  referencing conditions sampled earlier, so co-occurrence looks realistic.
* Everything is driven by config; no clinical constants live in this file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_config, parse_age_band

# The chronic conditions that count toward a patient's multimorbidity tally.
# (Frailty is derived from these, not counted as one of them.)
LTC_COLUMNS = [
    "hypertension", "diabetes", "asthma", "copd", "depression", "smi",
    "cancer", "osteoporosis", "learning_disability", "ihd", "heart_failure",
    "atrial_fibrillation", "stroke_tia", "ckd", "dementia",
]


# ---------------------------------------------------------------------------
# calibration helper
# ---------------------------------------------------------------------------
def _calibrate_scale(relative_risk: np.ndarray, target_prevalence: float,
                     iters: int = 60) -> float:
    """Find scale s so that mean(clip(s * relative_risk, 0, 1)) == target.

    The clipped mean is monotonic increasing in s, so bisection nails the target
    prevalence (up to numerical tolerance) regardless of how the relative risks
    are shaped.
    """
    if target_prevalence <= 0:
        return 0.0
    rr = relative_risk
    if not np.any(rr > 0):
        return 0.0
    lo, hi = 0.0, 1.0
    while np.mean(np.clip(hi * rr, 0.0, 1.0)) < target_prevalence and hi < 1e9:
        hi *= 2.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if np.mean(np.clip(mid * rr, 0.0, 1.0)) < target_prevalence:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _ethnicity_multiplier(cond_cfg: dict, south_asian: np.ndarray,
                          black: np.ndarray) -> np.ndarray:
    n = south_asian.shape[0]
    em = cond_cfg.get("ethnicity_multipliers")
    if not em:
        return np.ones(n)
    mult = np.full(n, em.get("default", 1.0), dtype=float)
    mult[south_asian] = em.get("south_asian", em.get("default", 1.0))
    mult[black] = em.get("black", em.get("default", 1.0))
    return mult


def _deprivation_multiplier(cond_cfg: dict, imd_decile: np.ndarray) -> np.ndarray:
    rr_ratio = cond_cfg.get("deprivation_rr")
    if not rr_ratio or rr_ratio == 1.0:
        return np.ones(imd_decile.shape[0])
    frac = (10 - imd_decile) / 9.0          # decile 1 -> rr_ratio, decile 10 -> 1.0
    return np.power(rr_ratio, frac)


# ---------------------------------------------------------------------------
# demographics
# ---------------------------------------------------------------------------
def _draw_demographics(n: int, pop: dict, age_sex_dist: dict,
                       rng: np.random.Generator) -> dict:
    """Sample age/sex/ethnicity/IMD for `n` patients using the given age pyramid.

    Ethnicity and IMD distributions are shared (taken from `pop`); only the age
    pyramid varies between the reference and the actual list.
    """
    bands = list(pop["age_bands"])
    band_ranges = {b: parse_age_band(b) for b in bands}

    shares = np.array([age_sex_dist[b]["share"] for b in bands], dtype=float)
    shares = shares / shares.sum()
    band_idx = rng.choice(len(bands), size=n, p=shares)

    ages = np.empty(n, dtype=int)
    for i, b in enumerate(bands):
        lo, hi = band_ranges[b]
        mask = band_idx == i
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        if b.endswith("+"):
            draws = lo + rng.geometric(p=0.18, size=cnt) - 1   # decaying tail
            ages[mask] = np.minimum(draws, 105)
        else:
            ages[mask] = rng.integers(lo, hi + 1, size=cnt)

    male_frac = np.array([age_sex_dist[b]["male_frac"] for b in bands])[band_idx]
    sex = np.where(rng.random(n) < male_frac, "M", "F")

    eth_dist = pop["ethnicity_distribution"]
    eth_names = list(eth_dist.keys())
    eth_p = np.array([eth_dist[k] for k in eth_names], dtype=float)
    eth_p = eth_p / eth_p.sum()
    ethnicity = np.array(eth_names)[rng.choice(len(eth_names), size=n, p=eth_p)]
    south_asian = np.isin(ethnicity, list(pop["south_asian_groups"]))
    black = np.isin(ethnicity, list(pop["black_groups"]))

    imd_p = np.array(pop["imd_decile_distribution"], dtype=float)
    imd_p = imd_p / imd_p.sum()
    imd_decile = rng.choice(np.arange(1, 11), size=n, p=imd_p)

    return {
        "bands": bands, "band_idx": band_idx,
        "band_labels": np.array(bands)[band_idx],
        "ages": ages, "sex": sex, "ethnicity": ethnicity,
        "south_asian": south_asian, "black": black, "imd_decile": imd_decile,
    }


def _sample_registers(demo: dict, pop: dict, rng: np.random.Generator,
                      scales: dict | None = None) -> tuple[dict, dict]:
    """Sample disease registers.

    If `scales` is None, calibrate a scale per condition to its QOF target and
    return the scales (reference mode). Otherwise apply the supplied scales so
    realised prevalence floats with this list's case-mix (actual-list mode).
    """
    n = demo["ages"].shape[0]
    bands = demo["bands"]
    band_idx = demo["band_idx"]
    conditions_cfg = pop["conditions"]
    calibrate = scales is None
    out_scales: dict = {} if calibrate else scales
    flags: dict[str, np.ndarray] = {}

    for cond in pop["condition_sampling_order"]:
        c = conditions_cfg[cond]
        rr = np.ones(n, dtype=float)
        rr *= np.array([c["age_multipliers"][b] for b in bands])[band_idx]
        sm = c.get("sex_multipliers")
        if sm:
            rr *= np.where(demo["sex"] == "M", sm["male"], sm["female"])
        rr *= _ethnicity_multiplier(c, demo["south_asian"], demo["black"])
        rr *= _deprivation_multiplier(c, demo["imd_decile"])
        for other, m in c.get("comorbidity_multipliers", {}).items():
            rr = rr * np.where(flags[other], m, 1.0)

        if calibrate:
            scale = _calibrate_scale(rr, float(c["target_prevalence"]))
            out_scales[cond] = scale
        else:
            scale = out_scales[cond]
        prob = np.clip(scale * rr, 0.0, 1.0)
        flags[cond] = rng.random(n) < prob

    return flags, out_scales


def _assemble_df(demo: dict, flags: dict) -> pd.DataFrame:
    n = demo["ages"].shape[0]
    df = pd.DataFrame({
        "patient_id": np.arange(1, n + 1),
        "age": demo["ages"],
        "age_band": demo["band_labels"],
        "sex": demo["sex"],
        "ethnicity": demo["ethnicity"],
        "south_asian": demo["south_asian"],
        "imd_decile": demo["imd_decile"],
    })
    for cond in LTC_COLUMNS:
        df[cond] = flags[cond]
    df["n_conditions"] = df[LTC_COLUMNS].sum(axis=1).astype(int)
    return df


def _add_frailty_and_behaviour(df: pd.DataFrame, demo: dict, pop: dict,
                               rng: np.random.Generator) -> None:
    n = len(df)
    bands = demo["bands"]
    band_idx = demo["band_idx"]

    fr = pop["frailty"]
    min_age = fr["min_age"]
    over_min = np.maximum(0, df["age"].to_numpy() - min_age)
    efi = (fr["base_index"]
           + fr["index_per_condition"] * df["n_conditions"].to_numpy()
           + fr["index_per_year_over_min"] * over_min)
    efi = np.clip(efi, 0.0, 1.0)
    is_old = df["age"].to_numpy() >= min_age
    efi = np.where(is_old, efi, 0.0)
    cats = fr["categories"]
    frailty_cat = np.full(n, "n/a", dtype=object)
    frailty_cat[is_old & (efi <= cats["fit"])] = "fit"
    frailty_cat[is_old & (efi > cats["fit"]) & (efi <= cats["mild"])] = "mild"
    frailty_cat[is_old & (efi > cats["mild"]) & (efi <= cats["moderate"])] = "moderate"
    frailty_cat[is_old & (efi > cats["moderate"])] = "severe"
    levels = {"fit": 0, "mild": 1, "moderate": 2, "severe": 3}
    threshold = levels[fr["frail_flag_from"]]
    df["frailty_index"] = np.round(efi, 3)
    df["frailty_category"] = frailty_cat
    df["frail"] = np.array([levels.get(c, -1) >= threshold for c in frailty_cat])

    beh = pop["behaviour"]
    age_prop = np.array([beh["age_consultation_propensity"][b] for b in bands])[band_idx]
    cond_factor = 1.0 + beh["condition_propensity_increment"] * df["n_conditions"].to_numpy()
    dep_frac = (10 - df["imd_decile"].to_numpy()) / 9.0
    dep_factor = np.power(beh["deprivation_propensity_rr"], dep_frac)
    noise = rng.lognormal(0.0, beh["propensity_lognormal_sigma"], size=n)
    propensity = age_prop * cond_factor * dep_factor * noise
    propensity = propensity / propensity.mean()
    df["consultation_propensity"] = np.round(propensity, 3)

    cont = np.array([beh["continuity_age_base"][b] for b in bands])[band_idx].astype(float)
    cont = cont + beh["continuity_condition_increment"] * df["n_conditions"].to_numpy()
    cont = cont + np.where(df["dementia"].to_numpy(), beh["continuity_dementia_boost"], 0.0)
    cont = cont + np.where(df["smi"].to_numpy(), beh["continuity_smi_boost"], 0.0)
    cont = cont + np.where(df["frail"].to_numpy(), beh["continuity_frailty_boost"], 0.0)
    cont = cont + rng.normal(0.0, beh["continuity_noise_sd"], size=n)
    df["continuity_preference"] = np.round(np.clip(cont, 0.0, 1.0), 3)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def _reference_age_dist(pop: dict) -> dict:
    ref = pop.get("calibration_reference", {})
    return ref.get("age_sex_distribution", pop["age_sex_distribution"])


def calibrate_scales(config: dict, rng: np.random.Generator,
                     return_df: bool = False):
    """Build the reference population and calibrate per-condition scales to QOF."""
    pop = config["population"]
    ref = pop.get("calibration_reference", {})
    ref_size = int(ref.get("size", 20000))
    demo = _draw_demographics(ref_size, pop, _reference_age_dist(pop), rng)
    flags, scales = _sample_registers(demo, pop, rng, scales=None)
    if return_df:
        return scales, _assemble_df(demo, flags)
    return scales, None


def generate_population(config: dict | None = None,
                        seed: int | None = None) -> pd.DataFrame:
    """Generate the synthetic registered list as a DataFrame."""
    if config is None:
        config = load_config()
    pop = config["population"]
    base_seed = config.get("seed", 42) if seed is None else seed
    ref_ss, act_ss = np.random.SeedSequence(base_seed).spawn(2)

    # 1) calibrate per-stratum rates on the fixed reference pyramid
    scales, _ = calibrate_scales(config, np.random.default_rng(ref_ss))

    # 2) build the actual list and apply those fixed rates
    act_rng = np.random.default_rng(act_ss)
    demo = _draw_demographics(int(pop["list_size"]), pop,
                              pop["age_sex_distribution"], act_rng)
    flags, _ = _sample_registers(demo, pop, act_rng, scales=scales)
    df = _assemble_df(demo, flags)
    _add_frailty_and_behaviour(df, demo, pop, act_rng)
    return df


def generate_calibration_reference(config: dict | None = None,
                                   seed: int | None = None) -> pd.DataFrame:
    """Reproduce the reference population (used to verify QOF calibration)."""
    if config is None:
        config = load_config()
    base_seed = config.get("seed", 42) if seed is None else seed
    ref_ss, _ = np.random.SeedSequence(base_seed).spawn(2)
    _, df = calibrate_scales(config, np.random.default_rng(ref_ss), return_df=True)
    return df


if __name__ == "__main__":
    pop_df = generate_population()
    print(pop_df.head())
    print(f"\nGenerated {len(pop_df):,} synthetic patients.")
