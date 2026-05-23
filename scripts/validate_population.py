"""Phase 1 validation -- regenerate the list and check it is plausible.

Run:  python -m scripts.validate_population
  or: python scripts/validate_population.py

Confirms realised prevalences, age/sex split and multimorbidity counts match
the configured targets within tolerance, prints a one-page "list profile", and
exits non-zero if any HARD check fails (so it doubles as a smoke test).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a plain script (python scripts/validate_population.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from gpsim.config import load_config  # noqa: E402
from gpsim.population import (  # noqa: E402
    LTC_COLUMNS, generate_calibration_reference, generate_population,
)

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"


def _status(ok: bool) -> str:
    return f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"


def _bar(frac: float, width: int = 24) -> str:
    filled = int(round(frac * width))
    return "#" * filled + "." * (width - filled)


def main() -> int:
    cfg = load_config()
    pop_cfg = cfg["population"]
    val = cfg["validation"]
    df = generate_population(cfg)
    n = len(df)

    failures: list[str] = []
    warnings: list[str] = []

    line = "=" * 78
    print(line)
    print(f"  SYNTHETIC GP LIST PROFILE  --  {n:,} patients  (seed={cfg.get('seed')})")
    print("  ALL DATA IS SYNTHETIC -- illustrative model, not real patient records")
    print(line)

    # --- age / sex ---------------------------------------------------------
    print("\nAGE / SEX")
    print(f"  Mean age: {df['age'].mean():5.1f}    Median age: {df['age'].median():5.1f}")
    male_pct = (df["sex"] == "M").mean()
    print(f"  Male: {male_pct*100:4.1f}%    Female: {(1-male_pct)*100:4.1f}%")
    print(f"  {'band':>6} {'target':>7} {'actual':>7}   distribution")
    asd = pop_cfg["age_sex_distribution"]
    tot_share = sum(asd[b]["share"] for b in pop_cfg["age_bands"])
    for b in pop_cfg["age_bands"]:
        tgt = asd[b]["share"] / tot_share
        act = (df["age_band"] == b).mean()
        ok = abs(act - tgt) <= 0.02
        if not ok:
            failures.append(f"age band {b}: target {tgt:.3f} vs actual {act:.3f}")
        print(f"  {b:>6} {tgt*100:6.1f}% {act*100:6.1f}%   {_bar(act)} {_status(ok)}")

    # --- ethnicity ---------------------------------------------------------
    print("\nETHNICITY")
    eth = pop_cfg["ethnicity_distribution"]
    tot_eth = sum(eth.values())
    for name, raw in eth.items():
        tgt = raw / tot_eth
        act = (df["ethnicity"] == name).mean()
        ok = abs(act - tgt) <= 0.02
        if not ok:
            failures.append(f"ethnicity {name}: target {tgt:.3f} vs actual {act:.3f}")
        print(f"  {name:<16} {tgt*100:5.1f}% -> {act*100:5.1f}%  {_bar(act)} {_status(ok)}")
    print(f"  South Asian (grouped): {df['south_asian'].mean()*100:4.1f}%")

    # --- calibration check: reference reproduces QOF targets ---------------
    print("\nCALIBRATION CHECK  (reference Harrow pyramid reproduces QOF targets)")
    abs_tol = val["prevalence_abs_tol"]
    rel_tol = val["prevalence_rel_tol"]
    ref = generate_calibration_reference(cfg)
    print(f"  {'condition':<22}{'QOF tgt':>8}{'ref':>8}{'diff':>8}   status")
    for cond in LTC_COLUMNS:
        tgt = pop_cfg["conditions"][cond]["target_prevalence"]
        act = ref[cond].mean()
        diff = act - tgt
        ok = (abs(diff) <= abs_tol) or (tgt > 0 and abs(diff) / tgt <= rel_tol)
        if not ok:
            failures.append(f"calibration {cond}: target {tgt:.3f} vs reference {act:.3f}")
        label = pop_cfg["conditions"][cond]["label"]
        print(f"  {label:<22}{tgt*100:7.2f}%{act*100:7.2f}%{diff*100:+7.2f}%   {_status(ok)}")

    # --- disease registers on THIS list (crude prevalence floats) ----------
    print("\nDISEASE REGISTERS  (this list -- crude prevalence floats with case-mix)")
    print(f"  {'condition':<22}{'QOF tgt':>8}{'actual':>8}{'vs nat':>8}   status")
    for cond in LTC_COLUMNS:
        tgt = pop_cfg["conditions"][cond]["target_prevalence"]
        act = df[cond].mean()
        ratio = act / tgt if tgt > 0 else float("nan")
        # Hard check is a sane band only -- legitimate case-mix shifts are fine.
        ok = 0.25 <= ratio <= 4.0
        if not ok:
            failures.append(f"prevalence {cond}: actual {act:.3f} implausible vs target {tgt:.3f}")
        label = pop_cfg["conditions"][cond]["label"]
        print(f"  {label:<22}{tgt*100:7.2f}%{act*100:7.2f}%{ratio:7.2f}x   {_status(ok)}")

    # --- multimorbidity (soft) --------------------------------------------
    print("\nMULTIMORBIDITY")
    counts = df["n_conditions"]
    for k in range(5):
        frac = (counts == k).mean()
        print(f"  exactly {k} LTC: {frac*100:5.1f}%  {_bar(frac)}")
    print(f"  5+ LTC:       {(counts >= 5).mean()*100:5.1f}%")
    adults = df[df["age"] >= 18]
    a2 = (adults["n_conditions"] >= 2).mean()
    a3 = (adults["n_conditions"] >= 3).mean()
    mm = val["multimorbidity"]
    ok2 = abs(a2 - mm["adult_2plus"]["target"]) <= mm["adult_2plus"]["tol"]
    ok3 = abs(a3 - mm["adult_3plus"]["target"]) <= mm["adult_3plus"]["tol"]
    print(f"  adults (18+) with 2+ LTC: {a2*100:4.1f}%  (target {mm['adult_2plus']['target']*100:.0f}%) {_status(ok2)}")
    print(f"  adults (18+) with 3+ LTC: {a3*100:4.1f}%  (target {mm['adult_3plus']['target']*100:.0f}%) {_status(ok3)}")
    if not ok2:
        warnings.append(f"adult 2+ multimorbidity {a2:.3f} outside target band")
    if not ok3:
        warnings.append(f"adult 3+ multimorbidity {a3:.3f} outside target band")

    # --- local epidemiology (South Asian elevation) -----------------------
    print("\nLOCAL EPIDEMIOLOGY  (South Asian vs rest)")
    sa = df[df["south_asian"]]
    rest = df[~df["south_asian"]]
    for cond, key in [("diabetes", "south_asian_diabetes_min_ratio"),
                      ("ihd", "south_asian_ihd_min_ratio")]:
        sa_p = sa[cond].mean()
        rest_p = rest[cond].mean()
        ratio = sa_p / rest_p if rest_p > 0 else float("nan")
        ok = ratio >= val[key]
        if not ok:
            failures.append(f"{cond} SA/rest ratio {ratio:.2f} < {val[key]}")
        print(f"  {cond:<10} SA {sa_p*100:4.1f}% vs rest {rest_p*100:4.1f}%  ratio {ratio:4.2f} (>= {val[key]}) {_status(ok)}")

    # --- behavioural attributes -------------------------------------------
    print("\nBEHAVIOURAL ATTRIBUTES")
    print(f"  consultation_propensity: mean {df['consultation_propensity'].mean():.2f} "
          f"(should be ~1.0), range {df['consultation_propensity'].min():.2f}-{df['consultation_propensity'].max():.2f}")
    print(f"  continuity_preference:   mean {df['continuity_preference'].mean():.2f}, "
          f"high-need (3+ LTC) mean {df[df['n_conditions']>=3]['continuity_preference'].mean():.2f}")
    print(f"  frailty: {(df['frail']).sum():,} flagged frail "
          f"({df['frail'].mean()*100:.1f}% of list; "
          f"{df[df['age']>=65]['frail'].mean()*100:.1f}% of 65+)")

    # --- summary -----------------------------------------------------------
    print("\n" + line)
    if warnings:
        print(f"{YELLOW}WARNINGS (soft checks):{RESET}")
        for w in warnings:
            print(f"  - {w}")
    if failures:
        print(f"{RED}VALIDATION FAILED -- {len(failures)} hard check(s) out of tolerance:{RESET}")
        for f in failures:
            print(f"  - {f}")
        print(line)
        return 1
    print(f"{GREEN}VALIDATION PASSED -- population is plausible within tolerance.{RESET}")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
