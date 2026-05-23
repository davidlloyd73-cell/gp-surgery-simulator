"""Phase 5 -- print the full grouped metric report and scenario comparison.

Run:  python scripts/show_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpsim.config import load_config  # noqa: E402
from gpsim.metrics import build_world, frontier, run_scenario, sweep_lever  # noqa: E402

GROUP_TITLES = {
    "responsiveness": "RESPONSIVENESS / ACCESS",
    "continuity": "CONTINUITY",
    "capacity": "CAPACITY / WORKLOAD",
    "safety": "SAFETY PROXIES",
    "efficiency": "EFFICIENCY",
}


def main() -> int:
    world = build_world(load_config())
    line = "=" * 76

    print(line)
    print("  FULL METRIC REPORT  --  baseline scenario (all data synthetic)")
    print(line)
    metrics = run_scenario(world)["metrics"]
    print("\nLEVERS: " + "  ".join(f"{k}={v}" for k, v in metrics["levers"].items()))
    for group in ("responsiveness", "continuity", "capacity", "safety", "efficiency"):
        print(f"\n{GROUP_TITLES[group]}")
        for name, val in metrics[group].items():
            if name == "role_utilisation":
                continue
            print(f"  {name:<40} {val}")

    print("\n" + line)
    print("  CONTINUITY-vs-RESPONSIVENESS FRONTIER  (each row = a point to plot)")
    print(line)
    scenarios = [
        ("Traditional, continuity off", {"triage_model": "traditional", "continuity_policy_strength": 0.0}),
        ("Traditional, continuity mid", {"triage_model": "traditional", "continuity_policy_strength": 0.5}),
        ("Traditional, continuity max", {"triage_model": "traditional", "continuity_policy_strength": 1.0}),
        ("Total triage", {"triage_model": "total_triage", "continuity_policy_strength": 0.5}),
        ("Total triage + full AI", {"triage_model": "total_triage", "continuity_policy_strength": 0.5,
                                    "ai_scribe_adoption": 1.0, "ai_admin_adoption": 1.0, "ai_triage_adoption": 1.0}),
    ]
    fr = frontier(world, scenarios)
    print()
    cols = ["scenario", "responsiveness_same_day", "continuity_chronic_usual",
            "upc_high_need", "gp_utilisation", "mean_wait_days", "missed_urgent", "cost_per_patient"]
    print(f"  {'scenario':<28}{'same-day':>9}{'contin.':>8}{'UPC hi':>7}{'GPutil':>7}{'wait':>6}{'missed':>7}{'£/pt':>7}")
    for _, r in fr.iterrows():
        print(f"  {r['scenario']:<28}{r['responsiveness_same_day']:>9.2f}{r['continuity_chronic_usual']:>8.2f}"
              f"{r['upc_high_need']:>7.2f}{r['gp_utilisation']:>7.2f}{r['mean_wait_days']:>6.1f}"
              f"{int(r['missed_urgent']):>7}{r['cost_per_patient']:>7.0f}")

    print("\n" + line)
    print("  CONTINUITY-POLICY SWEEP (traditional booking)")
    print(line)
    sw = sweep_lever(world, "continuity_policy_strength", [0.0, 0.25, 0.5, 0.75, 1.0],
                     base_levers={"triage_model": "traditional"})
    print(f"\n  {'cont':>5}{'same-day':>9}{'wait':>6}{'UPC hi':>8}{'chronic/usual':>14}")
    for _, r in sw.iterrows():
        print(f"  {r['continuity_policy_strength']:>5.2f}{r['pct_same_day']:>9.2f}{r['mean_wait_days']:>6.1f}"
              f"{r['upc_high_need']:>8.2f}{r['pct_chronic_with_usual']:>14.2f}")

    print("\n" + line)
    print("Metrics ready. Next: Phase 6 stress scenarios, Phase 7 dashboard.")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
