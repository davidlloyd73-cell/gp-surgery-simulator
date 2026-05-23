"""Phase 6 -- run the stress scenarios and show before/after on the same KPIs.

Run:  python scripts/show_scenarios.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpsim.config import load_config  # noqa: E402
from gpsim.scenarios import compare_scenarios, list_scenarios  # noqa: E402


def main() -> int:
    cfg = load_config()
    line = "=" * 92

    print(line)
    print("  STRESS SCENARIOS  --  before/after on the same KPIs (all data synthetic)")
    print(line)
    print("\nAvailable scenarios:")
    for name, desc in list_scenarios(cfg).items():
        print(f"  - {name:<16} {desc}")

    names = list(list_scenarios(cfg).keys())
    df = compare_scenarios(names, cfg)
    base = df.iloc[0]

    print(f"\n{'scenario':<16}{'same-day':>9}{'wait(d)':>8}{'unmet':>7}{'GPutil':>7}"
          f"{'esc/yr':>8}{'missed':>7}{'£/pt':>7}")
    for _, r in df.iterrows():
        print(f"  {r['scenario']:<14}{r['pct_same_day']:>9.2f}{r['mean_wait_days']:>8.1f}"
              f"{r['unmet_rate']:>7.3f}{r['gp_utilisation']:>7.2f}{int(r['escalations_111_ae']):>8}"
              f"{int(r['missed_urgent']):>7}{r['cost_per_patient']:>7.0f}")
    print(f"\n  (baseline same-day {base['pct_same_day']:.0%}, wait {base['mean_wait_days']:.1f}d -- "
          "scenarios show how far each shock pushes the practice)")

    # Can AI help weather a winter surge?
    print("\n" + line)
    print("  CAN AI MITIGATE A WINTER SURGE?  (winter_surge scenario, varying AI)")
    print(line)
    combos = [
        ("no AI", {}),
        ("AI scribe (full)", {"ai_scribe_adoption": 1.0}),
        ("AI scribe + admin", {"ai_scribe_adoption": 1.0, "ai_admin_adoption": 1.0}),
        ("total triage + full AI", {"triage_model": "total_triage", "ai_scribe_adoption": 1.0,
                                    "ai_admin_adoption": 1.0, "ai_triage_adoption": 1.0}),
    ]
    print(f"\n  {'AI configuration':<24}{'same-day':>9}{'wait(d)':>8}{'GPutil':>7}{'missed':>7}")
    for label, levers in combos:
        row = compare_scenarios(["winter_surge"], cfg, levers=levers).iloc[0]
        print(f"  {label:<24}{row['pct_same_day']:>9.2f}{row['mean_wait_days']:>8.1f}"
              f"{row['gp_utilisation']:>7.2f}{int(row['missed_urgent']):>7}")

    print("\n" + line)
    print("Scenarios ready. Next: Phase 7 -- the Streamlit dashboard ties it all together.")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
