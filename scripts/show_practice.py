"""Phase 3 -- run the appointment-book simulation and print a practice profile.

Run:  python scripts/show_practice.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from gpsim.config import load_config  # noqa: E402
from gpsim.demand import generate_demand  # noqa: E402
from gpsim.population import generate_population  # noqa: E402
from gpsim.practice import capacity_summary  # noqa: E402
from gpsim.simulation import simulate  # noqa: E402


def bar(v, vmax, width=34):
    n = int(round((v / vmax) * width)) if vmax > 0 else 0
    return "#" * n + "." * (width - n)


def main() -> int:
    cfg = load_config()
    pop = generate_population(cfg)
    con = generate_demand(pop, cfg)
    res = simulate(pop, con, cfg, routing_mode="traditional")
    c = res["contacts"]
    k = res["kpis"]

    line = "=" * 76
    print(line)
    print("  PRACTICE / APPOINTMENT-BOOK PROFILE  (traditional direct-booking)")
    print("  ALL DATA IS SYNTHETIC -- illustrative model")
    print(line)

    cap = capacity_summary(cfg)
    demand_per_wd = c.groupby("day").size()
    print(f"\nCAPACITY vs DEMAND")
    print(f"  total daily clinical capacity : {cap['total']:,} slots/working day")
    print(f"  of which GP slots             : {cap['gp_total']:,}")
    print(f"  avg demand per working day    : ~{int(demand_per_wd.mean())} contacts")

    print(f"\nHEADLINE KPIs")
    for key in ["same_day_rate", "seen_rate", "escalation_rate", "unmet_rate",
                "mean_routine_wait_days", "median_routine_wait_days",
                "pct_seen_within_2_days", "mean_sameday_wait_min",
                "overall_utilisation", "gp_utilisation", "locum_share_of_appts",
                "safe_limit_breach_rate", "max_load_vs_safe"]:
        print(f"  {key:28} {k[key]}")

    # access by contact type
    print(f"\nACCESS BY CONTACT TYPE  (same-day % / mean wait days / escalation %)")
    for t, g in c.groupby("type"):
        seen = g[g.status == "seen"]
        sd = (g.same_day & (g.status == "seen")).mean()
        wd = seen[~seen.same_day]["wait_days"]
        esc = (g.status == "escalated").mean()
        print(f"  {t:<15} same-day {sd*100:5.1f}%   wait {wd.mean() if len(wd) else 0:5.1f}d   esc {esc*100:4.1f}%")

    # who delivered care
    print(f"\nWORKLOAD BY ROLE  (utilisation, mean appts/working day)")
    ru = res["role_util"].sort_values("utilisation", ascending=False)
    umax = ru["utilisation"].max()
    for _, r in ru.iterrows():
        print(f"  {r['role']:<26} util {r['utilisation']*100:5.1f}%  {bar(r['utilisation'], umax)}  ({r['used_per_day']:.0f}/{r['capacity_per_day']})")

    # routine wait distribution for GP-delivered care
    gp_seen = c[(c.seen_role.isin(["gp_partner", "salaried_gp", "locum_gp"])) & (c.status == "seen")]
    gpw = gp_seen[~gp_seen.same_day]["wait_days"]
    print(f"\nGP ROUTINE WAIT (days):  mean {gpw.mean():.1f}  median {gpw.median():.0f}  "
          f"90th pct {gpw.quantile(0.9):.0f}  max {gpw.max():.0f}")

    # focus-day intraday same-day queue
    fp = res["focus_profile"]
    if fp is not None:
        d = fp["day"]
        date = res["daily"].loc[d, "date"]
        qlen = fp["qlen"]
        mins = fp["minutes"]
        print(f"\nINTRADAY SAME-DAY QUEUE  (busiest same-day day: {str(date)[:10]})")
        peak = qlen.max()
        for h in range(8, 19):
            mask = (mins >= h * 60) & (mins < (h + 1) * 60)
            avg_q = qlen[mask].mean() if mask.any() else 0
            print(f"  {h:02d}:00  avg queue {avg_q:4.1f}  {bar(avg_q, max(peak,1))}")
        print(f"  peak queue: {peak} patients waiting at once")

    print("\n" + line)
    print("Supply engine ready. Next: Phase 4 layers triage + AI + continuity.")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
