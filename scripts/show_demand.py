"""Phase 2 -- generate the year of demand and print a one-page profile.

Run:  python scripts/show_demand.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from gpsim.config import load_config  # noqa: E402
from gpsim.demand import (  # noqa: E402
    build_calendar, daily_demand, generate_demand, intraday_profile,
)
from gpsim.population import generate_population  # noqa: E402

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def bar(value: float, vmax: float, width: int = 40) -> str:
    n = int(round((value / vmax) * width)) if vmax > 0 else 0
    return "#" * n + "." * (width - n)


def main() -> int:
    cfg = load_config()
    pop = generate_population(cfg)
    con = generate_demand(pop, cfg)
    cal = build_calendar(cfg)
    n_days = len(cal)
    n_pat = len(pop)
    m = len(con)

    line = "=" * 74
    print(line)
    print(f"  SYNTHETIC DEMAND PROFILE  --  {m:,} contacts over {n_days} days")
    print(f"  {n_pat:,} patients  ->  {m/n_pat:.2f} contacts per patient per year")
    print("  ALL DATA IS SYNTHETIC -- illustrative model")
    print(line)

    # by month
    print("\nSEASONALITY (contacts per month)")
    by_month = con.groupby("month").size().reindex(range(1, 13), fill_value=0)
    vmax = by_month.max()
    for mth in range(1, 13):
        print(f"  {MONTHS[mth-1]} {by_month[mth]:6,}  {bar(by_month[mth], vmax)}")
    winter = by_month[[12, 1, 2]].mean()
    summer = by_month[[6, 7, 8]].mean()
    print(f"  winter avg/mo {winter:,.0f} vs summer {summer:,.0f}  -> winter is {winter/summer:.2f}x summer")

    # by weekday (average per occurrence of that weekday, working days only)
    print("\nWEEKLY PATTERN (avg contacts per weekday)")
    dd = daily_demand(con, n_days)
    wk = cal["weekday"].to_numpy()
    avg_by_wd = [dd[wk == w].mean() for w in range(7)]
    vmax = max(avg_by_wd)
    for w in range(7):
        print(f"  {WDAYS[w]} {avg_by_wd[w]:6.0f}  {bar(avg_by_wd[w], vmax)}")
    print(f"  Monday is {avg_by_wd[0]/avg_by_wd[2]:.2f}x a Wednesday")

    # intraday on the busiest weekday-day
    busiest = int(np.argmax(np.where(cal["weekend"].to_numpy(), 0, dd)))
    prof = intraday_profile(con, busiest)
    print(f"\nINTRADAY PROFILE (busiest day: {cal['date'].iloc[busiest].date()}, "
          f"{WDAYS[cal['weekday'].iloc[busiest]]}, {dd[busiest]} contacts)")
    vmax = prof.max()
    for h in range(7, 21):
        print(f"  {h:02d}:00 {prof[h]:4d}  {bar(prof[h], vmax, 48)}")
    am8 = prof[8] / prof.sum() if prof.sum() else 0
    print(f"  08:00 hour holds {am8*100:.0f}% of that day's contacts (the 8am rush)")

    # mixes
    def mix(col: str) -> None:
        vc = con[col].value_counts(normalize=True)
        for k, v in vc.items():
            print(f"    {k:<14} {v*100:5.1f}%  {bar(v, vc.max(), 30)}")

    print("\nCONTACT TYPE MIX")
    mix("type")
    print("\nCHANNEL MIX")
    mix("channel")
    print("\nURGENCY MIX (requested)")
    mix("urgency")
    print("\nTRUE ACUITY MIX")
    mix("acuity")

    # a safety teaser: high/emergency acuity arriving as routine/low urgency
    hidden = con[(con["acuity"].isin(["high", "emergency"])) &
                 (con["urgency"] == "routine")]
    print(f"\nSAFETY TEASER: {len(hidden):,} contacts ({len(hidden)/m*100:.1f}%) are truly "
          f"high/emergency acuity but were requested as 'routine'")
    print("  -> these are the under-triage risk Phase 4/5 will track.")

    print("\n" + line)
    print("Demand stream ready. Next: Phase 3 supply/queue engine consumes it.")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
