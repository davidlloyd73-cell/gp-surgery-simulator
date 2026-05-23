"""Phase 4 -- demonstrate the triage / AI / continuity levers and the
continuity-vs-responsiveness frontier.

Run:  python scripts/show_levers.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpsim.config import load_config  # noqa: E402
from gpsim.demand import generate_demand  # noqa: E402
from gpsim.population import generate_population  # noqa: E402
from gpsim.simulation import simulate  # noqa: E402


def main() -> int:
    cfg = load_config()
    pop = generate_population(cfg)
    con = generate_demand(pop, cfg)

    def run(**lev):
        return simulate(pop, con, cfg, levers=lev)["kpis"]

    line = "=" * 78
    print(line)
    print("  TRIAGE / AI / CONTINUITY LEVERS  (all data synthetic)")
    print(line)

    print("\n1) CONTINUITY-vs-RESPONSIVENESS FRONTIER (traditional booking)")
    print("   Pushing continuity trades away same-day access. Each row is a point")
    print("   on the frontier the dashboard plots.\n")
    print(f"   {'continuity':>10} {'UPC':>6} {'UPC(hi-need)':>13} {'chronic w/usual':>16} {'same-day':>9} {'GP wait':>8}")
    for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
        k = run(triage_model="traditional", continuity_policy_strength=s)
        print(f"   {s:>10.2f} {k['continuity_upc']:>6.2f} {k['continuity_upc_high_need']:>13.2f} "
              f"{k['pct_chronic_with_usual']:>16.2f} {k['same_day_rate']:>9.2f} {k['mean_routine_wait_days']:>7.1f}d")

    print("\n2) TRADITIONAL vs TOTAL TRIAGE (continuity policy 0.5)")
    print("   Total triage diverts demand and frees GPs, but the triage step")
    print("   itself consumes capacity.\n")
    print(f"   {'model':>14} {'GP util':>8} {'same-day':>9} {'GP wait':>8} {'diverted':>9} {'esc':>6} {'triage burden':>14}")
    for tm in ["traditional", "total_triage"]:
        k = run(triage_model=tm, continuity_policy_strength=0.5)
        print(f"   {tm:>14} {k['gp_utilisation']:>8.2f} {k['same_day_rate']:>9.2f} "
              f"{k['mean_routine_wait_days']:>7.1f}d {k['diversion_rate']:>9.2f} "
              f"{k['escalation_rate']:>6.3f} {k['triage_capacity_burden_slots_per_day']:>11.0f} sl")

    print("\n3) AI AMBIENT SCRIBE (the lever of most interest): raises throughput")
    print(f"   {'adoption':>9} {'GP capacity/day':>16} {'GP util':>8} {'same-day':>9} {'GP wait':>8}")
    for a in [0.0, 0.5, 1.0]:
        res = simulate(pop, con, cfg, levers={"triage_model": "traditional", "ai_scribe_adoption": a})
        k = res["kpis"]
        gpcap = res["role_util"].set_index("role").loc[["gp_partner", "salaried_gp", "locum_gp"], "capacity_per_day"].sum()
        print(f"   {a:>9.1f} {gpcap:>16.0f} {k['gp_utilisation']:>8.2f} {k['same_day_rate']:>9.2f} {k['mean_routine_wait_days']:>7.1f}d")

    print("\n4) AI ADMIN: absorbs admin contacts (no appointment needed)")
    print(f"   {'adoption':>9} {'resolved w/o appt':>18} {'GP wait':>8}")
    for a in [0.0, 0.5, 1.0]:
        k = run(triage_model="traditional", ai_admin_adoption=a)
        print(f"   {a:>9.1f} {k['resolved_no_appt_rate']:>18.2f} {k['mean_routine_wait_days']:>7.1f}d")

    print("\n5) AI TRIAGE -- the SAFETY trade-off (total triage, full AI triage)")
    print("   Faster/cheaper triage, but lower accuracy => more under-triaged urgent cases.\n")
    print(f"   {'accuracy':>9} {'triage burden':>14} {'under-triaged':>14} {'missed urgent':>14}")
    for acc in [0.99, 0.95, 0.90, 0.80, 0.65]:
        k = run(triage_model="total_triage", ai_triage_adoption=1.0, ai_triage_accuracy=acc)
        print(f"   {acc:>9.2f} {k['triage_capacity_burden_slots_per_day']:>11.0f} sl "
              f"{k['under_triage_count']:>14d} {k['missed_urgent_count']:>14d}")

    print("\n6) THE FRONTIER AS POINTS  (x = responsiveness = same-day %,")
    print("   y = continuity = chronic reviews with usual clinician)")
    scenarios = [
        ("Trad, continuity off", {"triage_model": "traditional", "continuity_policy_strength": 0.0}),
        ("Trad, continuity max", {"triage_model": "traditional", "continuity_policy_strength": 1.0}),
        ("Total triage", {"triage_model": "total_triage", "continuity_policy_strength": 0.5}),
        ("Total triage + AI", {"triage_model": "total_triage", "continuity_policy_strength": 0.5,
                               "ai_scribe_adoption": 1.0, "ai_admin_adoption": 1.0, "ai_triage_adoption": 1.0}),
    ]
    grid = [[" "] * 40 for _ in range(11)]
    pts = []
    for name, lev in scenarios:
        k = run(**lev)
        x, y = k["same_day_rate"], k["pct_chronic_with_usual"]
        pts.append((name, x, y))
        gx = min(39, int(x * 39)); gy = min(10, int(y * 10))
        grid[10 - gy][gx] = "*"
    print()
    for row in grid:
        print("   |" + "".join(row))
    print("   +" + "-" * 40)
    print("    responsiveness (same-day %) ->   (y-axis up = continuity)")
    for name, x, y in pts:
        print(f"     * {name:<22} same-day={x:.2f}  continuity={y:.2f}")

    print("\n" + line)
    print("Levers wired. Next: Phase 5 formal metrics, Phase 6 scenarios, Phase 7 dashboard.")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
