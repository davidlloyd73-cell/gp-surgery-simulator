"""Phase 3 -- practice / supply model: the staff roster and capacity.

Builds the list of individual clinicians from the config roster (so later phases
can track *which* clinician a patient saw, for continuity), and provides the
scope-of-practice / routing helpers the simulation uses to match a contact to an
appropriate professional.
"""
from __future__ import annotations

from dataclasses import dataclass, field

GP_ROLES = ("gp_partner", "salaried_gp", "locum_gp")


@dataclass
class Clinician:
    cid: int                 # index in the clinician list
    name: str                # e.g. "salaried_gp_2"
    role: str
    slots_per_day: int
    appt_minutes: float
    urgent_capable: bool
    safe_daily_contacts: int
    scope: frozenset
    is_gp: bool = field(default=False)


def expand_role_token(token: str) -> tuple[str, ...]:
    """'gp' -> the three GP roles; any other token -> itself."""
    return GP_ROLES if token == "gp" else (token,)


def build_clinicians(config: dict) -> list[Clinician]:
    """Materialise one Clinician per individual in the roster."""
    roles = config["practice"]["roles"]
    enforce = config["practice"].get("enforce_safe_limits", False)
    clinicians: list[Clinician] = []
    cid = 0
    for role, spec in roles.items():
        slots = int(spec["slots_per_day"])
        if enforce:
            slots = min(slots, int(spec["safe_daily_contacts"]))
        for k in range(int(spec["clinicians"])):
            clinicians.append(Clinician(
                cid=cid,
                name=f"{role}_{k + 1}",
                role=role,
                slots_per_day=slots,
                appt_minutes=float(spec["appt_minutes"]),
                urgent_capable=bool(spec["urgent_capable"]),
                safe_daily_contacts=int(spec["safe_daily_contacts"]),
                scope=frozenset(spec["scope"]),
                is_gp=role in GP_ROLES,
            ))
            cid += 1
    return clinicians


def routing_for(config: dict, mode: str) -> dict:
    """Return the routing-preference table for the given mode."""
    key = "routing_triage" if mode == "triage" else "routing_traditional"
    return config["practice"][key]


def eligible_clinician_indices(contact_type: str, routing: dict,
                               clinicians: list[Clinician]) -> list[int]:
    """Clinician indices eligible for a contact type, in routing-preference order.

    Honours both the routing preference (ordered roles) and scope-of-practice.
    """
    by_role: dict[str, list[int]] = {}
    for c in clinicians:
        by_role.setdefault(c.role, []).append(c.cid)
    ordered: list[int] = []
    for token in routing.get(contact_type, []):
        for role in expand_role_token(token):
            for cid in by_role.get(role, []):
                if contact_type in clinicians[cid].scope:
                    ordered.append(cid)
    return ordered


def capacity_summary(config: dict) -> dict:
    """Headline daily capacity by role and in total (working day)."""
    roles = config["practice"]["roles"]
    per_role = {r: int(s["clinicians"]) * int(s["slots_per_day"])
                for r, s in roles.items()}
    return {
        "per_role": per_role,
        "total": sum(per_role.values()),
        "gp_total": sum(v for r, v in per_role.items() if r in GP_ROLES),
    }
