"""
src/placement_logger.py
========================
Serializes a solved PlacementState into a fully detailed dict capturing:
  - SFC chain definitions and ground-station metadata
  - Per-user, per-VNF satellite assignment (satellite id, lat/lon/alt)
  - Per-hop delay breakdown: access delay, ISL shortest-path delay, VNF
    processing delay
  - Per-user E2E total vs. budget and compliance flag

Called from experiment.py when config["save_placements"] is True.

Delay model
-----------
  E2E(n, u) = d_access(n, s_0)
            + sum_{l=1}^{L-1} d_ISL_SP(s_{l-1}, s_l)
            + sum_{l=0}^{L-1} d_proc(f_l, i_l, s_l)

where d_ISL_SP is the Dijkstra shortest-path delay (ms) between consecutive
function satellites, computed once via all_pairs_sp_delays().

Note on repeated function types
--------------------------------
PlacementState.assignment is keyed by (slice_id, user_id, func_type).  If a
chain contains the same VNF type at two positions (e.g. ["FW","IDS","FW"]),
only the last assignment for that type is stored — the first is overwritten.
Hops for a repeated type that was overwritten are marked with
"note": "type_repeated_last_assignment_used".
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .types import (
    FuncID, InstID, PlacementState, RiskParameters,
    SatID, SFC, SliceID, TopologySnapshot, UserID, VNFInstance,
)
from .visibility import compute_access_delay_ms, all_pairs_sp_delays


# Known ground-station coordinates → human-readable name
_KNOWN_GS: List[Tuple[float, float, str]] = [
    (51.5,  -0.1,   "London"),
    (40.7,  -74.0,  "New York"),
    (35.7,   139.7, "Tokyo"),
]


def _gs_name(lat: float, lon: float) -> str:
    for la, lo, name in _KNOWN_GS:
        if abs(lat - la) < 0.3 and abs(lon - lo) < 0.3:
            return name
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{abs(lat):.1f}°{ns} {abs(lon):.1f}°{ew}"


def serialize_placement(
    placement: PlacementState,
    sfcs: List[SFC],
    topo: TopologySnapshot,
    vnf_instances: Dict[Tuple[FuncID, InstID, SatID], VNFInstance],
    method: str,
    instance_id: int,
    epoch: int,
    scenario_id: str,
) -> dict:
    """
    Build a detailed placement record for one (instance, epoch, method).

    Returns
    -------
    dict with keys:
      scenario_id, instance_id, epoch, method,
      slices      — list of SFC descriptors
      assignments — flat list of per-(user, hop) assignment rows
      delays      — per-user E2E delay summary
    """
    # All-pairs shortest-path ISL delays (ms) — computed once per epoch
    sp: Dict[Tuple[SatID, SatID], float] = (
        all_pairs_sp_delays(topo) if topo.sat_positions else {}
    )

    record: dict = {
        "scenario_id": scenario_id,
        "instance_id": instance_id,
        "epoch":       epoch,
        "method":      method,
        "slices":      [],
        "assignments": [],
        "delays":      [],
    }

    for sfc in sfcs:
        # Detect repeated function types in this chain
        seen: Dict[FuncID, int] = {}
        for ft in sfc.functions:
            seen[ft] = seen.get(ft, 0) + 1
        repeated_types = {ft for ft, cnt in seen.items() if cnt > 1}

        record["slices"].append({
            "slice_id":       sfc.slice_id,
            "ground_lat":     round(sfc.ground_lat, 4),
            "ground_lon":     round(sfc.ground_lon, 4),
            "location":       _gs_name(sfc.ground_lat, sfc.ground_lon),
            "criticality":    round(sfc.criticality, 4),
            "functions":      list(sfc.functions),
            "chain_len":      len(sfc.functions),
            "repeated_types": sorted(repeated_types),
            "e2e_budgets_ms": {
                str(u): (round(sfc.e2e_budget_ms[u], 2)
                         if sfc.e2e_budget_ms.get(u, 1e12) < 1e9 else None)
                for u in sfc.user_ids
            },
        })

        for u in sfc.user_ids:
            budget_ms = sfc.e2e_budget_ms.get(u, 1e12)

            access_ms = 0.0   # ground → ingress satellite
            total_isl = 0.0   # sum of ISL hops
            total_vnf = 0.0   # sum of VNF processing
            prev_sat: Optional[SatID] = None

            for f_pos, f_type in enumerate(sfc.functions):
                key = (sfc.slice_id, u, f_pos, f_type)
                val = placement.assignment.get(key)

                note = ""
                if f_type in repeated_types:
                    note = "type_repeated_last_assignment_used"

                if val is None:
                    record["assignments"].append({
                        "slice_id":        sfc.slice_id,
                        "user_id":         u,
                        "f_pos":           f_pos,
                        "f_type":          f_type,
                        "satellite":       None,
                        "inst_id":         None,
                        "sat_lat":         None,
                        "sat_lon":         None,
                        "sat_alt_km":      None,
                        "access_delay_ms": None,
                        "isl_delay_ms":    None,
                        "vnf_delay_ms":    None,
                        "note":            "unassigned",
                    })
                    continue

                inst_id, sat_id = val
                pos = topo.sat_positions.get(sat_id)
                sat_lat = round(pos[0], 4) if pos else None
                sat_lon = round(pos[1], 4) if pos else None
                sat_alt = round(pos[2], 2) if pos else None

                vi = vnf_instances.get((f_type, inst_id, sat_id))
                vnf_ms = round(vi.proc_delay_ms, 4) if vi else 0.0
                total_vnf += vnf_ms

                # Access delay: only for the ingress VNF (f_pos == 0).
                # Use this user's terminal location, not the slice centre.
                hop_access: Optional[float] = None
                if f_pos == 0 and pos is not None:
                    u_lat, u_lon = sfc.user_location(u)
                    hop_access = round(
                        compute_access_delay_ms(
                            u_lat, u_lon,
                            pos[0], pos[1], pos[2]
                        ), 4
                    )
                    access_ms = hop_access

                # ISL delay from previous function's satellite
                hop_isl: Optional[float] = None
                if f_pos > 0 and prev_sat is not None:
                    if sat_id == prev_sat:
                        hop_isl = 0.0
                    else:
                        hop_isl = round(sp.get((prev_sat, sat_id), float("inf")), 4)
                    total_isl += hop_isl

                prev_sat = sat_id

                record["assignments"].append({
                    "slice_id":        sfc.slice_id,
                    "user_id":         u,
                    "f_pos":           f_pos,
                    "f_type":          f_type,
                    "satellite":       sat_id,
                    "inst_id":         inst_id,
                    "sat_lat":         sat_lat,
                    "sat_lon":         sat_lon,
                    "sat_alt_km":      sat_alt,
                    "access_delay_ms": hop_access,
                    "isl_delay_ms":    hop_isl,
                    "vnf_delay_ms":    vnf_ms,
                    "note":            note,
                })

            e2e_ms = access_ms + total_isl + total_vnf
            record["delays"].append({
                "slice_id":        sfc.slice_id,
                "user_id":         u,
                "access_delay_ms": round(access_ms, 4),
                "total_isl_ms":    round(total_isl, 4),
                "total_vnf_ms":    round(total_vnf, 4),
                "e2e_total_ms":    round(e2e_ms, 4),
                "budget_ms":       round(budget_ms, 2) if budget_ms < 1e9 else None,
                "budget_ok":       bool(e2e_ms <= budget_ms + 1e-6),
            })

    return record
