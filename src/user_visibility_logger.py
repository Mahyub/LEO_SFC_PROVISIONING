"""
src/user_visibility_logger.py
==============================
Serialises per-user satellite visibility and access delays for one
(instance, epoch) snapshot.

For every user in every SFC the record captures:
  - User terminal location (lat, lon)
  - Every satellite that is above min_elevation_deg at this epoch,
    together with its position, elevation angle, and one-way access delay.

The output is intended for offline inspection, not for the solver.
It is written to  data/results/<scenario_id>_user_visibility.json
when config["save_user_visibility"] is True (the default).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .types import SFC, TopologySnapshot
from .visibility import compute_elevation_deg, compute_access_delay_ms


def serialize_user_visibility(
    sfcs: List[SFC],
    topo: TopologySnapshot,
    config: dict,
    instance_id: int,
    epoch: int,
    scenario_id: str,
) -> dict:
    """
    Build a visibility record for one (instance_id, epoch).

    Returns
    -------
    dict with keys:
      scenario_id, instance_id, epoch, min_elevation_deg,
      users  — list of per-user dicts, each containing:
                 sfc_idx, slice_id, user_id, lat, lon,
                 n_visible,
                 visible_satellites  — list sorted by elevation descending,
                   each entry: sat_id, sat_lat, sat_lon, sat_alt_km,
                                elevation_deg, access_delay_ms
    """
    min_el = float(config.get("min_elevation_deg", 10.0))
    users_records: List[dict] = []

    for sfc_idx, sfc in enumerate(sfcs):
        for u in sfc.user_ids:
            u_lat, u_lon = sfc.user_location(u)
            vis_sats: List[dict] = []

            if topo.sat_positions:
                for s, (s_lat, s_lon, s_alt) in topo.sat_positions.items():
                    el = compute_elevation_deg(u_lat, u_lon, s_lat, s_lon, s_alt)
                    if el >= min_el:
                        acc = compute_access_delay_ms(u_lat, u_lon, s_lat, s_lon, s_alt)
                        vis_sats.append({
                            "sat_id":          s,
                            "sat_lat":         round(s_lat, 4),
                            "sat_lon":         round(s_lon, 4),
                            "sat_alt_km":      round(s_alt, 2),
                            "elevation_deg":   round(el, 4),
                            "access_delay_ms": round(acc, 4),
                        })

            # Best (highest elevation) satellite first
            vis_sats.sort(key=lambda x: x["elevation_deg"], reverse=True)

            users_records.append({
                "sfc_idx":           sfc_idx,
                "slice_id":          sfc.slice_id,
                "user_id":           u,
                "lat":               round(u_lat, 6),
                "lon":               round(u_lon, 6),
                "n_visible":         len(vis_sats),
                "visible_satellites": vis_sats,
            })

    return {
        "scenario_id":      scenario_id,
        "instance_id":      instance_id,
        "epoch":            epoch,
        "min_elevation_deg": min_el,
        "users":            users_records,
    }
