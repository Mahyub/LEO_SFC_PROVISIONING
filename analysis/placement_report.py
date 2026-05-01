"""
analysis/placement_report.py
=============================
Loads a saved *_placements.json file and renders per-epoch placement tables
showing satellite assignments, positions, and delay breakdowns for every
user in every slice.  Uses only ASCII output so it works on any terminal.

Usage
-----
  # From command line:
  python -m analysis.placement_report --file data/results/base_placements.json

  # Specific method / epoch:
  python -m analysis.placement_report --file ... --method proposed_coarse --epoch 0

  # From code (e.g. during an experiment run):
  from analysis.placement_report import print_placement_record
  print_placement_record(record)   # record = one entry from placement_records list
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_placements(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def display_placements(
    path: str,
    method: Optional[str] = None,
    epochs: Optional[List[int]] = None,
    instance_id: int = 0,
) -> None:
    """
    Load a placements file and print all matching records.

    Parameters
    ----------
    path        : path to *_placements.json
    method      : filter by method name (None = all methods)
    epochs      : filter by epoch list (None = all epochs)
    instance_id : filter by instance id (default 0)
    """
    data = load_placements(path)
    records = data.get("records", [])

    filtered = [
        r for r in records
        if r["instance_id"] == instance_id
        and (method is None or r["method"] == method)
        and (epochs is None or r["epoch"] in epochs)
    ]

    if not filtered:
        print("No matching placement records found.")
        return

    by_epoch: Dict[int, List[dict]] = defaultdict(list)
    for r in filtered:
        by_epoch[r["epoch"]].append(r)

    for epoch in sorted(by_epoch):
        for rec in by_epoch[epoch]:
            print_placement_record(rec)


def print_placement_record(rec: dict) -> None:
    """Render one placement record (one epoch, one method) to stdout."""
    epoch  = rec["epoch"]
    method = rec["method"]
    slices = rec.get("slices", [])

    # Index assignments and delays
    asgn: Dict[int, Dict[int, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for a in rec.get("assignments", []):
        asgn[a["slice_id"]][a["user_id"]].append(a)

    dlay: Dict[int, Dict[int, dict]] = defaultdict(dict)
    for d in rec.get("delays", []):
        dlay[d["slice_id"]][d["user_id"]] = d

    W = 100
    div = "=" * W

    print()
    print(div)
    print(f"  Epoch {epoch}  |  {method}  |  Instance {rec['instance_id']}")
    print(div)

    for sl in slices:
        sid      = sl["slice_id"]
        loc      = sl["location"]
        lat      = sl["ground_lat"]
        lon      = sl["ground_lon"]
        crit     = sl["criticality"]
        funcs    = sl["functions"]
        repeated = sl.get("repeated_types", [])

        lat_str = f"{abs(lat):.1f}{'N' if lat >= 0 else 'S'}"
        lon_str = f"{abs(lon):.1f}{'E' if lon >= 0 else 'W'}"
        chain_s = " -> ".join(f"{f}[{i}]" for i, f in enumerate(funcs))
        budgets = sl.get("e2e_budgets_ms", {})

        print()
        print(f"  +-- Slice {sid} | {loc} ({lat_str}, {lon_str}) | Criticality {crit:.3f} " + "-" * 20)
        print(f"  | Chain : {chain_s}")
        print(f"  | Users : {list(asgn[sid].keys())}   "
              f"E2E budgets (ms): { {k: v for k, v in budgets.items()} }")
        if repeated:
            print(f"  | (!) Repeated VNF types in chain: {', '.join(repeated)}"
                  " -- only last assignment per type is stored")

        # Column header
        print()
        hdr = (f"  {'User':<5} {'VNF':<5} {'f_pos':<5} "
               f"{'Sat (inst)':<12} {'Lat':>8} {'Lon':>9} {'Alt':>7}  "
               f"{'Access ms':>10} {'ISL ms':>8} {'VNF ms':>8} {'E2E ms':>8}  Budget")
        print(hdr)
        print("  " + "-" * (W - 2))

        for u_id in sorted(asgn[sid].keys()):
            hops  = sorted(asgn[sid][u_id], key=lambda x: x["f_pos"])
            d_row = dlay[sid].get(u_id, {})
            budget = d_row.get("budget_ms")
            ok     = d_row.get("budget_ok", True)
            e2e    = d_row.get("e2e_total_ms")

            for h_idx, hop in enumerate(hops):
                sat    = hop["satellite"]
                inst   = hop["inst_id"]
                f_type = hop["f_type"]
                f_pos  = hop["f_pos"]
                s_lat  = hop["sat_lat"]
                s_lon  = hop["sat_lon"]
                s_alt  = hop["sat_alt_km"]
                acc    = hop["access_delay_ms"]
                isl    = hop["isl_delay_ms"]
                vnf    = hop["vnf_delay_ms"]
                note   = hop.get("note", "")

                user_s = f"U{u_id}" if h_idx == 0 else ""
                sat_s  = f"S{sat}(i{inst})" if sat is not None else "---"
                lat_s  = f"{s_lat:+.2f}" if s_lat is not None else "---"
                lon_s  = f"{s_lon:+.2f}" if s_lon is not None else "---"
                alt_s  = f"{s_alt:.0f}km" if s_alt is not None else "---"
                acc_s  = f"{acc:.3f}" if acc is not None else "---"
                isl_s  = f"{isl:.3f}" if isl is not None else "---"
                vnf_s  = f"{vnf:.3f}" if vnf is not None else "---"

                is_last = (h_idx == len(hops) - 1)
                e2e_s   = f"{e2e:.3f}" if (is_last and e2e is not None) else ""
                rpt_s   = " (*)" if note else ""

                bgt_s = ""
                if is_last and budget is not None:
                    verdict = "OK" if ok else "OVER"
                    slack   = budget - (e2e or 0.0)
                    bgt_s   = f"{budget:.1f}ms [{verdict} slack={slack:+.1f}]"

                print(f"  {user_s:<5} {f_type:<5}{rpt_s:<4} {f_pos:<5} "
                      f"{sat_s:<12} {lat_s:>8} {lon_s:>9} {alt_s:>7}  "
                      f"{acc_s:>10} {isl_s:>8} {vnf_s:>8} {e2e_s:>8}  {bgt_s}")

            # Blank line between users
            if u_id != sorted(asgn[sid].keys())[-1]:
                print()

        print("  " + "-" * (W - 2))

    # Delay summary table
    print()
    print("  Delay Summary")
    print("  " + "-" * 74)
    print(f"  {'Slice':>5} {'User':>4} {'Access ms':>10} "
          f"{'ISL ms':>8} {'VNF ms':>8} {'E2E ms':>8} {'Budget ms':>10} {'Status':>8}")
    print("  " + "-" * 74)
    for sl in slices:
        sid = sl["slice_id"]
        for u_id in sorted(dlay[sid].keys()):
            d      = dlay[sid][u_id]
            budget = d.get("budget_ms")
            ok     = d.get("budget_ok", True)
            verdict = "OK" if ok else "VIOLATED"
            print(f"  {sid:>5} {u_id:>4} "
                  f"{d.get('access_delay_ms', 0):>10.3f} "
                  f"{d.get('total_isl_ms', 0):>8.3f} "
                  f"{d.get('total_vnf_ms', 0):>8.3f} "
                  f"{d.get('e2e_total_ms', 0):>8.3f} "
                  f"{str(budget) if budget is not None else '---':>10} "
                  f"{verdict:>8}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Display placement detail from a *_placements.json file"
    )
    parser.add_argument("--file",     required=True, help="Path to *_placements.json")
    parser.add_argument("--method",   default=None,  help="Filter by method name")
    parser.add_argument("--epoch",    type=int, default=None, help="Single epoch to show")
    parser.add_argument("--instance", type=int, default=0,    help="Instance id (default 0)")
    args = parser.parse_args()

    epochs = [args.epoch] if args.epoch is not None else None
    display_placements(args.file, method=args.method,
                       epochs=epochs, instance_id=args.instance)


if __name__ == "__main__":
    _main()
