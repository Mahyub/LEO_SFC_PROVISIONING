# Cross-Slice Co-Location Risk-Aware SFC Provisioning in Multi-Slice LEO Satellite Networks

Source code for the paper **"Cross-Slice Co-Location Risk-Aware SFC Provisioning in Multi-Slice LEO Satellite Networks"** (IEEE GLOBECOM 2026- Under Review).

---

## Overview

This project addresses the problem of placing Virtual Network Functions (VNFs) and Service Function Chains (SFCs) across a Low-Earth Orbit (LEO) satellite constellation while jointly minimizing resource utilization, security co-location risk, and migration churn across orbital epochs.

The optimizer combines a Mixed-Integer Linear Program (MILP) with a Simulated Annealing (SA) warm-start to handle the dynamic topology that results from satellite motion. Two risk formulations are provided (exact and coarse), along with three baselines for comparison.

---

## Key Features

- **Risk-aware MILP** with two formulations:
  - *Exact*: p-variable co-location model, O(N^2 U^2 F I S) variables
  - *Coarse*: z/y-variable approximation, tractable at scale
- **Three-stage hybrid optimizer**: preprocessing -> SA warm-start -> CBC branch-and-bound
- **SA warm-start**: reduces solve time by ~23x after the first epoch via incumbent hint injection
- **Migration stabilization**: keep-indicator epigraph minimizes avoidable VNF migrations
- **Walker-Star topology**: 60 satellites, 4 orbital planes, 550 km altitude, 53 deg inclination
- **Three baselines**: B1 (capacity-only MILP), B2 (capacity + migration MILP), B3 (greedy heuristic)
- **9 publication-quality figures** reproducible from saved results JSON

---

## Project Structure

```
leo_sfc/
├── main.py                        # Entry point (all modes)
├── config/
│   ├── base_matlab.json           # Main experiment configuration
│   └── stress_large.json          # Large-scale stress scenario
├── src/
│   ├── milp.py                    # MILP formulation (proposed coarse + exact)
│   ├── sa.py                      # Simulated Annealing warm-start
│   ├── baselines.py               # B1, B2, B3 solvers
│   ├── preprocessing.py           # Normalization bounds and migration indicators
│   ├── metrics.py                 # Risk, utilization, delay, migration evaluators
│   ├── instance_generator.py      # Pure-Python topology generator
│   ├── instance_generator_matlab.py  # MATLAB-backed topology generator
│   ├── experiment.py              # Multi-instance experiment orchestrator
│   ├── visibility.py              # Satellite-ground visibility computation
│   ├── placement_logger.py        # Placement serialization
│   ├── user_visibility_logger.py  # User visibility logging
│   └── types.py                   # Shared dataclasses
├── analysis/
│   ├── figures.py                 # Figure generation (9 plots)
│   └── placement_report.py        # Per-hop placement/delay printer
├── tests/
│   └── test_all.py                # Unit test suite (10 tests)
├── data/
│   └── results/                   # Experiment output JSON files
└── matlab/                        # MATLAB topology scripts (optional)
```

---

## Installation

Python 3.9 or later is required.

```bash
pip install -r requirements.txt
```

The CBC solver is bundled with PuLP. No separate solver installation is needed.

**MATLAB topology bridge**: requires MATLAB R2022b or later with the Satellite Communications Toolbox. 

---

## Quick Start


### Reproduce paper figures from saved results

Requires the results file from the full experiment run:

```bash
python main.py --mode figures --results data/results/base_matlab_results.json
```


### Run the full experiment

Runs all instances, epochs, and methods and writes a results JSON:

```bash
python main.py --mode experiment --config config/base_matlab.json
```

### Print per-hop placement reports

Runs the solver for a few epochs and prints satellite assignments, function positions, and delay breakdowns:

```bash
python main.py --mode placements --config config/base_matlab.json --epochs 3
```

### Run the unit test suite

```bash
python main.py --mode test
# or directly:
pytest tests/test_all.py -v
```

---

## Configuration

The main scenario is defined in `config/base_matlab.json`. Key parameters:

| Parameter | Value | Description |
|---|---|---|
| `num_satellites` | 60 | Total satellites in the constellation |
| `orbital_planes` | 4 | Number of Walker-Star planes |
| `orbit_altitude_km` | 550.0 | Orbital altitude (km) |
| `orbit_inclination_deg` | 53.0 | Inclination (degrees) |
| `num_slices` | 5 | Number of network slices |
| `users_per_slice` | 10 | Users per slice |
| `num_epochs` | 15 | Number of time epochs |
| `num_instances` | 1 | Instances per scenario |
| `function_types` | FW, IDS, ENC, TM, SIEM | VNF types |
| `sfc_length_range` | [2, 4] | Min/max VNFs per chain |
| `e2e_budget_range_ms` | [75, 150] | End-to-end delay budgets (ms) |
| `omega_cap` / `omega_risk` / `omega_mig` | 0.3 / 0.5 / 0.2 | Objective weights |
| `sa_T0` / `sa_Tend` / `sa_iterations` | 1.0 / 0.01 / 50,000 | SA cooling parameters |
| `time_limit_s` | 300 | MILP time limit per epoch (s) |

---

## Methods

| Method | Description |
|---|---|
| `proposed_coarse` | Risk-aware MILP with coarse (z/y) risk model + SA warm-start |
| `proposed_exact` | Risk-aware MILP with exact (p-variable) risk model + SA warm-start |
| `B1` | Capacity-minimization MILP only (omega_risk = omega_mig = 0) |
| `B2` | Capacity + migration MILP (omega_risk = 0) |
| `B3` | Greedy nearest-feasible heuristic |

---

## Output Figures

`--mode figures` generates the following PDFs in `data/results/`:

| File | Content |
|---|---|
| `risk_bars.pdf` | Mean co-location risk per method with 95% CI |
| `resource_util.pdf` | CPU utilization (%) per method |
| `peak_vs_avg_util.pdf` | Peak vs. average CPU utilization per method |
| `migration_epochs.pdf` | Avoidable migrations per epoch |
| `bound_tightness.pdf` | Risk lower/upper bound tightness across epochs |
| `runtime_bars.pdf` | Mean solver runtime per method (warm-start epochs) |
| `runtime_epochs.pdf` | Solver runtime per epoch with cold-start annotation |
| `risk_epochs.pdf` | Co-location risk per epoch |
| `risk_resource_tradeoff.pdf` | Risk vs. utilization scatter per method |

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{leo_sfc_provisioning,
  title     = {Cross-Slice Co-Location Risk-Aware SFC Provisioning in Multi-Slice LEO Satellite Networks},
  author= {Mohammed Mahyoub, Wael Jaafar, Sami Muhaidat, and Halim Yanikomeroglu}
  booktitle = {Proc. IEEE GLOBECOM},
  year      = {2026}
}
```

---

## License

This repository is released for academic research use. See `LICENSE` for details.
