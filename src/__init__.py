# src/__init__.py
"""

Package structure:
    src.types               — Shared dataclasses and type aliases
    src.instance_generator  — Topology, SFC, and risk parameter generation
    src.preprocessing       — pi computation and normalization bounds
    src.sa                  — Simulated Annealing warm-start
    src.milp                — MILP model (exact + coarse risk formulations)
    src.baselines           — B1, B2, B3 baseline methods
    src.metrics             — Risk^ex, CapUse, migration metrics
    src.experiment          — Orchestration and result collection
"""
