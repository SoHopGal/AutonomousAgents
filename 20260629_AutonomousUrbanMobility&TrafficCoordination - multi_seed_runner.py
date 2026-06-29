"""
Multi-Seed Experiment Runner (Cell 3)
=====================================================================
HOW TO USE IN COLAB (run in this exact order, in the SAME kernel session,
without a long idle gap between cells — Colab disconnects idle runtimes,
which wipes all previously-defined variables):
  Cell 1: grid_traffic_simulation.py   (defines Env, Vehicle, SCENARIOS, ...)
  Cell 2: analytics_layer.py           (defines InstrumentedEnv, run_instrumented_scenario, ...)
  Cell 3: THIS file                    (defines run_multi_seed, aggregate_runs, ...)

This file implements the seed-repetition protocol requested:
    VS-01 -> seed1 -> seed2 -> ... -> seedN -> average, std, median
for every scenario, and for both light_mode='fixed' and 'adaptive'.

It depends only on run_instrumented_scenario() from Cell 2 — no new
simulation logic, only repeated calls + statistical aggregation.
"""

import numpy as np

_required_names = ['run_instrumented_scenario', 'SCENARIOS']
_missing = [n for n in _required_names if n not in globals()]
if _missing:
    raise RuntimeError(
        f"Missing required names from the global namespace: {_missing}. "
        f"Did you run grid_traffic_simulation.py (Cell 1) and analytics_layer.py "
        f"(Cell 2) BEFORE this one, in the same session?"
    )

print("Found run_instrumented_scenario, SCENARIOS — proceeding.")

DEFAULT_SEEDS = list(range(1, 11))  # seed1..seed10, as specified


# ---------------------------------------------------------------------------
# Flatten a metric_summary() dict (nested, e.g. summary['travel']['avg']) into
# a flat dict of scalar values, so aggregation can operate generically over
# any metric name without hardcoding each one by name.
# ---------------------------------------------------------------------------
def _flatten_summary(summary):
    flat = {}
    for key, val in summary.items():
        if isinstance(val, dict):
            for subkey, subval in val.items():
                flat[f'{key}.{subkey}'] = subval
        else:
            flat[key] = val
    return flat


def run_multi_seed(scenario_id, light_mode='fixed', seeds=None, overrides=None):
    """Runs the same scenario across multiple seeds, returns the list of raw
    per-seed flattened metric dicts (one per seed)."""
    seeds = seeds if seeds is not None else DEFAULT_SEEDS
    runs = []
    for seed in seeds:
        summary = run_instrumented_scenario(scenario_id, light_mode=light_mode,
                                            seed=seed, overrides=overrides)
        runs.append(_flatten_summary(summary))
    return runs


def aggregate_runs(runs):
    """Given a list of flattened per-seed metric dicts (same keys in each),
    returns a dict: {metric_name: {'mean':.., 'std':.., 'median':.., 'n':..}}.

    Only numeric (int/float) fields are aggregated; non-numeric fields
    (e.g. 'scenario', 'light_mode') are skipped automatically.
    """
    if not runs:
        return {}

    numeric_keys = [k for k, v in runs[0].items() if isinstance(v, (int, float))]
    agg = {}
    for key in numeric_keys:
        values = np.array([r[key] for r in runs], dtype=float)
        agg[key] = {
            'mean': float(values.mean()),
            'std': float(values.std()),
            'median': float(np.median(values)),
            'min': float(values.min()),
            'max': float(values.max()),
            'n': len(values),
        }
    return agg


def run_and_aggregate(scenario_id, light_mode='fixed', seeds=None, overrides=None):
    """Convenience wrapper: run_multi_seed + aggregate_runs in one call."""
    runs = run_multi_seed(scenario_id, light_mode=light_mode, seeds=seeds, overrides=overrides)
    agg = aggregate_runs(runs)
    return runs, agg


# ---------------------------------------------------------------------------
# Self-check: verify aggregation math against a manual hand-computed example,
# and verify that two different seeds of the SAME scenario actually produce
# DIFFERENT raw results (proving randomness is genuinely seeded per-run, not
# silently reusing a cached/identical state).
# ---------------------------------------------------------------------------
def _verify_aggregation_math():
    # Manual ground truth: 3 fake runs with known wait.avg values.
    fake_runs = [
        {'wait.avg': 10.0, 'scenario': 'X'},
        {'wait.avg': 20.0, 'scenario': 'X'},
        {'wait.avg': 30.0, 'scenario': 'X'},
    ]
    agg = aggregate_runs(fake_runs)
    expected_mean = 20.0
    expected_median = 20.0
    expected_std = np.std([10.0, 20.0, 30.0])  # population std, ddof=0, matches np.std default
    assert abs(agg['wait.avg']['mean'] - expected_mean) < 1e-9, "mean mismatch"
    assert abs(agg['wait.avg']['median'] - expected_median) < 1e-9, "median mismatch"
    assert abs(agg['wait.avg']['std'] - expected_std) < 1e-9, "std mismatch"
    assert 'scenario' not in agg, "non-numeric field should have been skipped"
    print("PASS: aggregate_runs() math matches manual hand-computed values "
          f"(mean={agg['wait.avg']['mean']}, median={agg['wait.avg']['median']}, "
          f"std={agg['wait.avg']['std']:.4f}).")


def _verify_seed_variation():
    runs = run_multi_seed('VS-01', light_mode='fixed', seeds=[1, 2, 3])
    travel_avgs = [r['travel.avg'] for r in runs]
    assert len(set(travel_avgs)) > 1, (
        f"All 3 seeds produced IDENTICAL travel.avg ({travel_avgs}) — "
        f"this suggests seeding is not actually varying simulation randomness."
    )
    print(f"PASS: seeds 1,2,3 on VS-01 produced different travel.avg values "
          f"{[round(t, 2) for t in travel_avgs]} — confirms randomness is genuinely seeded per-run.")


_verify_aggregation_math()
_verify_seed_variation()
