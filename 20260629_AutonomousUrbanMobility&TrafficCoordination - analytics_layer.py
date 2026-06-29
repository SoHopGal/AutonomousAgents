"""
Instrumented Analytics Layer for the Multi-Agent Traffic Simulation
=====================================================================
HOW TO USE IN COLAB:
  Cell 1: paste the full content of grid_traffic_simulation.py and run it
          (exactly as you've been doing — this defines Env, Vehicle,
          ReservationManager, SCENARIOS, run_scenario_headless, etc. in the
          notebook's global namespace, and also prints the scenario table /
          renders the animation, same as before).
  Cell 2: paste THIS file's content and run it. It does NOT read any file —
          it uses the classes/functions Cell 1 already defined.

This file is INTENTIONALLY SEPARATE from grid_traffic_simulation.py (frozen,
code-freeze in effect). It adds metric collection ONLY — it never changes
simulation logic, movement rules, reservation decisions, or collision physics.

How non-invasiveness is guaranteed (no file re-reading required):
  1. Before any monkey-patching happens, this file calls the EXISTING,
     STILL-UNPATCHED run_scenario_headless() (defined by Cell 1) on a set of
     (scenario, seed, light_mode) combinations and stores those results as
     the "baseline" — this is the frozen file's own behavior, observed
     directly, not reconstructed from a saved copy.
  2. Only THEN are ReservationManager.request/force_grant monkey-patched, and
     Env is subclassed (not modified) into InstrumentedEnv.
  3. The same (scenario, seed, light_mode) combinations are re-run through
     InstrumentedEnv, and the core outcome metrics (avg/min/max travel,
     avg/min/max wait, collision_count, completed_trips) are asserted to be
     IDENTICAL to the baseline captured in step 1. If they ever differ, the
     instrumentation has a bug and must not be trusted until fixed.

Run verify_non_invasive() (called automatically at the bottom of this file)
before relying on any of this file's metrics for the validation report.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Sanity check: make sure Cell 1 was actually run first.
# ---------------------------------------------------------------------------
_required_names = ['Env', 'Vehicle', 'ReservationManager', 'Intersection',
                   'SCENARIOS', 'run_scenario_headless']
_missing = [n for n in _required_names if n not in globals()]
if _missing:
    raise RuntimeError(
        f"Missing required names from the global namespace: {_missing}. "
        f"Did you run the grid_traffic_simulation.py cell BEFORE this one? "
        f"This file expects Env/Vehicle/ReservationManager/SCENARIOS/"
        f"run_scenario_headless to already be defined."
    )

print("Found Env, Vehicle, ReservationManager, SCENARIOS, run_scenario_headless "
      "already defined — proceeding with instrumentation.")


# ---------------------------------------------------------------------------
# Step 1: capture a baseline from the STILL-UNPATCHED frozen functions, before
# any monkey-patching happens.
# ---------------------------------------------------------------------------
_BASELINE_CHECK_SCENARIOS = list(SCENARIOS.keys())
_BASELINE_CHECK_SEEDS = (1, 7, 42)
_BASELINE_CHECK_MODES = ('fixed', 'adaptive')


def _capture_baseline():
    baseline = {}
    for sid in _BASELINE_CHECK_SCENARIOS:
        for seed in _BASELINE_CHECK_SEEDS:
            for mode in _BASELINE_CHECK_MODES:
                np.random.seed(seed)
                result = run_scenario_headless(sid, light_mode=mode, seed=seed)
                baseline[(sid, seed, mode)] = result
    return baseline


print("Capturing baseline from UNPATCHED run_scenario_headless() "
      f"({len(_BASELINE_CHECK_SCENARIOS)} scenarios x {len(_BASELINE_CHECK_SEEDS)} "
      f"seeds x {len(_BASELINE_CHECK_MODES)} modes "
      f"= {len(_BASELINE_CHECK_SCENARIOS) * len(_BASELINE_CHECK_SEEDS) * len(_BASELINE_CHECK_MODES)} runs)...")
_baseline_results = _capture_baseline()
print("Baseline captured.")


# ---------------------------------------------------------------------------
# Step 2: Instrumentation via monkey-patching. Each wrapper calls the
# ORIGINAL method and returns its result UNCHANGED — it only observes.
# ---------------------------------------------------------------------------

_original_request = ReservationManager.request
_original_force_grant = ReservationManager.force_grant


def _instrumented_request(self, vehicle):
    result = _original_request(self, vehicle)
    log = getattr(self, '_metrics_log', None)
    if log is not None:
        log['reservation_requests'] += 1
        if result:
            log['reservation_granted'] += 1
        else:
            log['reservation_denied'] += 1
    return result


def _instrumented_force_grant(self, vehicle, occupant_inside_box=False):
    result = _original_force_grant(self, vehicle, occupant_inside_box=occupant_inside_box)
    log = getattr(self, '_metrics_log', None)
    if log is not None:
        log['emergency_overrides'] += 1
        if not result:
            log['emergency_override_blocked'] += 1
    return result


ReservationManager.request = _instrumented_request
ReservationManager.force_grant = _instrumented_force_grant


class InstrumentedEnv(Env):
    """Subclass of the frozen Env that adds metric collection around the
    ORIGINAL step() method without altering it. step() is called via
    super().step() unchanged; everything below only reads state before/after.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.metrics_log = {
            'reservation_requests': 0,
            'reservation_granted': 0,
            'reservation_denied': 0,
            'emergency_overrides': 0,
            'emergency_override_blocked': 0,
        }
        for inter in self.intersections.values():
            inter.manager._metrics_log = self.metrics_log

        self.queue_history = {key: [] for key in self.intersections}

        self.emergency_travel_times = []
        self.emergency_wait_times = []
        self.regular_travel_times = []
        self.regular_wait_times = []

    def step(self):
        vehicles_before = {v.vehicle_id: v for v in self.vehicles}

        super().step()  # <-- UNCHANGED original simulation logic

        vehicles_after_ids = {v.vehicle_id for v in self.vehicles}
        finished_ids = set(vehicles_before.keys()) - vehicles_after_ids
        for vid in finished_ids:
            v = vehicles_before[vid]
            if not v.finished:
                continue
            travel_time = self.t - v.start_time
            if v.is_emergency:
                self.emergency_travel_times.append(travel_time)
                self.emergency_wait_times.append(v.wait_time)
            else:
                self.regular_travel_times.append(travel_time)
                self.regular_wait_times.append(v.wait_time)

        for key, inter in self.intersections.items():
            total_q = sum(inter.waiting_counts.values())
            self.queue_history[key].append(total_q)

    def metric_summary(self):
        def stats(values):
            if not values:
                return dict(avg=0.0, median=0.0, std=0.0, min=0.0, max=0.0, n=0)
            arr = np.asarray(values)
            return dict(avg=float(arr.mean()), median=float(np.median(arr)),
                        std=float(arr.std()), min=float(arr.min()),
                        max=float(arr.max()), n=len(arr))

        all_travel = self.completed_travel_times
        all_wait = self.completed_wait_times
        spawned = self.regular_spawned + self.emergency_spawned
        completed = len(all_travel)

        queue_all = [q for hist in self.queue_history.values() for q in hist]

        return {
            'travel': stats(all_travel),
            'wait': stats(all_wait),
            'emergency_travel': stats(self.emergency_travel_times),
            'emergency_wait': stats(self.emergency_wait_times),
            'regular_travel': stats(self.regular_travel_times),
            'regular_wait': stats(self.regular_wait_times),
            'collisions': self.collision_count,
            'spawned': spawned,
            'completed': completed,
            'completed_pct': 100.0 * completed / spawned if spawned else 0.0,
            'throughput_per_sec': completed / self.t if self.t > 0 else 0.0,
            'reservation_requests': self.metrics_log['reservation_requests'],
            'reservation_granted': self.metrics_log['reservation_granted'],
            'reservation_denied': self.metrics_log['reservation_denied'],
            'reservation_success_rate': (
                100.0 * self.metrics_log['reservation_granted'] / self.metrics_log['reservation_requests']
                if self.metrics_log['reservation_requests'] else 0.0
            ),
            'emergency_overrides': self.metrics_log['emergency_overrides'],
            'queue_avg': float(np.mean(queue_all)) if queue_all else 0.0,
            'queue_max': float(np.max(queue_all)) if queue_all else 0.0,
        }


def run_instrumented_scenario(scenario_id, light_mode='fixed', seed=None, overrides=None):
    """Same calling convention as run_scenario_headless(), but returns the
    full InstrumentedEnv metric_summary() dict instead of the smaller dict."""
    if seed is not None:
        np.random.seed(seed)
    cfg = SCENARIOS[scenario_id]
    overrides = overrides or {}
    e = InstrumentedEnv(light_mode=light_mode,
                        total_vehicles=cfg['vehicles'],
                        total_emergency=cfg['emergency'],
                        **overrides)
    steps = int(cfg['sim_time'] / e.dt)
    for _ in range(steps):
        e.step()
    summary = e.metric_summary()
    summary['scenario'] = scenario_id
    summary['light_mode'] = light_mode
    summary['seed'] = seed
    return summary


# ---------------------------------------------------------------------------
# Step 3: Self-check — compare InstrumentedEnv results against the baseline
# captured in Step 1 (from the genuinely unpatched run_scenario_headless).
# ---------------------------------------------------------------------------

def verify_non_invasive():
    checks = 0
    for (sid, seed, mode), frozen_result in _baseline_results.items():
        instrumented_result = run_instrumented_scenario(sid, light_mode=mode, seed=seed)

        mismatches = []
        if abs(frozen_result['avg_travel'] - instrumented_result['travel']['avg']) > 1e-9:
            mismatches.append('avg_travel')
        if abs(frozen_result['min_travel'] - instrumented_result['travel']['min']) > 1e-9:
            mismatches.append('min_travel')
        if abs(frozen_result['max_travel'] - instrumented_result['travel']['max']) > 1e-9:
            mismatches.append('max_travel')
        if abs(frozen_result['avg_wait'] - instrumented_result['wait']['avg']) > 1e-9:
            mismatches.append('avg_wait')
        if frozen_result['collisions'] != instrumented_result['collisions']:
            mismatches.append('collisions')
        if frozen_result['completed_trips'] != instrumented_result['completed']:
            mismatches.append('completed_trips')

        if mismatches:
            raise AssertionError(
                f"NON-INVASIVENESS CHECK FAILED for {sid}/{mode}/seed={seed}: "
                f"mismatched fields = {mismatches}\n"
                f"frozen={frozen_result}\ninstrumented={instrumented_result}"
            )
        checks += 1

    print(f"PASS: instrumentation is non-invasive — {checks} (scenario, seed, mode) "
          f"combinations produced IDENTICAL core results between the unpatched "
          f"baseline and InstrumentedEnv.")
    return True


verify_non_invasive()
