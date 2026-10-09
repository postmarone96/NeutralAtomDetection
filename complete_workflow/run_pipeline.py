"""Profile -> sensitivity -> optimize: the fixed-point widths the RTL is built at.

    python run_pipeline.py                  all three stages
    python run_pipeline.py --from optimize  pick up from the width search
    python run_pipeline.py --only profile   one stage
    python run_pipeline.py --no-resume      start over, ignoring state.json

Fidelity is scored by fixed_point.py, the integer model the RTL has to match.
"""

import argparse
import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
if str(here) not in sys.path:
    sys.path.insert(0, str(here))

import common
import model as model_module
import optimize
import profile_stage
import quantize
import sensitivity
import widths as widths_module

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

methods = [model_module.lightweight_cnn_site_neighbor_method]

saturation_margin_bits = 0
selected_exposures = [0.01]
weight_widths = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
optimize_max_eval = 150
acceptance_pp = 0.05

resume = False

results_root = here / "results"
stages = ("profile", "sensitivity", "optimize")
default_last = "optimize"


# ---------------------------------------------------------------------------
# Where a run keeps its state
# ---------------------------------------------------------------------------

def label_for(method):
    """Short directory name for this method's results."""
    return "gaussian" if method.endswith("_gaussian") else "inj"


def load_state(results_dir):
    path = results_dir / "state.json"
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def save_state(results_dir, state):
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "state.json").open("w") as handle:
        json.dump(state, handle, indent=2)


def stage_done(state, stage, key=None):
    entry = state.get(stage)
    if entry is None:
        return False
    if key is None:
        return bool(entry)
    return bool(entry.get(str(key))) if isinstance(entry, dict) else False


def mark_done(state, stage, key=None):
    if key is None:
        state[stage] = True
    else:
        state.setdefault(stage, {})[str(key)] = True


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def configure(method, results_dir):
    """Push the inputs into each stage before anything runs."""
    for module in (profile_stage, quantize, sensitivity, optimize):
        module.set_results_dir(results_dir)

    profile_stage.saturation_margin_bits = saturation_margin_bits
    profile_stage.target_method = method
    profile_stage.selected_exposures = tuple(selected_exposures)

    quantize.selected_method = method

    sensitivity.selected_exposures = tuple(selected_exposures)
    sensitivity.weight_widths = tuple(weight_widths)
    sensitivity.resume = resume

    optimize.set_method(method)
    optimize.max_evaluations = optimize_max_eval

    widths_module.acceptance_pp = acceptance_pp


def stage_profile(results_dir, state, force=False):
    profile_stage.run()
    mark_done(state, "profile")


def stage_sensitivity(results_dir, state, force=False):
    sensitivity.run()
    mark_done(state, "sensitivity")


def stage_optimize(results_dir, state, force=False):
    for exposure in sorted(selected_exposures):
        if resume and not force and stage_done(state, "optimize", exposure):
            print(f"  {1000.0 * exposure:g} ms already searched")
            continue
        optimize.run(exposure)
        mark_done(state, "optimize", exposure)


stage_functions = {
    "profile": stage_profile,
    "sensitivity": stage_sensitivity,
    "optimize": stage_optimize,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", choices=stages, default=None)
    parser.add_argument("--until", dest="last", choices=stages, default=None)
    parser.add_argument("--only", choices=stages)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    global resume
    if args.no_resume:
        resume = False

    if args.only:
        wanted, force_from = (args.only,), 0
    else:
        end = stages.index(args.last or default_last) + 1
        if args.start:
            force_from = stages.index(args.start)
            if args.last is None:
                end = max(end, force_from + 1)
        else:
            force_from = len(stages)
        wanted = stages[:end]

    for method in methods:
        label = label_for(method)
        results_dir = results_root / label
        results_dir.mkdir(parents=True, exist_ok=True)
        state = load_state(results_dir) if resume else {}

        print("=" * 78)
        print(method)
        print(f"neighbours: crop means   -> {results_dir}")
        print("=" * 78)

        configure(method, results_dir)

        for stage in wanted:
            forced = stages.index(stage) >= force_from
            per_exposure = stage == "optimize"
            if resume and not forced and not per_exposure and stage_done(state, stage):
                print(f"[{stage}] already done")
                continue
            print(f"[{stage}]")
            stage_functions[stage](results_dir, state, forced)
            save_state(results_dir, state)

        print()


if __name__ == "__main__":
    main()
