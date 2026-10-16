"""Frozen fixed-N dynamics regressions independent of retired fast backends.

Generate the fixture ONCE, before removing the old backends:

    python -m tests.test_golden_dynamics --write --verify-legacy
    python -m pytest -q tests/test_golden_dynamics.py

Commit the resulting ``tests/data/fixed_n_dynamics_v1.npz``. Pytest never
regenerates it. Replacing a fixture requires explicit scientific review.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests._helpers import (
    assert_cultures_bitwise_equal,
    fixed_n_grosmann_culture,
    run_steps,
)

FIXTURE = Path(__file__).parent / "data" / "fixed_n_dynamics_v1.npz"
STEPS = (1, 6, 17, 32)

# Fixed, deliberately small simulations: different noise channels, densities,
# deformation attempts, and spatial grid dimensions. These exercise the full
# production combination, not only isolated kernels.
CASES = {
    "dense_2d_deterministic": dict(seed=17, dimension=2, rho=0.85,
                                   attempts=1, eta=0.0, d_phi=None),
    "dense_3d_translational": dict(seed=29, dimension=3, rho=0.84,
                                   attempts=1, eta=0.033, d_phi=None),
    "dense_2d_rotational": dict(seed=42, dimension=2, rho=0.90,
                                attempts=5, eta=None, d_phi=0.13),
    "sparse_3d_both_noise": dict(seed=23, dimension=3, rho=0.22,
                                 attempts=10, eta=0.033, d_phi=0.13),
    "dense_3d_deterministic_many_attempts": dict(
        seed=37, dimension=3, rho=0.85,
        attempts=10, eta=0.0, d_phi=None,
    ),
}

ARRAYS = (
    "cell_positions", "cell_phies", "nematic_tensors",
    "cell_instantaneous_velocities",
)
CELL_FIELDS = (
    "aspect_ratio", "anisotropy", "squared_diagonal",
    "is_round", "shrink",
)
OVERLAP_FIELDS = (
    "max_normalized_overlap_interval",
    "normalized_overlap_sum_interval",
    "normalized_overlap_count_interval",
)


def _culture(settings, *, legacy=False):
    """The legacy path exists only while generating/verifying the fixture."""
    return fixed_n_grosmann_culture(
        interaction_backend=(
            "batch_cached_numba" if legacy else "batch_cached_noise_numba"
        ),
        elongation_backend=(
            "fast_numba" if legacy else "grouped_early_numba"
        ),
        seed=settings["seed"], n=64, rho=settings["rho"],
        dimension=settings["dimension"],
        eta=settings["eta"], d_phi=settings["d_phi"],
        attempts=settings["attempts"], deformation_period=0.005,
        output_dir=None,
    )


def _state(culture):
    """Copy continuous variables and extract discrete bookkeeping separately."""
    arrays = {
        attr: np.array(getattr(culture, attr), copy=True)
        for attr in ARRAYS
    }
    arrays.update({
        f"cell_{attr}": np.asarray(
            [getattr(cell, attr) for cell in culture.cells]
        )
        for attr in CELL_FIELDS
    })
    # The bucket order is intentionally significant to future trajectories.
    metadata = {
        "rng_state": repr(culture.rng.bit_generator.state),
        "deformation_events": dict(culture.deformation_event_counts),
        "ordered_grid": [
            [key.hex(), list(indices)]
            for key, indices in culture.grid.hash_table.items()
        ],
        "overlap": {
            attr: float(getattr(culture, attr))
            for attr in OVERLAP_FIELDS
        },
    }
    return arrays, metadata


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _json_numpy_scalar(value):
    """Serialize NumPy scalar values in the frozen JSON manifest."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot JSON serialize {type(value).__name__}")


def write_fixture(*, verify_legacy=False, overwrite=False):
    if FIXTURE.exists() and not overwrite:
        raise SystemExit(
            f"Refusing to replace existing reference {FIXTURE}. "
            "Use --overwrite only after reviewing the change."
        )

    import numba

    manifest = {
        "schema": 1,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "platform": platform.machine(),
        "steps": list(STEPS),
        "cases": CASES,
        "legacy_verified": bool(verify_legacy),
        "snapshots": {},
    }
    payload = {}
    for name, settings in CASES.items():
        current = _culture(settings)
        old = _culture(settings, legacy=True) if verify_legacy else None
        for step in range(1, STEPS[-1] + 1):
            run_steps(current, step - 1, step)
            if old is not None:
                run_steps(old, step - 1, step)
                assert_cultures_bitwise_equal(current, old)
            if step not in STEPS:
                continue
            arrays, metadata = _state(current)
            label = f"{name}_step_{step}"
            for attr, array in arrays.items():
                payload[f"{label}__{attr}"] = array
            manifest["snapshots"][label] = metadata
        print(f"Captured {name} (legacy comparison: {bool(verify_legacy)})")

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    payload["manifest"] = np.asarray(
        json.dumps(manifest, sort_keys=True, default=_json_numpy_scalar)
    )
    np.savez_compressed(FIXTURE, **payload)
    print(f"Saved {FIXTURE} from {manifest['git_commit']}")


def test_numpy_scalars_can_be_serialized_in_manifest():
    """Grid indices and statistics may use NumPy scalar types."""
    values = {
        "index": np.int64(42),
        "count": np.uint32(3),
        "overlap": np.float32(0.25),
        "flag": np.bool_(True),
    }
    encoded = json.dumps(values, sort_keys=True, default=_json_numpy_scalar)
    assert json.loads(encoded) == {
        "index": 42, "count": 3, "overlap": 0.25, "flag": True,
    }


@pytest.fixture(scope="module")
def golden():
    if not FIXTURE.exists():
        pytest.fail(
            f"Missing frozen regression fixture: {FIXTURE}. "
            "Generate it before removing the legacy backends with "
            "`python -m tests.test_golden_dynamics --write --verify-legacy`."
        )
    with np.load(FIXTURE, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest"].item()))
        assert manifest["schema"] == 1
        assert manifest["cases"] == CASES
        assert manifest["steps"] == list(STEPS)
        assert manifest["legacy_verified"], (
            "Golden fixture must be verified against the retiring "
            "implementations before removing them."
        )
        # Copy arrays out before the archive is closed.
        data = {key: archive[key].copy() for key in archive.files if key != "manifest"}
    return manifest, data


@pytest.mark.parametrize("case_name", tuple(CASES))
def test_frozen_dynamics(case_name, golden):
    """Catch changes in trajectories, RNG, deformations, and ordered grid."""
    manifest, arrays = golden
    settings = CASES[case_name]
    culture = _culture(settings)
    import numba

    exact_environment = (
        manifest["numba"] == numba.__version__
        and manifest["python"] == platform.python_version()
        and manifest["numpy"] == np.__version__
        and manifest["platform"] == platform.machine()
    )
    for step in range(1, STEPS[-1] + 1):
        run_steps(culture, step - 1, step)
        if step not in STEPS:
            continue
        label = f"{case_name}_step_{step}"
        current, meta = _state(culture)
        expected = manifest["snapshots"][label]
        for attr, actual in current.items():
            target = arrays[f"{label}__{attr}"]
            assert actual.shape == target.shape and actual.dtype == target.dtype, attr
            if np.issubdtype(actual.dtype, np.floating):
                if exact_environment:
                    np.testing.assert_array_equal(actual, target,
                                                  err_msg=f"{label}: {attr}")
                else:
                    np.testing.assert_allclose(
                        actual, target, rtol=1e-12, atol=1e-12,
                        err_msg=f"{label}: {attr}",
                    )
            else:
                np.testing.assert_array_equal(actual, target,
                                              err_msg=f"{label}: {attr}")
        assert meta["rng_state"] == expected["rng_state"], label
        assert meta["ordered_grid"] == expected["ordered_grid"], label
        for key, value in expected["deformation_events"].items():
            got = meta["deformation_events"][key]
            if key.startswith("max_"):
                assert np.isclose(got, value, rtol=1e-12, atol=1e-12), (label, key)
            else:
                assert got == value, (label, key)
        for key, value in expected["overlap"].items():
            got = meta["overlap"][key]
            if key == "normalized_overlap_count_interval":
                assert got == value, (label, key)
            else:
                assert np.isclose(got, value, rtol=1e-12, atol=1e-12), (label, key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="Explicitly create a frozen regression fixture.")
    parser.add_argument("--verify-legacy", action="store_true",
                        help="Compare every generated step with old backends.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly replace an existing golden fixture.")
    args = parser.parse_args()
    if not args.write:
        parser.error("Pass --write to create the fixture; pytest only reads it.")
    if not args.verify_legacy:
        parser.error("First fixture must be generated with --verify-legacy.")
    write_fixture(verify_legacy=args.verify_legacy, overwrite=args.overwrite)
