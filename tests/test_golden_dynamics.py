"""Frozen fixed-N dynamics regressions independent of retired backends.

The committed ``tests/data/fixed_n_dynamics_v1.npz`` was generated and
verified before backend retirement. Do not regenerate the fixture in normal
testing; changes require separate scientific review on the historical branch.
"""
from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pytest

from tests._helpers import (
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


def _culture(settings):
    """Construct the production configuration recorded in the fixture."""
    return fixed_n_grosmann_culture(
        backend="numba",
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




@pytest.fixture(scope="module")
def golden():
    if not FIXTURE.exists():
        pytest.fail(
            f"Missing frozen regression fixture: {FIXTURE}. "
            "Restore the committed fixture from the pre-cleanup branch."
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
