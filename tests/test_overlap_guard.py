"""Guard overlap diagnostics without modifying physical dynamics or output policy."""

import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from tumorsphere.core.culture import Culture
from tumorsphere.core.forces import Anisotropic_Grosmann
from tumorsphere.core.output import (
    OutputDemux,
    TumorsphereOutput,
    create_output_demux,
)
from tumorsphere.core.simulation import Simulation


class CustomOverlapOutput(TumorsphereOutput):
    """Custom output used to test overlap diagnostic detection."""

    def should_record_overlap_parameters(self, tic, final_tic):
        return True

    def begin_culture(self, *args, **kwargs):
        pass

    def record_cell(self, *args, **kwargs):
        pass

    def record_culture_state(self, *args, **kwargs):
        pass

    def record_deactivation(self, *args, **kwargs):
        pass

    def record_final_state(self, *args, **kwargs):
        pass

    def record_stemness(self, *args, **kwargs):
        pass


def test_demux_detects_overlap_collectors(tmp_path):
    none = create_output_demux("test", [], str(tmp_path))
    regular = create_output_demux("test", ["dat_pos_ar"], str(tmp_path))
    overlaps = create_output_demux("test", ["dat_overlap_par"], str(tmp_path))
    assert none.collect_overlap_statistics is False
    assert regular.collect_overlap_statistics is False
    assert overlaps.collect_overlap_statistics is True
    assert (
        OutputDemux("test", [CustomOverlapOutput()]).collect_overlap_statistics
        is True
    )


@pytest.mark.parametrize("output_enabled", [False, True])
def test_interaction_forwards_cached_policy(output_enabled, tmp_path):
    output = create_output_demux(
        "test", ["dat_overlap_par"] if output_enabled else [], str(tmp_path)
    )
    seen = []

    def spy(*, cell_index, update_overlap_diagnostic):
        seen.append(update_overlap_diagnostic)
        return np.empty(0, dtype=int)

    cell = SimpleNamespace(
        neighbors_relative_pos={},
        neighbors_overlap={},
        neighbors_normalized_overlap={},
    )
    force = SimpleNamespace(
        calculate_interaction=lambda *args: (np.zeros(3), 0.0)
    )
    fake = SimpleNamespace(
        backend="numpy",
        output=output,
        _get_significant_neighbors=spy,
        cells=[cell],
        cell_phies=np.array([0.0]),
        force=force,
        cell_area=np.pi,
        nematic_tensors=np.zeros((1, 3, 3)),
    )
    Culture.interaction(fake, 0, 0.005)
    assert seen == [output_enabled]


def _simulation(steps, backend):
    """Build a deterministic fixed-N configuration for output-policy tests."""
    return Simulation(
        forces=[
            Anisotropic_Grosmann(
                kRep=2.5,
                bExp=3.0,
                lambda_core=1.0,
                shrinking=True,
                noise_eta=None,
                d_phi=None,
            )
        ],
        rng_seed=13579,
        num_of_realizations=1,
        num_of_steps_per_realization=steps,
        initial_number_of_cells=[48],
        initial_fraction_elongated=[0.5],
        initial_density=[0.84],
        cell_max_def_attempts=1,
        reproduction=False,
        movement=True,
        deformation=True,
        stabilization_time=0,
        overlap_threshold_ratio=np.exp(-1),
        delta_t=0.005,
        deformation_attempt_period=0.05,
        initial_aspect_ratio=1.0,
        aspect_ratio_max=2.7,
        cell_speed_max=1.0,
        trabajo_final=True,
        spatial_dimension=3,
        backend=backend,
    )


def _run(output_dir, steps, with_overlap, backend):
    """Run one case and reload the saved physical state from checkpoint."""
    sim = _simulation(steps, backend)
    sim.simulate_parallel(
        sql=False,
        number_of_processes=1,
        output_dir=str(output_dir),
        dat_overlap_par=with_overlap,
        save_step_dat_overlap_par=2,
    )
    checkpoints = list((output_dir / "checkpoints").glob("*.pkl"))
    assert len(checkpoints) == 1
    with checkpoints[0].open("rb") as file:
        culture, tic, rng_state = pickle.load(file)
    assert tic == steps
    return culture


def _assert_same_state(a, b):
    """Assert bitwise-equivalent positions, shapes, angles and RNG state."""
    np.testing.assert_array_equal(a.cell_positions, b.cell_positions)
    np.testing.assert_array_equal(a.cell_phies, b.cell_phies)
    np.testing.assert_array_equal(
        [c.aspect_ratio for c in a.cells],
        [c.aspect_ratio for c in b.cells],
    )
    assert a.rng.bit_generator.state == b.rng.bit_generator.state


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize(
    "initial_output,new_output",
    [
        (False, True),
        (True, False),
        (False, False),
        (True, True),
    ],
)
def test_checkpoint_output_change(
    backend, initial_output, new_output, tmp_path
):
    if backend == "numba":
        pytest.importorskip("numba")
    partial = tmp_path / "partial"
    previous = _run(partial, 4, initial_output, backend)
    assert previous.output.collect_overlap_statistics == initial_output
    if not initial_output:
        assert previous.normalized_overlap_count_interval == 0
    resumed = _run(partial, 8, new_output, backend)
    assert resumed.output.collect_overlap_statistics == new_output
    if not new_output:
        assert resumed.normalized_overlap_count_interval == 0
    uninterrupted = _run(tmp_path / "full", 8, new_output, backend)
    _assert_same_state(resumed, uninterrupted)
    if new_output:
        assert list((partial / "dat_overlap_parameters").glob("*.dat"))
