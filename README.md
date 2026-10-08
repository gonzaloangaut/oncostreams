# Oncostreams

Research software for simulating **collective migration in glioma cells** using an active-matter model. The project studies how cell shape, motility, and interactions contribute to the emergence of collective motion.

**Status:** active research and development. Model definition, numerical validation, and the current analysis workflow are documented in [oncostreams_model_analysis](https://github.com/gonzaloangaut/oncostreams_model_analysis).

This project extends [tumorsphere_culture](https://github.com/JeroFotinos/tumorsphere_culture), originally developed by Jerónimo Fotinós for simulating tumorsphere growth. The Python package retains the original `tumorsphere` import name.

<img width="1600" height="1200" alt="Example configuration from an oncostreams simulation" src="https://github.com/user-attachments/assets/83f9a539-dbf6-413d-8b0c-b6bdd6363b5e" />

## Research focus

The current work uses a two-dimensional model with periodic boundaries to study the coupling between cell elongation and motility, collective order, clustering, and transitions between dynamical regimes.

## Main capabilities

- **Cell dynamics:** cell shape changes, self-propulsion, and cell-cell interactions.
- **Simulation management:** multiple initial conditions, parameter combinations, independent realizations, and parallel execution.
- **Neighbor search:** a spatial hash grid for local interaction calculations.
- **Checkpointing:** save and resume simulation runs.
- **Observables:** positions and aspect ratios, global and local order parameters, motion, clusters, deformation, and overlap diagnostics.
- **Visualization outputs:** files for inspecting cell configurations with OVITO.

## Code guide

| Path | Purpose |
| --- | --- |
| [`tumorsphere/core/cells.py`](tumorsphere/core/cells.py) | Cell state, shape, and motility |
| [`tumorsphere/core/culture.py`](tumorsphere/core/culture.py) | Cell population and dynamical updates |
| [`tumorsphere/core/forces.py`](tumorsphere/core/forces.py) | Interaction force definitions |
| [`tumorsphere/core/spatial_hash_grid.py`](tumorsphere/core/spatial_hash_grid.py) | Spatial indexing and neighbor search |
| [`tumorsphere/core/simulation.py`](tumorsphere/core/simulation.py) | Simulation configuration, parallel runs, and checkpoints |
| [`tumorsphere/core/output.py`](tumorsphere/core/output.py) | Simulation outputs and recorded observables |
| [`tumorsphere/library/`](tumorsphere/library/) | Data-processing and visualization utilities |
| [`tumorsphere/cli.py`](tumorsphere/cli.py) | Command-line interface inherited from the original package |

## Installation

The package declares **Python 3.10 or newer** in [`pyproject.toml`](pyproject.toml). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The package is installed as `oncostreams` and imported as `tumorsphere`. Simulation configuration and available output options are defined in [`Simulation`](tumorsphere/core/simulation.py).

## Model documentation and analysis

The companion repository, [oncostreams_model_analysis](https://github.com/gonzaloangaut/oncostreams_model_analysis), contains:

- the model definition, units, and parameter conventions;
- the neighbor criterion and force-regularization studies;
- integration-time-step validation;
- scripts for processing simulation outputs.

Numerical protocols are still being validated. The companion repository records the current analysis status and planned studies.

## Credits and license

Original tumorsphere software: **Jerónimo Fotinós**. Oncostreams extensions and research development: **Gonzalo Angaut**.

See [LICENSE](LICENSE) for the software license.
