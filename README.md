# Oncostreams

Software for simulating the formation and dynamics of **oncostreams** (collective motion of glioma cells) based on active matter principles.

This project is a fork and extension of [`tumorsphere_culture`]([LINK_AL_REPO_ORIGINAL_SI_ES_PUBLICO](https://github.com/JeroFotinos/tumorsphere_culture)), adapted to include cell motility, self-propulsion, interactions and active-passive mixture dynamics.

<img width="1600" height="1200" alt="steady_state_10000_0_85_0005" src="https://github.com/user-attachments/assets/83f9a539-dbf6-413d-8b0c-b6bdd6363b5e" />


## Key Features

* **Active Matter Dynamics:** Simulation of self-propelled particles modeling distinct cell behaviors.
* **Heterogeneity:** Support for mixtures of active (motile) and passive (non-motile) cells.
* **Extended Physics:** Adds motility parameters, persistence, and multiple interactions relevant to glioma invasion and other interesting cases.

## Directory Structure

- `tumorsphere` contains the source code of the package;
- `tests` contains the tests for pytest to find;
- `library` contains scripts for processing the output files produced by simulations;
- `examples` contains the bash scripts used to lunch simulations in the cluster;
- `data` contains one directory for each set of simulations sent to the cluster, including the corresponding scripts, output data, and post-processing.


## Requirements

The package requires Python 3.10 or newer versions, since before 3.10, the `slots` option for `dataclass` was not available.


## Credits

This software is an extension of the `tumorsphere_culture` package originally developed for simulating 3D tumorsphere growth.

