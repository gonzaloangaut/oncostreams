"""
Module containing the Simulation class.

Classes:
    - Simulation: Class that manages cultures of cells with different
    parameter combinations, for a given number of realizations per said
    combination.
"""

import multiprocessing as mp
from typing import List, Tuple, Optional, Union

import numpy as np
import os
import pickle

from tumorsphere.core.culture import Culture
from tumorsphere.core.output import create_output_demux
from tumorsphere.core.spatial_hash_grid import SpatialHashGrid
from tumorsphere.core.forces import Force


class Simulation:
    """Class for simulating multiple `Culture` objects.

    Parameters
    ----------
    first_cell_is_stem : bool, optional
        Whether the first cell of each `Culture` object should be a stem cell.
        If set to `False`, the first cell of the cultures will be a
        differentiated one. Default is `True` (because tumorspheres are CSC
        seeded cultures).
    prob_stem : list of floats, optional
        The probability that a stem cell will self-replicate. Defaults to 0.36
        for being the value measured by Benítez et al. (BMC Cancer, (2021),
        1-11, 21(1))for the experiment of Wang et al. (Oncology Letters,
        (2016), 1355-1360, 12(2)) on a hard substrate.
    prob_diff : list of floats, optional
        The probability that a stem cell will yield a differentiated cell and
        then lose its stemness, effectively yielding two differentiated cells.
        Defaults to 0 (because our intention was to see if percolation occurs,
        and if it doesn't happen at prob_diff = 0, it will never happen).
    num_of_realizations : int, optional
        Number of `Culture` objects to simulate for each combination of
        `prob_stem` and `prob_diff`. Default is `4`.
    num_of_steps_per_realization : int, optional
        Number of simulation steps (i.e., time steps) to perform for each
        `Culture` object. Default is `10`.
    rng_seed : int, optional
        Seed for the random number generator used in the simulation. This is
        the seed on which every other seed depends. Default is the hexadecimal
        number (representing a 128-bit integer)
        `0x87351080E25CB0FAD77A44A3BE03B491`.
    cell_radius : int, optional
        Radius of the cells in the simulation. Default is `1`.
    adjacency_threshold : int, optional
        Distance threshold for two cells to be considered neighbors. Default
        is `4`, which is an upper bound to the second neighbor distance of
        approximately `2 * sqrt(2)` in a hexagonal close packing.
    cell_max_repro_attempts : int, optional
        Maximum number of attempts to create a new cell during the
        reproduction of an existing cell in a `Culture` object.
        Default is`1000`.
    cell_max_def_attempts : int, optional
        The maximum number of deformation attempts a cell can make,
        by default 10.
    initial_number_of_cells : int, optional
        The number of cells in the culture. If None, we start with a single
        cell.
    initial_fraction_elongated : float, optional
        The initial fraction of elongated cells in the culture. If None, we
        start with all round cells.
    initial_density : float
        The initial density of the cells in the culture. None by default. If
        specified, it overrides the `culture_bounds` parameter to adjust the
        for the requested density.
    requested_number_of_cells : list of int, optional
        Approximate number of sites used to construct the complete
        triangular lattice.
    requested_density : list of float, optional
        Requested density after introducing vacancies. Mutually exclusive
        with requested_number_of_removed_cells.
    requested_number_of_removed_cells : list of int, optional
        Number of sites removed from the complete triangular lattice.
        Mutually exclusive with requested_density.
    reproduction : bool
        Whether the cells reproduces or not
    movement : bool
        Whether the cells moves or not
    deformation : bool
        Whether the cells deforms or not.
    stabilization_time : int
        The time we have to wait in order to start the deformation
    overlap_threshold_ratio : float
        A fraction (between 0 and 1) of the maximum allowed overlap between cells.
    overlap_threshold_tfg : float
        Overlap threshold used in the TFG.
    delta_t : float
        The time interval used to move
    initial_apect_ratio : float
        The aspect_ratio of all cells in the culture at the begining of the simulation.
    aspect_ratio_max : float
        The max value of the aspect ratio that a cell can have after deforms
    cell_speed_max : float
        The max value of speed that an elongated cell can achieve.
    delta_aspect_ratio : float
        Increase in the aspect ratio during deformation.
    culture_bounds : int, optional
        The bounds of the grid, by default None. If None, the space is
        unbouded. If provided, the space is bounded to the
        [0, culture_bounds)^3 cube.
    grid_cube_size : int, optional
        The size of the cubes in the grid, by default 2. This value comes
        from considering that cells have usually radius 1, so a cube of
        side $h=2r$ is enough to make sure that we only have to check
        superpositions with cells on the same or first neighboring grid
        cells. Enlarge if using larger cells.
        For simulations with eliptical cells, use $h=2r_{max}$.
    grid_torus : bool, optional
        Whether the grid is a torus or not, only relevant when bounds are
        provided, True by default. If True, the grid is a torus, so the
        cells that go out of the bounds appear on the other side of the
        grid. If False, the grid is a bounded cube, so behavior should be
        defined to manage what happens when cells go out of the bounds of
        the simulation.
    trabajo_final : bool
        Flag to determine wether to use or not mechanism of the TFG.
    initialization_mode: str
        String to determine the initial conditions to use.
    deformation_warmup_steps : int
        Number of initial simulation steps during which elongation
        attempts are always enabled.
    deformation_probe_steps : int
        Number of consecutive active steps without any successful
        deformation required to temporarily disable elongation attempts.
    elongation_sleep_steps : int
        Number of steps during which elongation attempts are disabled.
        Contractions remain active and immediately reactivate elongation
        from the following timestep if one occurs.

    Attributes
    ----------
    (All parameters, plus the following.)
    rng : `numpy.random.Generator`
        The random number generator used in the simulation to instatiate the
        generator of cultures and cells.
    cultures : dict
        Dictionary storing the `Culture` objects simulated by the `Simulation`.
        The keys are strings representing the combinations of `prob_stem` and
        `prob_diff` and the realization number.


    Methods
    -------
    simulate_parallel()
        Runs the simulation persisting data to one file for each culture.
    """

    def __init__(
        self,
        forces: List[Force] = None,
        first_cell_is_stem: bool = True,
        prob_stem: List[float] = [0.36],
        prob_diff: List[float] = [0],
        num_of_realizations: int = 4,
        num_of_steps_per_realization: int = 10,
        rng_seed=0x87351080E25CB0FAD77A44A3BE03B491,
        cell_radius: float = 1,
        adjacency_threshold: float = 4,
        cell_max_repro_attempts: int = 1000,
        cell_max_def_attempts: int = 10,
        swap_probability: float = 0.5,
        culture_bounds: float = None,
        grid_cube_size: Union[float, List[float]] = 2,
        grid_torus: bool = True,
        initial_number_of_cells: Optional[List[int]] = [400],
        initial_fraction_elongated: Optional[List[float]] = [0.0],
        initial_density: Optional[List[float]] = None,
        requested_number_of_cells: Optional[List[int]] = None,
        requested_density: Optional[List[float]] = None,
        requested_number_of_removed_cells: Optional[List[int]] = None,
        reproduction: bool = False,
        movement: bool = True,
        deformation: bool = True,
        stabilization_time: int = 120,
        overlap_threshold_ratio: float = 0.35,
        overlap_threshold_tfg: float = 0.61,
        delta_t: float = 0.05,
        initial_aspect_ratio: float = 1,
        aspect_ratio_max: float = 5,
        cell_speed_max : float = 1,
        delta_aspect_ratio: float = 0.1,
        trabajo_final: bool = False,
        initialization_mode: str = "random",
        deformation_warmup_steps: int = 5_000,
        deformation_probe_steps: int = 1_000,
        elongation_sleep_steps: int = 5_000,
    ):
        # main simulation attributes
        self.forces = forces
        self.initial_number_of_cells = initial_number_of_cells
        self.initial_fraction_elongated = initial_fraction_elongated
        self.initial_density = initial_density

        self.requested_number_of_cells = (
            requested_number_of_cells
        )

        self.requested_density = (
            requested_density
        )

        self.requested_number_of_removed_cells = (
            requested_number_of_removed_cells
        )
        self.reproduction = reproduction
        self.movement = movement
        self.deformation = deformation
        self.first_cell_is_stem = first_cell_is_stem
        self.prob_stem = prob_stem
        self.prob_diff = prob_diff
        # self.prob_supervivence_radiotherapy = prob_supervivence_radiotherapy
        self.num_of_realizations = num_of_realizations
        self.num_of_steps_per_realization = num_of_steps_per_realization
        self.swap_probability = swap_probability
        self._rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)
        self.stabilization_time = stabilization_time
        self.overlap_threshold_ratio = overlap_threshold_ratio
        self.overlap_threshold_tfg = overlap_threshold_tfg
        self.delta_t = delta_t
        self.initial_aspect_ratio = initial_aspect_ratio
        self.aspect_ratio_max = aspect_ratio_max
        self.cell_speed_max = cell_speed_max
        self.delta_aspect_ratio = delta_aspect_ratio

        # TFG 
        self.trabajo_final = trabajo_final

        # Adaptive elongation timing
        self.deformation_warmup_steps = (
            deformation_warmup_steps
        )

        self.deformation_probe_steps = (
            deformation_probe_steps
        )

        self.elongation_sleep_steps = (
            elongation_sleep_steps
        )
        # Initialization mode
        valid_initialization_modes = {
            "random",
            "triangular_vacancies",
        }

        if initialization_mode not in valid_initialization_modes:
            raise ValueError(
                "initialization_mode must be either "
                "'random' or 'triangular_vacancies'."
            )

        self.initialization_mode = initialization_mode

        if initialization_mode == "random":
            if requested_number_of_cells is not None:
                raise ValueError(
                    "requested_number_of_cells is only available "
                    "for triangular_vacancies initialization."
                )

            if requested_density is not None:
                raise ValueError(
                    "requested_density is only available "
                    "for triangular_vacancies initialization."
                )

            if requested_number_of_removed_cells is not None:
                raise ValueError(
                    "requested_number_of_removed_cells is only available "
                    "for triangular_vacancies initialization."
                )

        else:
            if requested_number_of_cells is None:
                raise ValueError(
                    "The triangular_vacancies initialization requires "
                    "requested_number_of_cells."
                )

            density_was_provided = (
                requested_density is not None
            )

            removed_cells_were_provided = (
                requested_number_of_removed_cells
                is not None
            )

            if (
                density_was_provided
                == removed_cells_were_provided
            ):
                raise ValueError(
                    "For triangular_vacancies, provide exactly one "
                    "of requested_density or "
                    "requested_number_of_removed_cells."
                )

        # dictionary storing the culture objects
        self.cultures = {}

        # attributes to pass to the culture (and to the cells)
        self.cell_max_repro_attempts = cell_max_repro_attempts
        self.cell_max_def_attempts = cell_max_def_attempts
        self.adjacency_threshold = adjacency_threshold
        self.cell_radius = cell_radius

        # attributes for the spatial hash grid
        self.culture_bounds = culture_bounds
        self.grid_cube_size = grid_cube_size
        self.grid_torus = grid_torus

    def calculate_culture_bounds_from_density(
        self,
        number_of_cells: int,
        density: float,
    ) -> float:
        """Calculate the culture bounds from the initial density, provided
        the number of cells."""
        if self.initial_density is None:
            pass
        else:
            cell_area = np.pi * self.cell_radius**2
            bounds = np.sqrt(number_of_cells * cell_area / density)
            return bounds

    def calculate_triangular_lattice_geometry(
        self,
        requested_number_of_cells: int,
    ) -> dict:
        """
        Calculate the geometry of an approximately triangular lattice
        contained in a square periodic box.

        The requested number of cells is interpreted as an approximate
        reference size. The actual number of lattice sites is

            reference_number_of_cells
                = number_of_columns * number_of_rows.

        The number of rows is chosen as the closest even integer to the
        triangular-lattice estimate. The number of columns is then chosen
        so that the natural rectangular lattice is approximately square.

        The square side is the maximum between the natural lattice width
        and height. Therefore, the lattice is never compressed and no
        geometrical overlaps are introduced.

        Parameters
        ----------
        requested_number_of_cells : int
            Approximate number of sites requested by the user.

        Returns
        -------
        geometry : dict
            Dictionary containing the lattice dimensions, square side,
            spacings and full-lattice packing fraction.
        """
        if requested_number_of_cells <= 0:
            raise ValueError(
                "requested_number_of_cells must be positive."
            )

        # Estimate the number of rows from
        # N ≈ (sqrt(3) / 2) * n_y**2
        estimated_number_of_rows = np.sqrt(
            (
                2
                * requested_number_of_cells
            )
            / np.sqrt(3)
        )

        # Choose the closest even number of rows
        number_of_rows = max(
            2,
            2
            * int(
                np.round(
                    estimated_number_of_rows / 2
                )
            ),
        )

        # Choose the number of columns so that
        # n_x / n_y ≈ sqrt(3) / 2
        number_of_columns = max(
            1,
            int(
                np.round(
                    (
                        np.sqrt(3)
                        / 2
                    )
                    * number_of_rows
                )
            ),
        )

        # Now we calculate the reference number of cells
        reference_number_of_cells = (
            number_of_columns
            * number_of_rows
        )

        # Natural dimensions of a compact triangular lattice
        natural_width = (
            2
            * self.cell_radius
            * number_of_columns
        )

        natural_height = (
            np.sqrt(3)
            * self.cell_radius
            * number_of_rows
        )

        # Use the longest natural dimension as the square side
        side = max(
            natural_width,
            natural_height,
        )

        spacing_x = (
            side
            / number_of_columns
        )

        spacing_y = (
            side
            / number_of_rows
        )

        cell_area = (
            np.pi
            * self.cell_radius**2
        )

        full_density = (
            reference_number_of_cells
            * cell_area
            / side**2
        )

        return {
            "requested_number_of_cells": (
                requested_number_of_cells
            ),
            "number_of_columns": (
                number_of_columns
            ),
            "number_of_rows": (
                number_of_rows
            ),
            "reference_number_of_cells": (
                reference_number_of_cells
            ),
            "natural_width": natural_width,
            "natural_height": natural_height,
            "side": side,
            "spacing_x": spacing_x,
            "spacing_y": spacing_y,
            "full_density": full_density,
        }

    def generate_triangular_lattice_positions(
        self,
        geometry: dict,
    ) -> np.ndarray:
        """
        Generate all positions of the approximately triangular lattice.

        Rows alternate between two horizontal offsets. The positions are
        centered inside the square periodic box.

        Parameters
        ----------
        geometry : dict
            Geometry returned by
            calculate_triangular_lattice_geometry().

        Returns
        -------
        positions : np.ndarray
            Array with shape (reference_number_of_cells, 3).
        """
        number_of_columns = int(
            geometry["number_of_columns"]
        )

        number_of_rows = int(
            geometry["number_of_rows"]
        )

        reference_number_of_cells = int(
            geometry["reference_number_of_cells"]
        )

        side = float(
            geometry["side"]
        )

        spacing_x = float(
            geometry["spacing_x"]
        )

        spacing_y = float(
            geometry["spacing_y"]
        )

        expected_number_of_cells = (
            number_of_columns
            * number_of_rows
        )

        if (
            reference_number_of_cells
            != expected_number_of_cells
        ):
            raise ValueError(
                "reference_number_of_cells must be equal to "
                "number_of_columns * number_of_rows."
            )

        # Row-major ordering:
        # row 0: columns 0, 1, ..., n_x - 1
        # row 1: columns 0, 1, ..., n_x - 1
        # ...
        row_indices = np.repeat(
            np.arange(
                number_of_rows,
                dtype=int,
            ),
            number_of_columns,
        )

        column_indices = np.tile(
            np.arange(
                number_of_columns,
                dtype=int,
            ),
            number_of_rows,
        )

        # Odd rows are shifted by half a horizontal spacing
        horizontal_offsets = (
            0.5
            * (
                row_indices
                % 2
            )
        )

        x_positions = (
            (
                column_indices
                + 0.5
                + horizontal_offsets
            )
            * spacing_x
        )

        # The modulo is needed because the final point of an odd row
        # can coincide with the right periodic boundary
        x_positions = np.mod(
            x_positions,
            side,
        )

        y_positions = (
            row_indices
            + 0.5
        ) * spacing_y

        z_positions = np.zeros(
            reference_number_of_cells,
            dtype=float,
        )

        positions = np.column_stack(
            (
                x_positions,
                y_positions,
                z_positions,
            )
        )

        return positions

    def select_triangular_lattice_positions(
        self,
        lattice_positions: np.ndarray,
        geometry: dict,
        target_density: Optional[float] = None,
        number_of_removed_cells: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """
        Select the occupied sites of a triangular lattice after introducing
        random vacancies.

        The final configuration can be specified either through a target
        packing fraction or through the number of sites to remove. Exactly
        one of these two arguments must be provided.

        Parameters
        ----------
        lattice_positions : np.ndarray
            Positions of all sites in the complete lattice. Its shape must
            be (reference_number_of_cells, 3).

        geometry : dict
            Geometry returned by
            calculate_triangular_lattice_geometry().

        target_density : float, optional
            Requested packing fraction after removing lattice sites.

        number_of_removed_cells : int, optional
            Number of sites to remove from the complete lattice.

        rng : numpy.random.Generator, optional
            Random number generator used to select the vacancies. If None,
            the Simulation random number generator is used.

        Returns
        -------
        selection : dict
            Dictionary containing the occupied positions, occupied and
            vacant indices, final number of cells and actual density.
        """
        # Make sure that only the density or the number of cells to removed
        # was provided
        target_density_was_provided = (
            target_density is not None
        )

        number_removed_was_provided = (
            number_of_removed_cells is not None
        )

        if (
            target_density_was_provided
            == number_removed_was_provided
        ):
            raise ValueError(
                "Provide exactly one of target_density or "
                "number_of_removed_cells."
            )

        if rng is None:
            rng = self.rng

        # Take the data from the lattice geometry
        reference_number_of_cells = int(
            geometry["reference_number_of_cells"]
        )

        side = float(
            geometry["side"]
        )

        full_density = float(
            geometry["full_density"]
        )

        expected_shape = (
            reference_number_of_cells,
            3,
        )

        if lattice_positions.shape != expected_shape:
            raise ValueError(
                "lattice_positions must have shape "
                f"{expected_shape}, but received "
                f"{lattice_positions.shape}."
            )

        cell_area = (
            np.pi
            * self.cell_radius**2
        )

        # Analyzed the case in which the target density was provided
        if target_density_was_provided:
            target_density = float(
                target_density
            )

            if target_density <= 0:
                raise ValueError(
                    "target_density must be positive."
                )

            if (
                target_density > full_density
                and not np.isclose(
                    target_density,
                    full_density,
                )
            ):
                raise ValueError(
                    "The requested density is larger than the "
                    "density of the complete lattice. "
                    f"Requested density: {target_density:.6f}. "
                    f"Maximum available density: "
                    f"{full_density:.6f}."
                )

            number_of_cells = int(
                np.rint(
                    (
                        target_density
                        * side**2
                    )
                    / cell_area
                )
            )

            # Protect against small floating-point differences when the
            # requested density is equal to the complete-lattice density.
            number_of_cells = min(
                number_of_cells,
                reference_number_of_cells,
            )

            if number_of_cells <= 0:
                raise ValueError(
                    "The requested density produces no occupied sites."
                )

            number_of_removed_cells = (
                reference_number_of_cells
                - number_of_cells
            )

        # Analyzed the case in which the number of removed cells was provided
        else:
            if not isinstance(
                number_of_removed_cells,
                (int, np.integer),
            ):
                raise TypeError(
                    "number_of_removed_cells must be an integer."
                )

            number_of_removed_cells = int(
                number_of_removed_cells
            )

            if not (
                0
                <= number_of_removed_cells
                < reference_number_of_cells
            ):
                raise ValueError(
                    "number_of_removed_cells must satisfy "
                    "0 <= number_of_removed_cells "
                    "< reference_number_of_cells."
                )

            number_of_cells = (
                reference_number_of_cells
                - number_of_removed_cells
            )

        # For each case, take all the indices
        all_indices = np.arange(
            reference_number_of_cells,
            dtype=int,
        )

        # Take the indices to be vacant
        if number_of_removed_cells == 0:
            vacant_indices = np.empty(
                0,
                dtype=int,
            )

        else:
            vacant_indices = rng.choice(
                reference_number_of_cells,
                size=number_of_removed_cells,
                replace=False,
            )

            vacant_indices = np.sort(
                vacant_indices
            )

        # Applied a mask to the geometry to have vacants
        occupied_mask = np.ones(
            reference_number_of_cells,
            dtype=bool,
        )

        occupied_mask[
            vacant_indices
        ] = False

        occupied_indices = all_indices[
            occupied_mask
        ]

        occupied_positions = lattice_positions[
            occupied_indices
        ].copy()

        actual_density = (
            number_of_cells
            * cell_area
            / side**2
        )

        return {
            "positions": occupied_positions,
            "occupied_indices": occupied_indices,
            "vacant_indices": vacant_indices,
            "number_of_cells": number_of_cells,
            "number_of_removed_cells": (
                number_of_removed_cells
            ),
            "target_density": target_density,
            "actual_density": actual_density,
            "full_density": full_density,
        }

    def simulate_single_culture(
        self,
        sql: bool = True,
        dat_files: bool = False,
        dat_pos_ar: bool = False,
        dat_order_par: bool = False,
        dat_motion_par: bool = False,
        dat_cluster_par: bool = False,
        dat_deformation_par: bool = False,
        dat_local_order_par: bool = False,
        ovito: bool = False,
        df: bool = False,
        output_dir: str = ".",
        prob_stem_index: int = 0,
        prob_diff_index: int = 0,
    ):
        """Like simulate_parallel but for a single culture.

        Mainly intended to be used when debugging or testing the simulation,
        tasks with which the parallelization can interfere.

        Notes
        -----
        As the RNG is already initialized, the use of this method can alter
        reproducibility.
        """
        seed = self.rng.integers(low=2**20, high=2**50, size=1)

        outputs = []
        if sql:
            outputs.append("sql")
        if dat_files:
            outputs.append("dat")
        if ovito:
            outputs.append("ovito")
        if df:
            outputs.append("df")
        if dat_pos_ar:
            outputs.append("dat_pos_ar")
        if dat_order_par:
            outputs.append("dat_order_par")
        if dat_motion_par:
            outputs.append("dat_motion_par")
        if dat_cluster_par:
            outputs.append("dat_cluster_par")
        if dat_deformation_par:
            outputs.append("dat_deformation_par")
        if dat_local_order_par:
            outputs.append("dat_local_order_par")

        # We compute the name of the realization
        current_realization_name = realization_name(
            self.prob_diff[prob_diff_index],
            self.prob_stem[prob_stem_index],
            seed.item(),
        )

        # We create the output object
        output = create_output_demux(
            current_realization_name, outputs, output_dir
        )

        # We create the spatial hash grid object
        spatial_hash_grid = SpatialHashGrid(
            culture=None,
            bounds=self.culture_bounds,
            cube_size=self.grid_cube_size,
            torus=self.grid_torus,
        )

        # We create the culture object and simulate it
        self.cultures[current_realization_name] = Culture(
            output=output,
            grid=spatial_hash_grid,
            adjacency_threshold=self.adjacency_threshold,
            cell_radius=self.cell_radius,
            cell_max_repro_attempts=self.cell_max_repro_attempts,
            cell_max_def_attempts=self.cell_max_def_attempts,
            first_cell_is_stem=self.first_cell_is_stem,
            prob_stem=self.prob_stem[prob_stem_index],
            prob_diff=self.prob_diff[prob_diff_index],
            rng_seed=seed.item(),
            swap_probability=self.swap_probability,
            trabajo_final=self.trabajo_final,
        )
        self.cultures[current_realization_name].simulate(
            self.num_of_steps_per_realization,
        )

    def simulate_parallel(
        self,
        sql: bool = True,
        dat_files: bool = False,
        dat_pos_ar: bool = False,
        save_step_dat_pos_ar: int = 1,
        dat_order_par: bool = False,
        dat_motion_par: bool = False,
        dat_cluster_par: bool = False,
        dat_deformation_par: bool = False,
        dat_local_order_par: bool = False,
        save_step_dat_order_par: int = 100,
        save_step_dat_motion_par: int = 100,    
        save_step_dat_cluster_summary: int = 100,
        save_step_dat_cluster_raw: int = 1000,
        save_step_dat_deformation_par: int = 100,
        save_step_dat_local_order_summary: int = 100,
        save_step_dat_local_order_raw: int = 1000,
        ovito: bool = False,
        save_step_ovito: int = 1,
        df: bool = False,
        number_of_processes: int = None,
        output_dir: str = ".",
    ) -> None:
        """
        Simulate the growth of multiple cultures in parallel.

        Simulate culture growth `self.num_of_realizations` number of times
        for each combination of self-replication (elements of the
        `self.prob_stem` list) and differentiation probabilities (elements of
        the `self.prob_diff` list), persisting the data of each culture to its
        own file. The simulations are parallelized using multiprocessing.

        Several different output types are simultaneously available, and the
        data that is recorded is handled by the `TumorsphereOutput` classes.
        If `number_of_processes` is None (default), the number of processes is
        equal to the number of cores in the machine. Limitting the number of
        processes is useful when running the simulation in a cluster, where
        the number of cores is limited, or when running with all the resources
        might trigger an alarm.

        Parameters
        ----------
        number_of_processes : int
            The number of the processes. If None (default), the number of
            processes is equal to the number of cores in the machine.
        """
        if number_of_processes is None:
            number_of_processes = mp.cpu_count()

        # Generate seeds for all realizations
        seeds = self.rng.integers(
            low=2**20, high=2**50, size=self.num_of_realizations
        )

        outputs = []
        if sql:
            outputs.append("sql")
        if dat_files:
            outputs.append("dat")
        if ovito:
            outputs.append("ovito")
        if df:
            outputs.append("df")
        if dat_pos_ar:
            outputs.append("dat_pos_ar")
        if dat_order_par:
            outputs.append("dat_order_par")
        if dat_motion_par:
            outputs.append("dat_motion_par")
        if dat_cluster_par:
            outputs.append("dat_cluster_par")
        if dat_deformation_par:
            outputs.append("dat_deformation_par")
        if dat_local_order_par:
            outputs.append("dat_local_order_par")

        # Choose the parameters depending on the initialization mode
        if self.initialization_mode == "random":
            number_of_cells_values = (
                self.initial_number_of_cells
            )

            density_values = (
                self.initial_density
            )

            removed_cells_values = None

        else:
            number_of_cells_values = (
                self.requested_number_of_cells
            )

            density_values = (
                self.requested_density
            )

            removed_cells_values = (
                self.requested_number_of_removed_cells
            )

        with mp.Pool(number_of_processes) as p:
            p.map(
                simulate_single_culture,
                [
                    (
                        k,
                        i,
                        f,
                        t,
                        g,
                        r,
                        seeds[j],
                        self,
                        outputs,
                        save_step_dat_pos_ar,
                        save_step_dat_order_par,
                        save_step_dat_motion_par,
                        save_step_dat_cluster_summary,
                        save_step_dat_cluster_raw,
                        save_step_dat_deformation_par,
                        save_step_dat_local_order_summary,
                        save_step_dat_local_order_raw,
                        save_step_ovito,
                        m,
                        output_dir,
                        self.culture_bounds,
                        self.grid_cube_size,
                        self.grid_torus,
                    )
                    for k in range(len(self.prob_diff))
                    for i in range(len(self.prob_stem))
                    for f in range(len(number_of_cells_values))
                    for t in range(len(self.initial_fraction_elongated))
                    for g in (
                        range(len(density_values))
                        if density_values is not None
                        else [None]
                    )
                    for r in (
                        range(len(removed_cells_values))
                        if removed_cells_values is not None
                        else [None]
                    )
                    for j in range(self.num_of_realizations)
                    for m in range(len(self.forces))
                ],
            )


def realization_name(
    pd: float,
    ps: float,
    nc: int,
    f_e: float,
    rho: float,
    seed: int,
    force_name: str,
    bounds: Optional[float],
    repro: bool,
    moving: bool,
    initialization_mode: str = "random",
    reference_number_of_cells: Optional[int] = None,
    actual_number_of_cells: Optional[int] = None,
    actual_density: Optional[float] = None,
    requested_number_of_removed_cells: Optional[int] = None,
) -> str:
    """Return the name of the realization."""
    not_supported = not (repro or moving)

    if not_supported:
        raise NotImplementedError(
            "Simulations that do not involve either reproduction or movement "
            "are not implemented."
        )
    name = "culture"

    if repro:
        name += (
            f"_pd={pd}"
            f"_ps={ps}"
        )

    if moving:
        if initialization_mode == "random":
            name += (
                f"_initial_nc={nc}"
            )

            if rho is not None:
                name += (
                    f"_density={rho:g}"
                )
            else:
                name += (
                    f"_bounds={bounds:g}"
                )

        elif (
            initialization_mode
            == "triangular_vacancies"
        ):
            name += (
                f"_requested_nc={nc}"
            )

            name += (
                f"_reference_nc="
                f"{reference_number_of_cells}"
            )

            name += (
                f"_initial_nc="
                f"{actual_number_of_cells}"
            )

            if rho is not None:
                name += (
                    f"_requested_density={rho:g}"
                )

            else:
                name += (
                    f"_removed_nc="
                    f"{requested_number_of_removed_cells}"
                )

            name += (
                f"_density="
                f"{actual_density:.6f}"
            )

        if not np.isclose(
            f_e,
            0.0,
        ):
            name += (
                f"_initial_f_e="
                f"{f_e:g}"
            )

        name += (
            f"_force={force_name}"
        )

    name += (
        f"_rng_seed={seed}"
    )

    return name

def simulate_single_culture(
    args: Tuple[int, int, int, Simulation, List[str], str]
) -> None:
    """A worker function for multiprocessing.

    This function is used by the multiprocessing.Pool instance in the
    simulate_parallel method to parallelize the simulation of different
    cultures. This simulates the growth of a single culture with the given
    parameters and persists the data.

    Parameters
    ----------
    args : tuple
        A tuple containing the indices for the self-replication probability,
        differentiation probability, the seed to be used in the random number
        generator of the culture, an instance of the Simulation class, and a
        list of strings specifying the desired output types.

    Notes
    -----
    Due to the way multiprocessing works in Python, you can't directly use
    instance methods as workers for multiprocessing. The multiprocessing
    module needs to be able to pickle the target function, and instance
    methods can't be pickled. Therefore, the instance method worker had to be
    refactored to a standalone function (or a static method).
    """
    # We unpack the arguments
    (
        k,
        i,
        f,
        t,
        g,
        r,
        seed,
        sim,
        outputs,
        save_step_dat_pos_ar,
        save_step_dat_order_par,
        save_step_dat_motion_par,
        save_step_dat_cluster_summary,
        save_step_dat_cluster_raw,
        save_step_dat_deformation_par,
        save_step_dat_local_order_summary,
        save_step_dat_local_order_raw,
        save_step_ovito,
        m,
        output_dir,
        culture_bounds,
        grid_cube_size,
        grid_torus,
    ) = args

    # Requested simulation parameters depending on the initialization mode
    if sim.initialization_mode == "random":
        number_of_cells = int(
            sim.initial_number_of_cells[f]
        )

        density = (
            float(sim.initial_density[g])
            if sim.initial_density is not None
            else None
        )

        number_of_removed_cells = None

    else:
        number_of_cells = int(
            sim.requested_number_of_cells[f]
        )

        density = (
            float(sim.requested_density[g])
            if sim.requested_density is not None
            else None
        )

        number_of_removed_cells = (
            int(
                sim.requested_number_of_removed_cells[r]
            )
            if (
                sim.requested_number_of_removed_cells
                is not None
            )
            else None
        )
        
    # Default values for the random initialization
    reference_number_of_cells = (
        number_of_cells
    )

    actual_number_of_cells = (
        number_of_cells
    )

    actual_density = density

    initial_positions = None

    effective_stabilization_time = (
        sim.stabilization_time
    )

    if (
        sim.initialization_mode
        == "triangular_vacancies"
    ):
        if not np.isclose(
            sim.initial_aspect_ratio,
            1.0,
        ):
            raise ValueError(
                "The triangular_vacancies initialization "
                "requires initial_aspect_ratio = 1."
            )

        if not grid_torus:
            raise ValueError(
                "The triangular_vacancies initialization "
                "requires periodic boundary conditions."
            )
        geometry = (
            sim.calculate_triangular_lattice_geometry(
                requested_number_of_cells=(
                    number_of_cells
                ),
            )
        )

        lattice_positions = (
            sim.generate_triangular_lattice_positions(
                geometry=geometry,
            )
        )

        # Encode the vacancy-control parameter for SeedSequence
        if density is not None:
            vacancy_control_type = 0

            vacancy_control_value = int(
                np.rint(
                    density
                    * 1_000_000_000
                )
            )

        else:
            vacancy_control_type = 1

            vacancy_control_value = int(
                number_of_removed_cells
            )

        # Use the same vacancy configuration for equal realizations,
        # sizes and densities across other parameter combinations
        vacancy_seed_sequence = np.random.SeedSequence(
            [
                int(seed),
                int(number_of_cells),
                vacancy_control_type,
                vacancy_control_value,
                1,
            ]
        )

        vacancy_rng = np.random.default_rng(
            vacancy_seed_sequence
        )

        selection = (
            sim.select_triangular_lattice_positions(
                lattice_positions=lattice_positions,
                geometry=geometry,
                target_density=density,
                number_of_removed_cells=(
                    number_of_removed_cells
                ),
                rng=vacancy_rng,
            )
        )

        culture_bounds = float(
            geometry["side"]
        )

        initial_positions = selection[
            "positions"
        ]

        reference_number_of_cells = int(
            geometry[
                "reference_number_of_cells"
            ]
        )

        actual_number_of_cells = int(
            selection["number_of_cells"]
        )

        actual_density = float(
            selection["actual_density"]
        )

        # Deformation is enabled from the first dynamic step.
        effective_stabilization_time = 0

    else:
        if density is not None:
            culture_bounds = (
                sim.calculate_culture_bounds_from_density(
                    number_of_cells=(
                        number_of_cells
                    ),
                    density=density,
                )
            )
        else:
            culture_bounds = sim.culture_bounds


    # We compute the name of the realization after preparing
    # the initial condition.
    current_realization_name = realization_name(
        sim.prob_diff[k],
        sim.prob_stem[i],
        number_of_cells,
        sim.initial_fraction_elongated[t],
        density,
        seed,
        sim.forces[m].name(),
        culture_bounds,
        sim.reproduction,
        sim.movement,
        initialization_mode=(
            sim.initialization_mode
        ),
        reference_number_of_cells=(
            reference_number_of_cells
        ),
        actual_number_of_cells=(
            actual_number_of_cells
        ),
        actual_density=actual_density,
        requested_number_of_removed_cells=(
            number_of_removed_cells
        ),
    )

    checkpoint_path_save = os.path.join(output_dir, "checkpoints", current_realization_name + ".pkl")
    # checkpoint_dir = os.path.join(os.environ["HOME"], "oncostream", "checkpoints")
    # os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints") #
    os.makedirs(checkpoint_dir, exist_ok=True) #
    checkpoint_path = os.path.join(checkpoint_dir, current_realization_name + ".pkl")
    # Verify if there is a checkpoint
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            #culture, start_tic = pickle.load(f)
            #sim.cultures[current_realization_name] = culture
            culture, start_tic, state = pickle.load(f)
            sim.cultures[current_realization_name] = culture
            sim.cultures[current_realization_name].rng = np.random.default_rng()
            sim.cultures[current_realization_name].rng.bit_generator.state = state
    else:
        # We create the output object
        output = create_output_demux(
            culture_name=current_realization_name,
            requested_outputs=outputs,
            output_dir=output_dir,
            save_step_dat_pos_ar=save_step_dat_pos_ar,
            save_step_dat_order_par=save_step_dat_order_par,
            save_step_dat_motion_par=save_step_dat_motion_par,
            save_step_dat_cluster_summary=(
                save_step_dat_cluster_summary
            ),
            save_step_dat_cluster_raw=(
                save_step_dat_cluster_raw
            ),
            save_step_dat_deformation_par=(
                save_step_dat_deformation_par
            ),
            save_step_dat_local_order_summary=(
                save_step_dat_local_order_summary
            ),
            save_step_dat_local_order_raw=(
                save_step_dat_local_order_raw
            ),
            save_step_ovito=save_step_ovito,
        )

        spatial_hash_grid = SpatialHashGrid(
            culture=None,
            bounds=culture_bounds,
            cube_size=grid_cube_size,
            torus=grid_torus,
        )

        # We create the culture object and simulate it
        sim.cultures[current_realization_name] = Culture(
            output=output,
            force=sim.forces[m],
            initial_number_of_cells=actual_number_of_cells,
            initial_fraction_elongated=sim.initial_fraction_elongated[t],
            grid=spatial_hash_grid,
            adjacency_threshold=sim.adjacency_threshold,
            cell_radius=sim.cell_radius,
            cell_max_repro_attempts=sim.cell_max_repro_attempts,
            cell_max_def_attempts=sim.cell_max_def_attempts,
            first_cell_is_stem=sim.first_cell_is_stem,
            prob_stem=sim.prob_stem[i],
            prob_diff=sim.prob_diff[k],
            rng_seed=seed,
            swap_probability=sim.swap_probability,    
            reproduction=sim.reproduction,
            movement=sim.movement,
            deformation=sim.deformation,
            stabilization_time=effective_stabilization_time,
            overlap_threshold_ratio=sim.overlap_threshold_ratio,
            overlap_threshold_tfg=sim.overlap_threshold_tfg,
            delta_t=sim.delta_t,
            initial_aspect_ratio=sim.initial_aspect_ratio,
            aspect_ratio_max=sim.aspect_ratio_max,
            cell_speed_max=sim.cell_speed_max,
            delta_aspect_ratio=sim.delta_aspect_ratio,
            trabajo_final=sim.trabajo_final,
            initialization_mode=sim.initialization_mode,
            initial_positions=initial_positions,
            deformation_warmup_steps=(
                sim.deformation_warmup_steps
            ),
            deformation_probe_steps=(
                sim.deformation_probe_steps
            ),
            elongation_sleep_steps=(
                sim.elongation_sleep_steps
            ),
        )
        start_tic=0
    sim.cultures[current_realization_name].simulate(
        sim.num_of_steps_per_realization,
        start_tic,
        checkpoint_path_save,
    )
