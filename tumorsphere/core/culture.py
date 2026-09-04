"""
Module containing the Culture class.

Classes:
    - Culture: Class that represents a culture of cells. Usually dependent
    on the Simulation class.
"""

# import os
from datetime import datetime
from typing import (
    Set,
    Dict,
    List,
    Tuple,
    Optional,
)

import pandas as pd
import numpy as np
import pickle
import os

from tumorsphere.core.cells import Cell
from tumorsphere.core.output import TumorsphereOutput
from tumorsphere.core.spatial_hash_grid import SpatialHashGrid
from tumorsphere.core.forces import Force

class _UnionFind:
    """
    Manage disjoint groups of integer indices.

    In the cluster calculation, each integer index represents a cell
    and each disjoint group represents one connected component.

    Notes
    -----
    - ``parent[index]`` points to the parent of ``index`` in the internal tree.
    - A root is an index that points to itself.
    - ``size[root]`` stores the number of elements in the group represented
      by ``root``. Values of ``size`` are only meaningful for roots.
    """

    def __init__(self, number_of_elements: int):
        """
        Initialize one independent group for each element.

        Initially, every element is its own root and therefore represents
        a connected component of size one.
        """
        self.parent = np.arange(
            number_of_elements,
            dtype=int,
        )

        self.size = np.ones(
            number_of_elements,
            dtype=int,
        )

    def find(self, index: int) -> int:
        """
        Return the root of the group containing ``index``.

        Path compression is applied so that all nodes visited during the
        search point directly to the root. This makes future searches faster.
        """
        root = index

        # Follow parent links until reaching the root.
        while self.parent[root] != root:
            root = self.parent[root]

        # Make every node visited on the path point directly to the root.
        while self.parent[index] != index:
            next_index = self.parent[index]
            self.parent[index] = root
            index = next_index

        return int(root)

    def union(self, index_i: int, index_j: int) -> None:
        """
        Merge the groups containing ``index_i`` and ``index_j``.

        The smaller group is attached below the larger group to keep the
        internal trees shallow. If both groups have the same size, the
        numerically smaller root is retained for deterministic behavior.
        """
        root_i = self.find(index_i)
        root_j = self.find(index_j)

        # The two elements already belong to the same connected component.
        if root_i == root_j:
            return

        # After this possible swap, root_i represents the larger group.
        should_swap = (
            self.size[root_i] < self.size[root_j]
            or (
                self.size[root_i] == self.size[root_j]
                and root_i > root_j
            )
        )

        if should_swap:
            root_i, root_j = root_j, root_i

        # Attach the smaller tree below the root of the larger tree.
        self.parent[root_j] = root_i

        # Only the size stored at the surviving root remains relevant.
        self.size[root_i] += self.size[root_j]

    def groups(self) -> dict[int, list[int]]:
        """
        Return all connected components as lists of element indices.

        Returns
        -------
        groups : dict[int, list[int]]
            Dictionary whose keys are the roots and whose values contain
            the indices belonging to each connected component.
        """
        groups = {}

        for index in range(len(self.parent)):
            root = self.find(index)

            if root not in groups:
                groups[root] = []

            groups[root].append(index)

        return groups


class Culture:
    """
    Class that represents a culture of cells.

    This class handles the simulation, as well as some behavior of the cells,
    such as reproduction.
    """

    def __init__(
        self,
        output: TumorsphereOutput,
        force: Force,
        grid: SpatialHashGrid,
        adjacency_threshold: float = 4,
        cell_radius: float = 1,
        cell_max_repro_attempts: int = 1000,
        cell_max_def_attempts: int = 10,
        first_cell_is_stem: bool = True,
        prob_stem: float = 0,
        prob_diff: float = 0,
        rng_seed: int = 110293658491283598,
        swap_probability: float = 0.5,
        initial_number_of_cells: int = 1,
        initial_fraction_elongated: float = 0.0,
        reproduction: bool = False,
        movement: bool = True,
        deformation: bool = True,
        stabilization_time: int = 120,
        overlap_threshold_ratio: float = 0.35,
        overlap_threshold_tfg: float = 0.61,
        delta_t: float = 0.05,
        initial_aspect_ratio: float = 1,
        aspect_ratio_max: float = 5,
        cell_speed_max: float = 1,
        delta_aspect_ratio: float = 0.1,
        trabajo_final: bool = False,
        initialization_mode: str = "random",
        initial_positions: Optional[np.ndarray] = None,
        deformation_warmup_steps: int = 5_000,
        deformation_probe_steps: int = 1_000,
        elongation_sleep_steps: int = 5_000,
        contraction_overlap_safety_ratio: Optional[float] = None,
    ):
        """
        Initialize a new culture of cells.

        Parameters
        ----------
        output : TumorsphereOutput
            The output object to record the simulation data.
        force : Force
            The force used in the interaction between cells.
        grid : SpatialHashGrid
            The spatial hash grid to be used in the simulation.
        adjacency_threshold : int, optional
            The maximum distance at which two cells can be considered
            neighbors, by default 4.
        cell_radius : int, optional
            The radius of a cell, by default 1.
        cell_max_repro_attempts : int, optional
            The maximum number of reproduction attempts a cell can make,
            by default 1000.
        cell_max_def_attempts : int, optional
            The maximum number of deformation attempts a cell can make,
            by default 10.
        first_cell_is_stem : bool, optional
            Whether the first cell is a stem cell or not, by default False.
        prob_stem : float, optional
            The probability that a cell becomes a stem cell, by default 0.
        prob_diff : float, optional
            The probability that a cell differentiates, by default 0.
        rng_seed : int, optional
            Seed for the random number generator, by default
            110293658491283598.
        initial_number_of_cells : int, optional
            The initial number of cells in the culture.
        initial_fraction_elongated : float, optional
            The initial fraction of elongated cells.
        reproduction : bool
            Whether the cells reproduces or not.
        movement : bool
            Whether the cells moves or not.
        deformation : bool
            Whether the cells deforms or not.
        cell_area : float
            The area of all cells in the culture.
        stabilization_time : int
            The time we have to wait in order to start the deformation.
        overlap_threshold_ratio : float
            A fraction (between 0 and 1) of the maximum allowed overlap between cells.
        overlap_threshold_tfg : float
            Overlap threshold used in the TFG.
        delta_t : float
            The time interval used to move the cells.
        initial_apect_ratio : float
            The aspect_ratio of all cells in the culture at the begining of the simulation.
        aspect_ratio_max : float
            The max value of the aspect ratio that a cell can have after deforms.
        cell_speed_max : float
            The max value of speed that an elongated cell can achieve.
        delta_aspect_ratio : float
            Increase in the aspect ratio during deformation. If trabajo_final is True, then
            delta_aspect_ratio = aspect_ratio_max - 1
        trabajo_final : bool
            Flag to determine wether to use or not mechanism of the TFG.
        initialization_mode: str
            String to determine the initial conditions to use.
        initial_positions: Optional[np.ndarray] = None
            Initial positions for the case of triangular lattice.
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
        contraction_overlap_safety_ratio : float or None
            Maximum normalized overlap allowed after an instantaneous
            contraction. Set to None to disable the safety check.

        Attributes
        ----------
        force : Force
            The force used in the interaction between cells.
        cell_max_repro_attempts : int
            Maximum number of reproduction attempts a cell can make.
        cell_max_def_attempts : int
            Maximum number of deformation attempts a cell can make.
        adjacency_threshold : int
            The maximum distance at which two cells can be considered
            neighbors.
        cell_radius : int
            The radius of a cell.
        prob_stem : float
            The probability that a cell becomes a stem cell.
        prob_diff : float
            The probability that a cell differentiates.
        swap_probability : float
            The probability that a cell swaps its type with its offspring.
        initial_number_of_cells : int, optional
            The initial number of cells in the culture.
        initial_fraction_elongated : float, optional
            The initial fraction of elongated cells.
        side : int, optional
            The length of the side of the square where the cells move.
        reproduction : bool
            Whether the cells reproduce or not
        movement : bool
            Whether the cells move or not
        deformation : bool
            Whether the cells deforms or not.
        cell_area : float
            The area of all cells in the culture.
        stabilization_time : int
            The time we have to wait in order to start the deformation
        overlap_threshold_ratio : float
            A fraction (between 0 and 1) of the maximum allowed overlap between cells.
        overlap_threshold_tfg : float
            Overlap threshold used in the TFG.
        delta_t : float
            The time interval used to move
        initial_apect_ratio : float
            the aspect_ratio of all cells in the culture at the begining of the simulation.
        aspect_ratio_max : float
            The max value of the aspect ratio that a cell can have after deforms
        cell_speed_max : float
            The max value of speed that an elongated cell can achieve.
        delta_aspect_ratio : float
            Increase in the aspect ratio during deformation.
        rng : numpy.random.Generator
            Random number generator.
        first_cell_is_stem : bool
            Whether the first cell is a stem cell or not.
        cell_positions : numpy.ndarray
            Matrix to store the positions of all cells in the culture.
        cell_phies : numpy.ndarray
            Matrix to store the orientations in the x-y plane of all cells in the culture.
        cells : list[Cell]
            List of all cells in the culture.
        active_cells : list[Cell]
            List of all active cells in the culture.
        """
        # cell attributes
        self.force = force
        self.cell_max_repro_attempts = cell_max_repro_attempts
        self.cell_max_def_attempts = cell_max_def_attempts
        self.adjacency_threshold = adjacency_threshold
        self.cell_radius = cell_radius
        self.prob_stem = prob_stem
        self.prob_diff = prob_diff
        self.swap_probability = swap_probability
        self.initial_number_of_cells = initial_number_of_cells
        self.initial_fraction_elongated = initial_fraction_elongated
        self.reproduction = reproduction
        self.movement = movement
        self.deformation = deformation
        self.overlap_threshold_ratio = overlap_threshold_ratio
        self.overlap_threshold_tfg = overlap_threshold_tfg
        self.contraction_overlap_safety_ratio = contraction_overlap_safety_ratio
        self.delta_t = delta_t
        self.initial_aspect_ratio = initial_aspect_ratio
        self.aspect_ratio_max = aspect_ratio_max
        self.cell_speed_max = cell_speed_max
        self.stabilization_time = stabilization_time

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

        # Adaptive elongation state
        self.steps_without_deformation = 0
        self.elongation_sleep_remaining = 0

        # TFG
        self.trabajo_final = trabajo_final

        # delta aspect ratio
        if trabajo_final is True:
            self.delta_aspect_ratio = aspect_ratio_max-1
        else:
            self.delta_aspect_ratio = delta_aspect_ratio
        # we instantiate the culture's RNG with the provided entropy
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)

        # state whether this is a csc-seeded culture
        self.first_cell_is_stem = first_cell_is_stem

        # initialize the positions matrix
        self.cell_positions = np.empty((0, 3), float)

        # Instantaneous velocity resulting from the cell dynamics
        self.cell_instantaneous_velocities = np.empty(
            (0, 3),
            dtype=float,
        )

        # the phies matrix
        self.cell_phies = np.array([])

        # and the nematic tensors matrix
        self.nematic_tensors = np.empty((0, 3, 3), float) 

        # we initialize the lists of cells
        self.cells = []
        self.active_cell_indexes = []

        # time at wich the culture was created
        self.simulation_start = self._get_simulation_time()

        # Additional objects
        self.output = output
        self.grid = grid

        # Deformation events accumulated since the previous output
        self.reset_deformation_event_counts()

        # First timestep included in the current deformation interval
        self.deformation_interval_start_tic = 1

        # we set the grid's culture to this one
        self.grid.culture = self

        # calculation of the side of the culture using other parameters
        self.side = self.grid.bounds
        # and calculation of the cells_area given the radius
        self.cell_area = np.pi*self.cell_radius**2

        # Initial-condition protocol
        self.initialization_mode = initialization_mode

        if initial_positions is None:
            self.initial_positions = None

        else:
            self.initial_positions = np.asarray(
                initial_positions,
                dtype=float,
            ).copy()

    # ----------------database related behavior----------------

    def _get_simulation_time(self):
        # we get the current date and time
        current_time = datetime.now()
        # we format the string
        time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
        return time_string

    # ------------------cell related behavior------------------

    def generate_new_position(self, cell_index: int) -> np.ndarray:
        """
        Generate a proposed position for the child, adjacent to the given one.

        A new position for the child cell is randomly generated, at a distance
        equals to two times the radius of a cell (all cells are assumed to
        have the same radius). This is done by randomly choosing the angular
        spherical coordinates from a uniform distribution. It uses the cell
        current position and its radius.

        Returns
        -------
        new_position : numpy.ndarray
            A 3D vector representing the new position of the cell.

        Notes
        -----
        - All cells are assumed to have the same radius.
        - To get a uniform distribution of points in the unit sphere, we have
        to choose cos(theta) uniformly in [-1, 1] instead of theta uniformly
        in [0, pi].
        """
        cos_theta = self.rng.uniform(low=-1, high=1)
        theta = np.arccos(cos_theta)  # Convert cos(theta) to theta
        phi = self.rng.uniform(low=0, high=2 * np.pi)

        x = 2 * self.cell_radius * np.sin(theta) * np.cos(phi)
        y = 2 * self.cell_radius * np.sin(theta) * np.sin(phi)
        z = 2 * self.cell_radius * np.cos(theta)
        cell_position = self.cell_positions[cell_index]
        new_position = cell_position + np.array([x, y, z])
        return new_position

    def reproduce(self, cell_index: int, tic: int) -> None:
        """The given cell reproduces, generating a new child cell.

        Attempts to create a new cell in a random position, adjacent to the
        current cell, if the cell has available space. If the cell fails to
        find a position that doesn't overlap with existing cells, (for the
        estabished maximum number of attempts), no new cell is created, and
        the current one is deactivated. This means that we set its available
        space to `False` and remove it from the list of active cells.

        Notes
        -----
        The `if cell.available_space` might be redundant since we remove the
        cells from the `active_cells` list when seting that to `False`, but
        the statement is kept as a way of double checking.
        """
        cell = self.cells[cell_index]

        if cell.available_space:
            for attempt in range(self.cell_max_repro_attempts):
                # we generate a new proposed position for the child cell
                child_position = self.generate_new_position(cell_index)

                # if the position is not within the bounds of the simulation
                # we get the corresponding position
                if not self.grid.is_position_in_bounds(child_position):
                    child_position = self.grid.get_in_bounds_position(
                        child_position
                    )

                # set of all existing cell indexes that would neighbor the new
                # cell
                neighbor_indices = list(
                    self.grid.find_neighbors(
                        position=child_position,
                    )
                )
                # modifies the set in-place to remove the parent cell index
                neighbor_indices.remove(cell_index)

                # array with the distances from the proposed child position to
                # the other cells
                if len(neighbor_indices) > 0:
                    neighbor_position_mat = self.cell_positions[
                        neighbor_indices, :
                    ]
                    distance = np.linalg.norm(
                        child_position - neighbor_position_mat, axis=1
                    )
                else:
                    distance = np.array([])

                # boolean array specifying if there is no overlap between
                # the proposed child position and the other cells
                no_overlap = np.all(distance >= 2 * self.cell_radius)
                # if it is true that there is no overlap for
                # every element of the array, we break the loop
                if no_overlap:
                    break

            # if there was no overlap, we create a child in that position
            # if not, we do nothing but specifying that there is no available
            # space
            if no_overlap:
                # we create a child in that position
                if cell.is_stem:
                    random_number = self.rng.random()
                    if random_number <= self.prob_stem:  # ps
                        child_cell = Cell(
                            position=child_position,
                            culture=self,
                            is_stem=True,
                            parent_index=cell_index,
                            creation_time=tic,
                        )
                    else:
                        child_cell = Cell(
                            position=child_position,
                            culture=self,
                            is_stem=False,
                            parent_index=cell_index,
                            creation_time=tic,
                        )
                        if random_number <= (
                            self.prob_stem + self.prob_diff
                        ):  # pd
                            cell.is_stem = False
                            self.output.record_stemness(
                                cell_index, tic, cell.is_stem
                            )
                        elif (
                            self.rng.random() <= self.swap_probability
                        ):  # pa = 1-ps-pd
                            cell.is_stem = False
                            self.output.record_stemness(
                                cell_index, tic, cell.is_stem
                            )
                            child_cell.is_stem = True
                            self.output.record_stemness(
                                child_cell._index, tic, child_cell.is_stem
                            )
                else:
                    child_cell = Cell(
                        position=child_position,
                        culture=self,
                        is_stem=False,
                        parent_index=cell_index,
                        creation_time=tic,
                    )
            else:
                # The cell has no available space to reproduce
                cell.available_space = False
                # We no longer consider it active, so we remove *all* of its
                # instances from the list of active cell indexes
                set_of_current_active_cells = set(self.active_cell_indexes)
                set_of_current_active_cells.discard(cell_index)
                self.active_cell_indexes = list(set_of_current_active_cells)
                # We record the deactivation
                self.output.record_deactivation(cell_index, tic)
                # if there was no available space, we turn off reproduction
                # and record the change in the Cells table of the DataBase
        # else:
        #     pass
        # if the cell's neighbourhood is already full, we do nothing
        # (reproduction is turned off)

    # --------------------------- Radiotherapy things ------------------------

    def realization_name(self) -> str:
        """Return the name of the realization."""
        name = (
            f"culture_pd={self.prob_diff}"
            f"_ps={self.prob_stem}"
            f"_rng_seed={self.rng_seed}"
        )
        return name

    def radiotherapy_w_susceptibility(self) -> None:
        """Simulate a radiotherapy session by assigning susceptibilities.

        This function simulates a radiotherapy session where, due to increased
        O2 consumption, the active cells are more sensitive to radiation than
        quiescent cells. The probability of survival is different for active
        and quiescent cells, by a factor beta. However, all of this is left
        for postprocessing, so data can be used both for the described
        situation, or for another one where the cells are killed with a
        probability that varies with their position.

        A pandas.DataFrame is generated and saved with the following columns:
        - the norm of the position of the cell
        - the cell's stemness
        - whether the cell is active
        - a “suceptibility” that will indicate whether the cell was killed
          given the survival ratio (in postprocessing).
        """
        # we make the dictionary for the dataframe that will store the data
        susceptibility = self.rng.random(size=len(self.cells))
        norms = np.linalg.norm(self.cell_positions, axis=1)
        data = {
            "position_norm": norms,
            "stemness": [],
            "active": [],
            "susceptibility": susceptibility,
        }

        # we get the stemness, activity, and killing status of the cells
        for cell in self.cells:
            data["stemness"].append(cell.is_stem)
            data["active"].append(cell._index in self.active_cell_indexes)
            assert (
                cell._index in self.active_cell_indexes
            ) == cell.available_space

        # we make the dataframe
        df = pd.DataFrame(data, index=False)

        # we save the dataframe to a file
        filename = (
            f"radiotherapy_active_targeted_{self.realization_name()}.csv"
        )
        df.to_csv(filename)

    # ------------------movement related behavior------------------
    def calculate_relative_positions(
        self, 
        cell_position: np.ndarray, 
        neighbor_positions: np.ndarray
    ) -> np.ndarray:
        """
        It calculates the relative position in x and y of q cell with every neighbor 
        taking into account that they move in a box with periodic boundary conditions.

        Parameters
        ----------
        cell_position : np.ndarray
            The position of the cell.
        neighbor_positions : np.ndarray
            An array with all the positions of the neighbors.

        Returns
        -------
        relative_pos : np.ndarray
            The relative position of the cell with every neighbor.
        """
        
        # Calculate the relative positions between all the neighbors and the cell
        relative_positions = cell_position - neighbor_positions

        # Calculate the absolut distances
        abs_rx = np.abs(relative_positions[:, 0])
        abs_ry = np.abs(relative_positions[:, 1])

        # Choose the distance between two cells as the shortest distance taking into account the box
        # Create a mask that tells us if the distance is greater than half the side
        mask_x = abs_rx > 0.5 * self.side
        mask_y = abs_ry > 0.5 * self.side

        # For those True in masks, we adjust the position
        relative_positions[mask_x, 0] -= np.sign(relative_positions[mask_x, 0]) * self.side
        relative_positions[mask_y, 1] -= np.sign(relative_positions[mask_y, 1]) * self.side

        return relative_positions

    def calculate_overlaps(
        self,
        cell_index: int,
        neighbor_indices: np.ndarray,        # shape (N,)
        relative_positions: np.ndarray       # shape (N, 3)
    ) -> np.ndarray:                         # returns shape (N,)
        """
        Calculates the overlap between a single cell and multiple neighbors using
        overlap calculated in the TF in a vectorized way.

        Parameters
        ----------
        cell_index : int
            Index of the cell of reference.
        neighbor_indices : np.ndarray
            Indices of the neighboring cells.
        relative_positions : np.ndarray
            Array of relative positions of the cell with its neighbors (shape: (N, 3)).

        Returns
        -------
        overlaps : np.ndarray
            Array of overlaps with each neighbor.
        """
        cell = self.cells[cell_index]
        neighbor_indices = np.array(neighbor_indices, dtype=int)
        neighbors = [self.cells[i] for i in neighbor_indices]

        # extract phies
        phi_i = self.cell_phies[cell_index]
        phi_j = self.cell_phies[neighbor_indices]

        # diagonal terms
        d_i = cell.squared_diagonal
        d_j = np.array([neighbor.squared_diagonal for neighbor in neighbors])

        eps_i = cell.anisotropy
        eps_j = np.array([neighbor.anisotropy for neighbor in neighbors])

        cos_phi_diff = np.cos(phi_i - phi_j)

        beta = (
            (d_i + d_j)**2
            - (d_i * eps_i - d_j * eps_j)**2
            - 4 * d_i * d_j * eps_i * eps_j * (cos_phi_diff**2)
        )

        # get Q matrices
        Q_i = self.nematic_tensors[cell_index]           # shape (3,3)
        Q_j = self.nematic_tensors[neighbor_indices]     # shape (N,3,3)

        # matrix M
        M = (d_i * eps_i * Q_i + (d_j * eps_j)[:, None, None] * Q_j) / (d_i + d_j)[:, None, None]

        I = np.eye(3)
        diff_matrix = I - M                         # shape (N,3,3)

        # calculate i_0
        i_0 = 4 * self.cell_area**2 / (np.pi * np.sqrt(beta))  # shape (N,)

        # quadratic form: rᵀ (I - M) r, vectorized
        r = np.array(relative_positions)                      # shape (N,3)
        r_T = r[:, :, None]                         # shape (N,3,1)
        r_b = r[:, None, :]                         # shape (N,1,3)

        quad = np.matmul(r_b, np.matmul(diff_matrix, r_T)).reshape(-1)  # shape (N,)

        overlaps = i_0 * np.exp(-(d_i + d_j)/beta * quad)

        return overlaps

    def propose_new_position_to_deform(
        self, cell_index: int, new_phi: float, new_aspect_ratio: float
    ) -> np.ndarray:
        """Generate a proposed position for the cell, given a new phi and a new aspect
        ratio that help us to know if there is space available to deform the
        cell.

        Parameters
        ----------
        cell_index : int
            The index of the cell.
        new_phi : float
            The orientation of the new cell.
        new_aspect_ratio : float
            The aspect ratio of the new cell.

        Returns
        -------
        new_position : numpy.ndarray
            A 3D vector representing the new position of the cell.
        """
        # Calculate the major semi axis of the new cell
        new_semi_major_axis = np.sqrt(
            (self.cell_area * new_aspect_ratio) / np.pi
        )
        # and of the old cell
        old_semi_major_axis = np.sqrt(
            (self.cell_area * self.cells[cell_index].aspect_ratio) / np.pi
        )
        # Calculate the relative position of the old and new cells
        x = (new_semi_major_axis - old_semi_major_axis) * np.cos(new_phi)
        y = (new_semi_major_axis - old_semi_major_axis) * np.sin(new_phi)
        # Update the position
        new_position = self.cell_positions[cell_index] + np.array([x, y, 0])
        # Periodic boundary conditions
        new_position = np.mod(new_position, self.side)
        return new_position

    def calculate_max_overlaps(
        self,
        cell_index: int,
        neighbor_indices: list,        # shape (N,)
    ) -> np.ndarray:                         # returns shape (N,)
        """
        Calculates the maximum overlap between a single cell and multiple neighbors using
        overlap calculated in the TF in a vectorized way. (with the orientation of each cell)

        Parameters
        ----------
        cell_index : int
            Index of the cell of reference.
        neighbor_indices : np.ndarray
            Indices of the neighboring cells.

        Returns
        -------
        overlaps : np.ndarray
            Array of maximum overlaps with each neighbor.
        """
        cell = self.cells[cell_index]
        neighbor_indices = np.array(neighbor_indices, dtype=int)
        neighbors = [self.cells[i] for i in neighbor_indices]

        # extract phies
        phi_i = self.cell_phies[cell_index]
        phi_j = self.cell_phies[neighbor_indices]

        # diagonal terms
        d_i = cell.squared_diagonal
        d_j = np.array([neighbor.squared_diagonal for neighbor in neighbors])

        eps_i = cell.anisotropy
        eps_j = np.array([neighbor.anisotropy for neighbor in neighbors])

        cos_phi_diff = np.cos(phi_i - phi_j)

        beta = (
            (d_i + d_j)**2
            - (d_i * eps_i - d_j * eps_j)**2
            - 4 * d_i * d_j * eps_i * eps_j * (cos_phi_diff**2)
        )
        # finally we can calculate i_0
        # i_0 = (4*pi*l_par_k*l_perp_k*l_par_j*l_perp_j)/sqrt(beta)
        # with l_parallel = np.sqrt((cell_area*cell.aspect_ratio)/np.pi)
        # and l_perp = sqrt(cell_area/(np.pi*cell.aspect_ratio))
        max_overlap = 4 * self.cell_area**2 / (np.pi * np.sqrt(beta))
        
        return max_overlap

    def update_nematic_tensors(self, cell_indices: np.ndarray = None) -> None:
        """
        Updates the nematic tensors Q for the specified subset of cells.
        If no indices are provided, all tensors are updated.

        Parameters:
        -----------
        indices : np.ndarray or list of int, optional
            Indices of the cells to update. If None, updates all cells.
        """
        if cell_indices is None:
            cell_indices = np.arange(len(self.active_cell_indexes))

        phies = self.cell_phies[cell_indices]
        cos2 = np.cos(2 * phies)
        sin2 = np.sin(2 * phies)

        Q = np.zeros((len(cell_indices), 3, 3))
        Q[:, 0, 0] = cos2
        Q[:, 0, 1] = sin2
        Q[:, 1, 0] = sin2
        Q[:, 1, 1] = -cos2

        self.nematic_tensors[cell_indices] = Q


    def reset_deformation_event_counts(
        self,
    ) -> None:
        """
        Reset the deformation-event counters for a new recording interval.
        """
        self.deformation_event_counts = {
            "round_elongation_attempts": 0,
            "round_elongation_successes": 0,
            "elliptical_elongation_attempts": 0,
            "elliptical_elongation_successes": 0,
            "contraction_events": 0,
            "contraction_to_round_events": 0,
            "contraction_overlap_rejections": 0,
            "max_contraction_proposed_overlap": 0.0,
            "max_contraction_accepted_overlap": 0.0,
        }

    def _update_deformation_interval_max(
        self,
        key: str,
        value: float,
    ) -> None:
        """Update one maximum accumulated during the output interval."""
        self.deformation_event_counts[key] = max(
            self.deformation_event_counts.get(
                key,
                0.0,
            ),
            float(value),
        )

    def elongate_from_round(self, cell_index: int) -> bool:
        """If the cell is round, an angle is chosen randomly.
        If the new cell with these angle and an increment in the
        aspect ratio does not overlap with others, it remains.
        If not, try again up to cell_max_def_attempts.
        If it fails to deform, it remains as it was originally.

        Parameters
        ----------
        cell_index : int
            The index of the cell.

        Returns
        ----------
        succesful_elongation : bool
            True if the elongation was successful, False otherwise.
        """
        cell = self.cells[cell_index]

        # Number of attempts
        n_attempts = self.cell_max_def_attempts

        # we save the old attributes
        old_position = np.array(self.cell_positions[cell_index])
        old_phi = self.cell_phies[cell_index]
        old_aspect_ratio = cell.aspect_ratio
        # and get the place of the grid that correspond to the cell
        old_index = self.grid.get_hash_key(old_position)
        # create a dict that contains the total overlap of the cell with others
        total_overlap = dict()
        for attempt in range(n_attempts):
            # random phi and new aspect ratio and generate a position with them
            new_phi = self.rng.uniform(low=0, high=2 * np.pi)
            new_aspect_ratio = old_aspect_ratio + self.delta_aspect_ratio
            new_position = self.propose_new_position_to_deform(
                cell_index, new_phi, new_aspect_ratio
            )
            # updating attributes
            self.cell_positions[cell_index] = new_position
            self.cell_phies[cell_index] = new_phi
            self.update_nematic_tensors([cell_index])
            cell.set_aspect_ratio(new_aspect_ratio)
            # list of neighbors
            candidate_neighbors = list(
                self.grid.find_neighbors(
                    position=new_position,
                )
            )
            # modifies the set in-place to remove the actual cell index
            candidate_neighbors.remove(cell_index)

            if not candidate_neighbors:
                # If there are no neighbors, total_overlap = 0
                total_overlap[(new_phi, tuple(new_position))] = 0
            else:
                # Calculate relative positions for all neighbors
                relative_positions = self.calculate_relative_positions(
                    self.cell_positions[cell_index],
                    np.array([self.cell_positions[i] for i in candidate_neighbors])
                )

                # TFG criterion: only consider neighbors whose distance
                # is smaller than the sum of the major semi-axes
                if self.trabajo_final:

                    cell_semi_major_axis = np.sqrt(
                        (self.cell_area * cell.aspect_ratio) / np.pi
                    )

                    distances = np.linalg.norm(
                        relative_positions,
                        axis=1,
                    )

                    neighbor_semi_major_axes = np.sqrt(
                        self.cell_area
                        * np.array(
                            [
                                self.cells[i].aspect_ratio
                                for i in candidate_neighbors
                            ]
                        )
                        / np.pi
                    )

                    distance_mask = (
                        distances
                        <= (
                            cell_semi_major_axis
                            + neighbor_semi_major_axes
                        )
                    )

                    candidate_neighbors = list(
                        np.array(candidate_neighbors)[distance_mask]
                    )

                    relative_positions = relative_positions[
                        distance_mask
                    ]

                # Calculate overlaps
                overlaps = self.calculate_overlaps(cell_index, candidate_neighbors, relative_positions)

                # Filter neighbors
                if self.trabajo_final:

                    # TFG criterion
                    mask = overlaps > self.overlap_threshold_tfg

                else:

                    # Calculate max overlaps
                    max_overlaps = self.calculate_max_overlaps(
                        cell_index,
                        candidate_neighbors,
                    )

                    mask = (
                        overlaps
                        > (
                            self.overlap_threshold_ratio
                            * max_overlaps
                        )
                    )

                # Sum total overlap if there is no significant overlap
                if not mask.any():
                    total_overlap[(new_phi, tuple(new_position))] = np.sum(overlaps)

            # Restore original values
            self.cell_positions[cell_index] = old_position
            self.cell_phies[cell_index] = old_phi
            self.update_nematic_tensors([cell_index])
            cell.set_aspect_ratio(old_aspect_ratio)

        # Check if total_overlap is not empty (else, pass)
        if total_overlap:
            # get the minimum overlap value
            min_overlap = min(total_overlap.values())
            # find all angles and positions with the minimum overlap
            min_angles_positions = [key for key, overlap in total_overlap.items() if overlap == min_overlap]
            # choose a random key from those with the minimum overlap
            #chosen_key = self.rng.choice(min_angles_positions)
            chosen_key = self.rng.choice(np.array(min_angles_positions, dtype=object))
            chosen_phi = chosen_key[0]
            chosen_position = np.array(chosen_key[1]) 
            # and set the new values of aspect ratio, position and orientation
            cell.set_aspect_ratio(old_aspect_ratio + self.delta_aspect_ratio)
            self.cell_phies[cell_index] = chosen_phi
            self.update_nematic_tensors([cell_index])
            self.cell_positions[cell_index] = chosen_position
            # and calculate the new place in the grid
            new_index = self.grid.get_hash_key(chosen_position)
            succesful_elongation = True
            if old_index != new_index:
                self.grid.remove_cell_from_hash_table(cell_index, old_position)
                self.grid.add_cell_to_hash_table(cell_index, chosen_position)
        else:
            succesful_elongation = False

        return succesful_elongation

    def elongate_from_elliptical(self, cell_index: int) -> bool:
        """If the cell is round, an angle is chosen randomly.
        If the new cell with these angle and aspect ratio = maximum (given as an
        attribute) does not overlap with others, it remains.
        If not, try again up to cell_max_def_attempts.
        If it fails to deform, it remains as it was originally.

        Parameters
        ----------
        cell_index : int
            The index of the cell.

        Returns
        ----------
        succesful_elongation : bool
            True if the elongation was successful, False otherwise.
        """
        cell = self.cells[cell_index]
        # we save the old attributes
        old_position = np.array(self.cell_positions[cell_index])
        old_aspect_ratio = cell.aspect_ratio
        # and get the place of the grid that correspond to the cell
        old_index = self.grid.get_hash_key(old_position)
   
        # random phi and aspect ratio=max and generate a position with them
        #new_phi = self.rng.uniform(low=0, high=2 * np.pi)
        new_aspect_ratio = min(
            old_aspect_ratio + self.delta_aspect_ratio,
            self.aspect_ratio_max
        )
        new_position = self.propose_new_position_to_deform(
            cell_index, self.cell_phies[cell_index], new_aspect_ratio
        )
        # updating attributes
        self.cell_positions[cell_index] = new_position
        cell.set_aspect_ratio(new_aspect_ratio)
        # and calculate the new place in the grid
        new_index = self.grid.get_hash_key(new_position)
        candidate_neighbors = list(
            self.grid.find_neighbors(
                position=new_position,
            )
        )
        # modifies the set in-place to remove the actual cell index
        candidate_neighbors.remove(cell_index)

        if not candidate_neighbors:
            no_overlap = True
        else:
            # Calculate relative positions for all neighbors
            relative_positions = self.calculate_relative_positions(
                self.cell_positions[cell_index],
                np.array([self.cell_positions[i] for i in candidate_neighbors])
            )
            # Vectorized overlap + threshold check
            overlaps = self.calculate_overlaps(cell_index, candidate_neighbors, relative_positions)
            max_overlaps = self.calculate_max_overlaps(cell_index, candidate_neighbors)
            mask = overlaps > self.overlap_threshold_ratio * max_overlaps

            # If there is overlap, turn back to original values
            if np.any(mask):
                self.cell_positions[cell_index] = old_position
                cell.set_aspect_ratio(old_aspect_ratio)
                no_overlap = False
            else:
                no_overlap = True

        if no_overlap:
            # if there is no overlap, the new cell remains and we finish the loop
            succesful_elongation = True
            # if we have change the index, the candidate for neighbors also change
            # Update the index of the cell if necessary
            if old_index != new_index:
                self.grid.remove_cell_from_hash_table(cell_index, old_position)
                self.grid.add_cell_to_hash_table(cell_index, new_position)
        else:
            succesful_elongation = False

        return succesful_elongation

    def _calculate_max_normalized_overlap(
        self,
        cell_index: int,
    ) -> float:
        """Return the largest normalized overlap around one cell.

        The cell must already have the proposed shape when this method is
        called.
        """
        # Find possible neighbors using the spatial hash grid
        candidate_neighbors = [
            neighbor_index
            for neighbor_index in self.grid.find_neighbors(
                position=self.cell_positions[cell_index],
            )
            if neighbor_index != cell_index
        ]

        if not candidate_neighbors:
            return 0.0

        # Calculate relative positions for all neighbors
        relative_positions = self.calculate_relative_positions(
            self.cell_positions[cell_index],
            self.cell_positions[candidate_neighbors],
        )

        # Calculate overlaps and max overlaps
        overlaps = self.calculate_overlaps(
            cell_index=cell_index,
            neighbor_indices=candidate_neighbors,
            relative_positions=relative_positions,
        )

        # Calculate max overlaps
        max_overlaps = self.calculate_max_overlaps(
            cell_index=cell_index,
            neighbor_indices=candidate_neighbors,
        )

        # Calculate normalized overlaps, avoiding division by zero
        normalized_overlaps = np.divide(
            overlaps,
            max_overlaps,
            out=np.zeros_like(overlaps),
            where=max_overlaps > 0,
        )

        # Return the maximum normalized overlap
        return float(np.max(normalized_overlaps))

    def shrink_from_elliptical(
        self,
        cell_index: int,
    ) -> bool:
        """
        Attempt to contract an elliptical cell.

        The overlap introduced by the proposed contraction is always
        measured. If a contraction overlap safety ratio is provided,
        contractions that exceed it are rejected.

        Parameters
        ----------
        cell_index : int
            Index of the cell.

        Returns
        -------
        successful_shrinking : bool
            True if the contraction was accepted, False otherwise.
        """
        cell = self.cells[cell_index]

        # The force did not request a contraction
        if not cell.shrink:
            return False

        # Calculate the aspect ratio after the proposed contraction
        old_aspect_ratio = cell.aspect_ratio

        new_aspect_ratio = max(
            old_aspect_ratio
            - self.delta_aspect_ratio,
            1.0,
        )

        # Temporarily apply the proposed shape
        cell.set_aspect_ratio(
            new_aspect_ratio,
        )

        # Measure the maximum normalized overlap produced by the
        # proposed contraction
        proposed_max_normalized_overlap = (
            self._calculate_max_normalized_overlap(
                cell_index=cell_index,
            )
        )

        # Update the maximum overlap proposed
        self._update_deformation_interval_max(
            key="max_contraction_proposed_overlap",
            value=proposed_max_normalized_overlap,
        )

        # Reject the proposed contraction if the safety criterion
        # is enabled and its threshold is exceeded
        safety_check_is_enabled = (
            self.contraction_overlap_safety_ratio
            is not None
        )
        if (
            safety_check_is_enabled
            and proposed_max_normalized_overlap
            > self.contraction_overlap_safety_ratio
        ):
            self.deformation_event_counts[
                "contraction_overlap_rejections"
            ] = (
                self.deformation_event_counts.get(
                    "contraction_overlap_rejections",
                    0,
                )
                + 1
            )

            # Restore the previous shape
            cell.set_aspect_ratio(
                old_aspect_ratio,
            )

            cell.shrink = False
            return False

        # The proposed contraction was accepted
        self._update_deformation_interval_max(
            key="max_contraction_accepted_overlap",
            value=proposed_max_normalized_overlap,
        )

        # A completely round cell has no meaningful orientation
        if np.isclose(
            new_aspect_ratio,
            1.0,
        ):
            self.cell_phies[cell_index] = 0.0

            self.update_nematic_tensors(
                [cell_index],
            )

        # Finalize the accepted contraction
        cell.shrink = False
        return True

    def _get_significant_neighbors(
        self,
        cell_index: int,
    ) -> np.ndarray:
        """
        Return the indices of cells that significantly interact with one cell.

        The interaction criterion is exactly the same one used by the dynamics:

        - For the TFG model, neighbors must satisfy the major-semi-axis
        distance criterion and the absolute overlap threshold.
        - Otherwise, the overlap must exceed a fixed fraction of the
        maximum overlap.

        Notes
        -----
        The dictionaries ``neighbors_relative_pos`` and ``neighbors_overlap``
        are used as temporary symmetric caches. When a quantity is calculated
        for a pair of cells, it is stored for both cells so that it does not
        need to be recalculated when the other cell is processed.
        """
        cell = self.cells[cell_index]

        # Find possible neighbors using the spatial hash grid
        candidate_neighbors = [
            neighbor_index
            for neighbor_index in self.grid.find_neighbors(
                position=self.cell_positions[cell_index],
            )
            if neighbor_index != cell_index
        ]

        # Identify neighbors whose relative positions have not yet
        # been calculated during this timestep
        to_calculate_relative_pos = [
            neighbor_index
            for neighbor_index in candidate_neighbors
            if neighbor_index not in cell.neighbors_relative_pos
        ]
        # Calculate the relative position to them
        if to_calculate_relative_pos:
            neighbor_positions = self.cell_positions[
                to_calculate_relative_pos
            ]

            relative_positions = self.calculate_relative_positions(
                self.cell_positions[cell_index],
                neighbor_positions,
            )

            # Store each result for both cells in the pair.
            for neighbor_index, relative_position in zip(
                to_calculate_relative_pos,
                relative_positions,
            ):
                cell.neighbors_relative_pos[
                    neighbor_index
                ] = relative_position

                self.cells[
                    neighbor_index
                ].neighbors_relative_pos[
                    cell_index
                ] = -relative_position

        # In the TFG model, discard candidate neighbors that are farther
        # apart than the sum of their major semi-axes
        if self.trabajo_final and candidate_neighbors:
            cell_semi_major_axis = np.sqrt(
                (
                    self.cell_area
                    * cell.aspect_ratio
                )
                / np.pi
            )

            distances = np.array([
                np.linalg.norm(
                    cell.neighbors_relative_pos[neighbor_index]
                )
                for neighbor_index in candidate_neighbors
            ])

            neighbor_semi_major_axes = np.sqrt(
                (
                    self.cell_area
                    * np.array([
                        self.cells[
                            neighbor_index
                        ].aspect_ratio
                        for neighbor_index in candidate_neighbors
                    ])
                )
                / np.pi
            )

            distance_mask = (
                distances
                <= (
                    cell_semi_major_axis
                    + neighbor_semi_major_axes
                )
            )

            candidate_neighbors = list(
                np.asarray(
                    candidate_neighbors,
                    dtype=int,
                )[distance_mask]
            )

        # Identify overlaps that have not yet been calculated
        to_calculate_overlap = [
            neighbor_index
            for neighbor_index in candidate_neighbors
            if neighbor_index not in cell.neighbors_overlap
        ]
        # Calculate them
        if to_calculate_overlap:
            relative_positions_overlap = [
                cell.neighbors_relative_pos[neighbor_index]
                for neighbor_index in to_calculate_overlap
            ]

            overlaps = self.calculate_overlaps(
                cell_index=cell_index,
                neighbor_indices=to_calculate_overlap,
                relative_positions=relative_positions_overlap,
            )

            # Store each overlap for both cells in the pair
            for neighbor_index, overlap in zip(
                to_calculate_overlap,
                overlaps,
            ):
                cell.neighbors_overlap[
                    neighbor_index
                ] = overlap

                self.cells[
                    neighbor_index
                ].neighbors_overlap[
                    cell_index
                ] = overlap

        # Get the indices and overlap calculated of the neighbors
        neighbor_indices = np.asarray(
            list(cell.neighbors_overlap.keys()),
            dtype=int,
        )
        overlaps = np.asarray(
            list(cell.neighbors_overlap.values()),
            dtype=float,
        )

        # A cell with no significant candidates has no interacting neighbors
        if neighbor_indices.size == 0:
            return np.empty(
                0,
                dtype=int,
            )

        # Filter with the mask
        if self.trabajo_final:
            significant_mask = (
                overlaps
                > self.overlap_threshold_tfg
            )

        else:
            max_overlaps = self.calculate_max_overlaps(
                cell_index,
                neighbor_indices,
            )

            significant_mask = (
                overlaps
                > (
                    self.overlap_threshold_ratio
                    * max_overlaps
                )
            )

        return neighbor_indices[significant_mask]

    def interaction(self, cell_index: int, delta_t: float) -> Tuple[np.ndarray, float]:
        """The given cell interacts with others if they are close enough.

        It describes the interaction of the cells given a force. It changes the position
        of the cell (because of the forces exerted by all the other cells and the
        intrinsic velocity) and it's angle in the x-y plane, phi (because of a torque).

        Parameters
        ----------
        cell_index : int
            The index of the cell.
        delta_t : float
            The time step.

        Returns
        -------
        dif_position : np.ndarray
            The change in the position of the cell.
        dif_phi : float
            The change in the angle phi of the cell.
        -----
        """
        cell = self.cells[cell_index]
        
        significant_neighbors_indexes = (
            self._get_significant_neighbors(
                cell_index=cell_index,
            )
        )

        # Calculate interaction with final neighbors
        dif_position, dif_phi = self.force.calculate_interaction(
            self.cells,
            self.cell_phies,
            cell_index,
            delta_t,
            self.cell_area,
            significant_neighbors_indexes,
            self.nematic_tensors,
        )

        # Reset the neighbor dictionaries to empty
        cell.neighbors_relative_pos.clear()
        cell.neighbors_overlap.clear()

        # Return the change in the position and in the phi angle of the cell
        return dif_position, dif_phi

    def calculate_clusters(
        self,
    ) -> dict[str, list[list[int]]]:
        """
        Calculate the connected clusters of round and elongated cells.

        Two cells belong to the same cluster when:

        1. They interact significantly according to the same criterion
        used by the dynamics.
        2. They have the same phenotype: both are round or both are
        elongated.

        Connectivity is transitive. Therefore, if cell A interacts with B
        and B interacts with C, all three cells belong to the same cluster,
        even if A and C do not interact directly.

        The complete cluster structure is recalculated independently for
        every snapshot. No cluster labels or connections are preserved
        between consecutive calls.

        Returns
        -------
        clusters : dict[str, list[list[int]]]
            Dictionary containing two entries:

            - ``"round"``: clusters composed of round cells.
            - ``"elongated"``: clusters composed of elongated cells.

            Each cluster is represented by a list containing the indices
            of its cells. Isolated cells appear as clusters of size one.
        """
        number_of_cells = len(self.cells)

        if number_of_cells == 0:
            return {
                "round": [],
                "elongated": [],
            }

        # Start from a completely new connectivity structure.
        union_find = _UnionFind(number_of_cells)

        # Process the interaction network of the current snapshot.
        for cell_index in range(number_of_cells):
            cell = self.cells[cell_index]

            significant_neighbors = (
                self._get_significant_neighbors(
                    cell_index=cell_index,
                )
            )

            # In the current model, a cell is round when its aspect ratio
            # is numerically equal to one. All other cells are elongated.
            cell_is_round = np.isclose(
                cell.aspect_ratio,
                1.0,
            )

            for neighbor_index in significant_neighbors:
                neighbor_index = int(neighbor_index)
                neighbor = self.cells[neighbor_index]

                neighbor_is_round = np.isclose(
                    neighbor.aspect_ratio,
                    1.0,
                )

                same_phenotype = (
                    cell_is_round
                    == neighbor_is_round
                )

                # Only interacting cells of the same phenotype are joined.
                if same_phenotype:
                    union_find.union(
                        cell_index,
                        neighbor_index,
                    )

            # The temporary cached data of the processed cell are no
            # longer needed.
            #
            # Cached values stored in cells that have not yet been
            # processed remain available and can be reused when those
            # cells are visited.
            cell.neighbors_relative_pos.clear()
            cell.neighbors_overlap.clear()

        # Transform the internal Union-Find representation into explicit
        # lists containing the indices of the cells in every cluster.
        connected_components = union_find.groups()

        round_clusters = []
        elongated_clusters = []

        for cluster_indices in connected_components.values():
            # A component cannot contain both phenotypes because union()
            # was only called between cells of the same phenotype.
            representative_index = cluster_indices[0]

            representative_is_round = np.isclose(
                self.cells[
                    representative_index
                ].aspect_ratio,
                1.0,
            )

            if representative_is_round:
                round_clusters.append(
                    list(cluster_indices)
                )
            else:
                elongated_clusters.append(
                    list(cluster_indices)
                )

        return {
            "round": round_clusters,
            "elongated": elongated_clusters,
        }


    def move(
        self,
        dif_positions: np.ndarray,
        dif_phies: np.ndarray,
    ) -> None:
        """The given cell moves with a given velocity and changes its orientation.
 
        Attempts to move one step with a particular velocity and changes its orientation.
        If the cell arrives to a border of the culture's square, it appears on the other
        side (periodic boundary conditions).

        Parameters
        ----------
        dif_positions : np.ndarray
            Matrix that contains the changes in position of all the cells.
        dif_phies : np.ndarray
            Matrix that contains the changes in orientation of all the cells.
        -----
        """
        # Copy the positions of the cells
        old_positions = self.cell_positions.copy()
        # Updating the cell's position
        self.cell_positions = self.cell_positions + dif_positions

        # and the angle
        self.cell_phies = self.cell_phies + dif_phies
        # Update the nematic tensors
        self.update_nematic_tensors()
        # Enforcing boundary condition
        self.cell_positions = np.mod(self.cell_positions, self.side)

        # Remove the cells from their old place in grid and add them to their 
        # new place 
        for cell_index in self.active_cell_indexes:
            old_key = self.grid.get_hash_key(old_positions[cell_index])
            new_key = self.grid.get_hash_key(self.cell_positions[cell_index])
            if old_key != new_key:
                self.grid.remove_cell_from_hash_table(cell_index, old_positions[cell_index])
                self.grid.add_cell_to_hash_table(cell_index, self.cell_positions[cell_index])


    def _record_clusters_if_needed(
        self,
        tic: int,
        final_tic: int,
    ) -> None:
        """
        Calculate and record clusters when required by the active outputs.

        The clusters are calculated using the current cell positions,
        phenotypes and interaction network.
        """
        if not self.output.should_record_clusters(
            tic=tic,
            final_tic=final_tic,
        ):
            return

        clusters = self.calculate_clusters()

        self.output.record_cluster_state(
            tic=tic,
            final_tic=final_tic,
            cells=self.cells,
            cell_positions=self.cell_positions,
            cell_phies=self.cell_phies,
            cell_instantaneous_velocities=(
                self.cell_instantaneous_velocities
            ),
            clusters=clusters,
            side=self.side,
        )

    def _record_local_order_if_needed(
        self,
        tic: int,
        final_tic: int,
    ) -> None:
        """
        Calculate and record local order parameters when required
        by the active outputs.

        The local observables are calculated using the current cell
        positions, phenotypes and orientations.
        """
        if not self.output.should_record_local_order(
            tic=tic,
            final_tic=final_tic,
        ):
            return

        self.output.record_local_order_state(
            tic=tic,
            final_tic=final_tic,
            cells=self.cells,
            cell_positions=self.cell_positions,
            cell_phies=self.cell_phies,
            side=self.side,
        )

    def _record_deformation_events_if_needed(
        self,
        tic: int,
        final_tic: int,
    ) -> None:
        """
        Record the deformation events accumulated during the current
        interval when required by the active outputs.

        After recording, reset the counters and begin a new interval.
        """
        if not self.output.should_record_deformation_events(
            tic=tic,
            final_tic=final_tic,
        ):
            return

        self.output.record_deformation_events(
            tic_start=self.deformation_interval_start_tic,
            tic_end=tic,
            final_tic=final_tic,
            event_counts=dict(
                self.deformation_event_counts
            ),
        )

        self.reset_deformation_event_counts()

        self.deformation_interval_start_tic = (
            tic + 1
        )

    # ---------------------------------------------------------

    def simulate(self, num_times: int, start_tic: int, checkpoint_path: str) -> None:
        """Simulate culture growth for a specified number of time steps.

        At each time step, we randomly sort the list of active cells and then
        we tell them to reproduce one by one.

        Parameters
        ----------
        num_times : int
            The number of time steps to simulate the cellular automaton.
        """
        # if the culture is brand-new, we create the tables of the DB and the
        # first cell
        if len(self.cells) == 0 and start_tic == 0:
            # we insert the register corresponding to this culture
            self.output.begin_culture(
                self.prob_stem,
                self.prob_diff,
                self.rng_seed,
                self.simulation_start,
                self.adjacency_threshold,
                self.swap_probability,
            )

            # we instantiate the first cell (only if reproduction)
            if self.reproduction:
                Cell(
                    position=np.array([0, 0, 0]),
                    culture=self,
                    is_stem=self.first_cell_is_stem,
                    parent_index=0,
                    available_space=True,
                )
            else:
                pass

            # We add all the cells in the case of movement
            if self.movement:
                # Calculate the number of initially elongated cells
                number_of_elongated_cells = int(
                    self.initial_number_of_cells
                    * self.initial_fraction_elongated
                )

                # Store the elongated indices in a boolean mask
                elongated_mask = np.zeros(
                    self.initial_number_of_cells,
                    dtype=bool,
                )

                # Take the elongated cells
                if number_of_elongated_cells > 0:
                    elongated_indices = self.rng.choice(
                        self.initial_number_of_cells,
                        size=number_of_elongated_cells,
                        replace=False,
                    )

                    elongated_mask[
                        elongated_indices
                    ] = True

                # Define the parameters if the cell is round or elongated
                for cell_index in range(
                    self.initial_number_of_cells
                ):
                    if elongated_mask[cell_index]:
                        phi = self.rng.uniform(
                            low=0,
                            high=2 * np.pi,
                        )

                        aspect_ratio = (
                            self.aspect_ratio_max
                        )

                    else:
                        phi = (
                            0
                            if np.isclose(
                                self.initial_aspect_ratio,
                                1.0,
                            )
                            else self.rng.uniform(
                                low=0,
                                high=2 * np.pi,
                            )
                        )

                        aspect_ratio = (
                            self.initial_aspect_ratio
                        )

                    # Take the positions of the cells depending on the
                    # initialization mode
                    if (
                        self.initialization_mode
                        == "random"
                    ):
                        position = np.array(
                            [
                                self.rng.uniform(
                                    low=0,
                                    high=self.side,
                                ),
                                self.rng.uniform(
                                    low=0,
                                    high=self.side,
                                ),
                                0,
                            ],
                            dtype=float,
                        )

                    else:
                        position = (
                            self.initial_positions[
                                cell_index
                            ].copy()
                        )

                    Cell(
                        position=position,
                        culture=self,
                        is_stem=(
                            self.first_cell_is_stem
                        ),
                        phi=phi,
                        aspect_ratio=aspect_ratio,
                        parent_index=0,
                        shrink=False,
                        available_space=True,
                    )

            self.cell_instantaneous_velocities = np.zeros_like(
                self.cell_positions,
                dtype=float,
            )

            # Save the data (for dat, ovito, and/or SQLite)
            self.output.record_culture_state(
                tic=0,
                cells=self.cells,
                cell_positions=self.cell_positions,
                cell_phies=self.cell_phies,
                active_cell_indexes=self.active_cell_indexes,
                side=self.side,
                cell_area=self.cell_area,
            )

            # Calculate and save local order parameters
            self._record_local_order_if_needed(
                tic=0,
                final_tic=num_times,
            )
            # Save the clusters data
            self._record_clusters_if_needed(
                tic=0,
                final_tic=num_times,
            )

        # we simulate for num_times time steps
        for i in range(start_tic+1, num_times + 1):
            # we reproduce and (or) move the cells
            if self.reproduction:
                # we get a permuted copy of the cells list
                active_cell_indexes = self.rng.permutation(
                    self.active_cell_indexes
                )
                # and reproduce the cells in this random order
                for index in active_cell_indexes:
                    self.reproduce(cell_index=index, tic=i)

            if self.movement:
                # We wait for the system to stabilize if neccessary
                if i > self.stabilization_time and self.deformation:
                    # Boolean to see if the elongation is sleeping
                    elongation_is_sleeping = (
                        i > self.deformation_warmup_steps
                        and self.elongation_sleep_remaining > 0
                    )

                    # Boolean to determine if any deformation occurred
                    deformation_occurred = False

                    # Run for every cell
                    active_cell_indexes = self.rng.permutation(
                        self.active_cell_indexes
                    )

                    for index in active_cell_indexes:
                        cell = self.cells[index]

                        if np.isclose(
                            cell.aspect_ratio,
                            1.0,
                        ):

                            # Round cells only try to elongate while elongation is active
                            if not elongation_is_sleeping:
                                # Add to the count of deformation events
                                self.deformation_event_counts[
                                    "round_elongation_attempts"
                                ] += 1

                                # Try to elongate it
                                success = self.elongate_from_round(
                                    index
                                )

                                if success:
                                    # Deformation succesful
                                    deformation_occurred = True

                                    self.deformation_event_counts[
                                        "round_elongation_successes"
                                    ] += 1

                        else:
                            # Contractions are always allowed, including during sleep
                            success = self.shrink_from_elliptical(
                                index
                            )

                            if success:
                                # Deformation succesful
                                deformation_occurred = True

                                self.deformation_event_counts[
                                    "contraction_events"
                                ] += 1

                                # Check if the final state is a round cell
                                if np.isclose(
                                    cell.aspect_ratio,
                                    1.0,
                                ):
                                    self.deformation_event_counts[
                                        "contraction_to_round_events"
                                    ] += 1

                            # If the cell cant shrink, it tries to elongate
                            elif (
                                not elongation_is_sleeping
                                and cell.aspect_ratio
                                < self.aspect_ratio_max
                            ):

                                self.deformation_event_counts[
                                    "elliptical_elongation_attempts"
                                ] += 1

                                success = self.elongate_from_elliptical(
                                    index
                                )

                                if success:
                                    # Deformation succesful
                                    deformation_occurred = True

                                    self.deformation_event_counts[
                                        "elliptical_elongation_successes"
                                    ] += 1

                    # Adaptive elongation starts only after the initial warmup
                    if i > self.deformation_warmup_steps:

                        if elongation_is_sleeping:
                            # A contraction changes the geometry, so elongation
                            # is reactivated from the next timestep
                            if deformation_occurred:
                                self.elongation_sleep_remaining = 0
                                self.steps_without_deformation = 0

                            else:
                                self.elongation_sleep_remaining -= 1

                        else:
                            # In the non sleeping phase, we see if there is
                            # a deformation
                            if deformation_occurred:
                                self.steps_without_deformation = 0
                            else:
                                self.steps_without_deformation += 1
                            # Deactivation of deformation
                            if (
                                self.steps_without_deformation
                                >= self.deformation_probe_steps
                            ):
                                self.elongation_sleep_remaining = (
                                    self.elongation_sleep_steps
                                )

                                self.steps_without_deformation = 0

                # We initialize the change in the position and angle of all cells
                dif_positions = np.zeros(
                    (len(self.cells), 3),
                    dtype=float,
                )

                dif_phies = np.zeros(
                    len(self.cells),
                    dtype=float,
                )
                # Calculate the interaction for every cell
                for index in self.active_cell_indexes:
                    dif_position, dif_phi = self.interaction(
                        cell_index=index, delta_t=self.delta_t,
                    )
                    # add the change in position to the matrix
                    dif_positions[index] = dif_position
                    # add the change in angle to the matrix
                    dif_phies[index] = dif_phi

                # Instantaneous resultant velocities
                self.cell_instantaneous_velocities = (
                    dif_positions
                    / self.delta_t
                )
                # Move all cells
                self.move(dif_positions=dif_positions, dif_phies=dif_phies)

            # Save the data (for dat, ovito, and/or SQLite)
            self.output.record_culture_state(
                tic=i,
                cells=self.cells,
                cell_positions=self.cell_positions,
                cell_phies=self.cell_phies,
                active_cell_indexes=self.active_cell_indexes,
                side=self.side,
                cell_area=self.cell_area,
            )

            # Calculate and save local order parameters only when requested
            self._record_local_order_if_needed(
                tic=i,
                final_tic=num_times,
            )

            # Calculate and save clusters only when requested
            self._record_clusters_if_needed(
                tic=i,
                final_tic=num_times,
            )

            # Save the deformation events accumulated during the
            # current recording interval.
            self._record_deformation_events_if_needed(
                tic=i,
                final_tic=num_times,
            )
            
            if checkpoint_path and i % 100 == 0:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                with open(checkpoint_path, "wb") as f:
                    #pickle.dump((self, i), f)
                    state = self.rng.bit_generator.state
                    pickle.dump((self, i, state), f)


        self.output.record_final_state(
            tic=num_times,
            cells=self.cells,
            cell_positions=self.cell_positions,
            active_cell_indexes=self.active_cell_indexes,
        )
