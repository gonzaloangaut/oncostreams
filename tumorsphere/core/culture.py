"""
Module containing the Culture class.

Classes:
    - Culture: Class that represents a culture of cells. Usually dependent
    on the Simulation class.
"""

# import os
from datetime import datetime
from typing import Set, Dict, List, Tuple


import pandas as pd
import numpy as np

from tumorsphere.core.cells import Cell
from tumorsphere.core.output import TumorsphereOutput
from tumorsphere.core.spatial_hash_grid import SpatialHashGrid
from tumorsphere.core.forces import Force


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
        reproduction: bool = False,
        movement: bool = True,
        deformation: bool = True,
        stabilization_time: int = 120,
        overlap_threshold_ratio: float = 0.35,
        delta_t: float = 0.05,
        initial_aspect_ratio: float = 1,
        aspect_ratio_max: float = 5,
        delta_aspect_ratio = 0.1,
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
        delta_t : float
            The time interval used to move the cells.
        initial_apect_ratio : float
            The aspect_ratio of all cells in the culture at the begining of the simulation.
        aspect_ratio_max : float
            The max value of the aspect ratio that a cell can have after deforms.
        delta_aspect_ratio : float
            Increase in the aspect ratio during deformation.


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
        delta_t : float
            The time interval used to move
        initial_apect_ratio : float
            the aspect_ratio of all cells in the culture at the begining of the simulation.
        aspect_ratio_max : float
            The max value of the aspect ratio that a cell can have after deforms
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
        self.reproduction = reproduction
        self.movement = movement
        self.deformation = deformation
        self.overlap_threshold_ratio = overlap_threshold_ratio
        self.delta_t = delta_t
        self.initial_aspect_ratio = initial_aspect_ratio
        self.aspect_ratio_max = aspect_ratio_max
        self.delta_aspect_ratio = delta_aspect_ratio
        self.stabilization_time = stabilization_time

        # we instantiate the culture's RNG with the provided entropy
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)

        # state whether this is a csc-seeded culture
        self.first_cell_is_stem = first_cell_is_stem

        # initialize the positions matrix
        self.cell_positions = np.empty((0, 3), float)

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

        # we set the grid's culture to this one
        self.grid.culture = self

        # calculation of the side of the culture using other parameters
        self.side = self.grid.bounds
        # and calculation of the cells_area given the radius
        self.cell_area = np.pi*self.cell_radius**2

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

    def relative_pos(self, cell_position: np.ndarray, neighbor_position: np.ndarray) -> np.ndarray:
        """
        It calculates the relative position in x and y of 2 cells taking into account
        that they move in a box with periodic boundary conditions.

        Parameters
        ----------
        cell_position : np.ndarray
            The position of the cell.
        neighbor_position : np.ndarray
            The position of the neighbor.

        Returns
        -------
        relative_pos : np.ndarray
            The relative position of the cell.
        """

        relative_pos_x = -(neighbor_position[0] - cell_position[0])
        relative_pos_y = -(neighbor_position[1] - cell_position[1])
        abs_rx = abs(relative_pos_x)
        abs_ry = abs(relative_pos_y)

        # we choose the distance between two cells as the shortest distance taking into account the box
        if abs_rx > 0.5 * self.side:
            relative_pos_x = np.sign(relative_pos_x) * (abs_rx - self.side)
        if abs_ry > 0.5 * self.side:
            relative_pos_y = np.sign(relative_pos_y) * (abs_ry - self.side)

        return np.array([relative_pos_x, relative_pos_y, 0])

    def calculate_overlap(
        self,
        cell_index: int,
        neighbor_index: int,
        relative_pos: np.ndarray,
    ) -> float:
        """
        Calculates the overlap between two cells using overlap calculated in the TF.

        Parameters
        ----------
        cell_index : int
            The index of the cell.
        neighbor_index : int
            The index of the neighbor.
        relative_pos : np.ndarray
            The relative position of the cells.

        Returns
        -------
        overlap : float
            The overlap between cells
        """
        cell = self.cells[cell_index]
        neighbor = self.cells[neighbor_index]

        # now we introduce the constant beta introduced by us in the TF
        beta = (
            (cell.squared_diagonal + neighbor.squared_diagonal) ** 2
            - (cell.squared_diagonal * cell.anisotropy - neighbor.squared_diagonal * neighbor.anisotropy) ** 2
            - 4
            * cell.squared_diagonal
            * cell.anisotropy
            * neighbor.squared_diagonal
            * neighbor.anisotropy
            * (
                np.cos(
                    self.cell_phies[cell_index]
                    - self.cell_phies[neighbor_index]
                )
            )
            ** 2
        )

        # then we obtain the nematic matrixes/tensor
        Q_cell = self.nematic_tensors[cell_index]
        Q_neighbor = self.nematic_tensors[neighbor_index]
        # and calculate the matriz M
        matrix_M = (
            cell.squared_diagonal * cell.anisotropy * Q_cell
            + neighbor.squared_diagonal * neighbor.anisotropy * Q_neighbor
        ) / (cell.squared_diagonal + neighbor.squared_diagonal)

        # finally we can calculate i_0 and the overlap
        # i_0 = (4*pi*l_par_k*l_perp_k*l_par_j*l_perp_j)/sqrt(beta)
        # with l_parallel = np.sqrt((cell_area*cell.aspect_ratio)/np.pi)
        # and l_perp = sqrt(cell_area/(np.pi*cell.aspect_ratio))
        i_0 = 4 * self.cell_area**2 / (np.pi * np.sqrt(beta))

        #relative_pos = np.array([relative_pos_x, relative_pos_y, 0])
        overlap = i_0 * np.exp(
            -((cell.squared_diagonal + neighbor.squared_diagonal) / beta)
            * np.matmul(
                relative_pos,
                np.matmul(np.identity(3) - matrix_M, relative_pos),
            )
        )
        # we return the overlap between the cell and its neighbor
        return overlap

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

    def calculate_max_overlap(
        self,
        cell_index: int,
        neighbor_index: int,
    ) -> float:
        """
        Calculates the maximum overlap between two cells using overlap calculated 
        in the TF. (with the orientation of each cell)

        Parameters
        ----------
        cell_index : int
            The index of the cell.
        neighbor_index : int
            The index of the neighbor.

        Returns
        -------
        max_overlap : float
            The maximum overlap between cells
        """
        cell = self.cells[cell_index]
        neighbor = self.cells[neighbor_index]

        # now we introduce the constant beta introduced by us in the TF
        beta = (
            (cell.squared_diagonal + neighbor.squared_diagonal) ** 2
            - (cell.squared_diagonal * cell.anisotropy - neighbor.squared_diagonal * neighbor.anisotropy) ** 2
            - 4
            * cell.squared_diagonal
            * cell.anisotropy
            * neighbor.squared_diagonal
            * neighbor.anisotropy
            * (
                np.cos(
                    self.cell_phies[cell_index]
                    - self.cell_phies[neighbor_index]
                )
            )
            ** 2
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
        # we save the old attributes
        old_position = np.array(self.cell_positions[cell_index])
        old_phi = self.cell_phies[cell_index]
        old_aspect_ratio = cell.aspect_ratio
        # and get the place of the grid that correspond to the cell
        old_index = self.grid.get_hash_key(old_position)
        # create a dict that contains the total overlap of the cell with others
        total_overlap = dict()
        for attempt in range(self.cell_max_def_attempts):
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
            # Calculate relative positions for all neighbors
            relative_positions = np.array(
                [
                    self.relative_pos(
                        self.cell_positions[cell_index],
                        self.cell_positions[neighbor_index],
                    )
                    for neighbor_index in candidate_neighbors
                ]
            )

            # initialize a number that is the sum of all the overlaps
            total_overlap_angle = 0
            # create a list with booleans that are True if cells overlap
            existing_overlap = []

            for neighbor_index, relative_pos in zip(candidate_neighbors, relative_positions):
                overlap = self.calculate_overlap(
                    cell_index,
                    neighbor_index,
                    relative_pos,
                )
                # add the overlap to the total overlap of the cell
                total_overlap_angle += overlap
                # we calculate the overlap threshold
                max_overlap = self.calculate_max_overlap(cell_index, neighbor_index)

                # if the overlap is greater than the threshold, add it to the list
                if overlap > self.overlap_threshold_ratio*max_overlap:
                    existing_overlap.append(True)
                else:
                    existing_overlap.append(False)
            # if the list is empty or the cell does not overlap
            if not existing_overlap or not any(existing_overlap):
                # add the angle and overlap in the dict
                #total_overlap[(new_phi, new_position)] = total_overlap_angle
                total_overlap[(new_phi, tuple(new_position))] = total_overlap_angle

            # turn back to the original values
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
   
        # for attempt in range(self.cell_max_def_attempts): # NO HAY ATTEMPTS AHORA
        # random phi and aspect ratio=max and generate a position with them
        #new_phi = self.rng.uniform(low=0, high=2 * np.pi)
        new_aspect_ratio = old_aspect_ratio + self.delta_aspect_ratio
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
        # Calculate relative positions for all neighbors
        relative_positions = np.array(
            [
                self.relative_pos(
                    self.cell_positions[cell_index],
                    self.cell_positions[neighbor_index],
                )
                for neighbor_index in candidate_neighbors
            ]
        )

        # calculation of overlap
        no_overlap = True
        for neighbor_index, relative_pos in zip(candidate_neighbors, relative_positions):
            overlap = self.calculate_overlap(
                cell_index,
                neighbor_index,
                relative_pos,
            )
            # we calculate the overlap threshold
            max_overlap = self.calculate_max_overlap(cell_index, neighbor_index)
            if overlap > self.overlap_threshold_ratio*max_overlap:
                # if the new cell overlaps with another, we turn back to the
                # original values
                self.cell_positions[cell_index] = old_position
                cell.set_aspect_ratio(old_aspect_ratio)
                no_overlap = False
                break

        if no_overlap:
            # if there is no overlap, the new cell remains and we finish the loop
            succesful_elongation = True
            # if we have change the index, the candidate for neighbors also change
            # Update the index of the cell if necessary
            if old_index != new_index:
                self.grid.remove_cell_from_hash_table(cell_index, old_position)
                self.grid.add_cell_to_hash_table(cell_index, new_position)
            return succesful_elongation

        succesful_elongation = False
        return succesful_elongation

    def shrink_from_elliptical(self, cell_index: int) -> bool:
        """If the cell is not round and its `shrink` attribute is True, it attempts to 
        shrink.

        Parameters
        ----------
        cell_index : int
            The index of the cell.

        Returns
        ----------
        succesful_shrinking : bool
            True if the deformation was successful, False otherwise.
        """
        cell = self.cells[cell_index]
        if cell.shrink == True:
            # turn the cell back to round
            old_aspect_ratio = cell.aspect_ratio
            cell.set_aspect_ratio(old_aspect_ratio-self.delta_aspect_ratio)
            if np.isclose(old_aspect_ratio-self.delta_aspect_ratio, 1):
                self.cell_phies[cell_index] = 0
                self.update_nematic_tensors([cell_index])
            # and the shrink turns back to False 
            cell.shrink = False
            succesful_shrinking = True
        else:
            succesful_shrinking = False
        return succesful_shrinking

    def interaction(self, cell_index: int, delta_t: float) -> Tuple[np.ndarray, float]:
        """The given cell interacts with others if they are close enough.

        It describes the interaction of the cells given a force. It changes the position
        of the cell (because of the forces exerted by all the other cells and the
        intrinsic velocity) and it's angle in the x-y plane, phi (becuase of a torque).

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
        
        candidate_neighbors = list(
            self.grid.find_neighbors(
                position=self.cell_positions[cell_index],
            )
        )
        # modifies the set in-place to remove the actual cell index
        candidate_neighbors.remove(cell_index)
        # For all the neighbor candidates, we calculate the relative position with the cell
        for neighbor_index in candidate_neighbors:
            # We make sure that it is not already calculated
            if neighbor_index not in cell.neighbors_relative_pos:
                relative_pos = self.relative_pos(
                    self.cell_positions[cell_index],
                    self.cell_positions[neighbor_index]
                )
                # Add the relative position to the dictionary
                cell.neighbors_relative_pos[neighbor_index] = relative_pos
                # we update also the attribute of the neighbor corresponding to the actual cell                
                neighbor = self.cells[neighbor_index]
                neighbor.neighbors_relative_pos[cell_index] = -relative_pos

        for neighbor_index, relative_pos in cell.neighbors_relative_pos.items():
            # We make sure that it is not already calculated
            if neighbor_index not in cell.neighbors_overlap:
                overlap = self.calculate_overlap(
                    cell_index,
                    neighbor_index,
                    cell.neighbors_relative_pos[neighbor_index]
                )
                # Add the overlap to the dictionary
                cell.neighbors_overlap[neighbor_index] = overlap
                # we update also the attribute of the neighbor corresponding to the actual cell
                neighbor = self.cells[neighbor_index]
                neighbor.neighbors_overlap[cell_index] = overlap

        # The significant neighbors are those which the overlap is more than the threshold
        significant_neighbors_indexes = [
            neighbor_index
            for neighbor_index, overlap in cell.neighbors_overlap.items()
            if overlap > self.overlap_threshold_ratio*self.calculate_max_overlap(cell_index, neighbor_index)
        ]
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

        #we return the change in the position and in the phi angle of the cell
        return dif_position, dif_phi

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

    # ---------------------------------------------------------

    def simulate(self, num_times: int) -> None:
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
        if len(self.cells) == 0:
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
                for i in range(0, self.initial_number_of_cells):
                    # choose a random position and angle in the xy plane (phi)
                    Cell(
                        position=np.array(
                            [
                                self.rng.uniform(low=0, high=self.side),
                                self.rng.uniform(low=0, high=self.side),
                                0,
                            ]
                        ),
                        culture=self,
                        is_stem=self.first_cell_is_stem,
                        phi=0 if self.initial_aspect_ratio==1 else self.rng.uniform(low=0, high=2*np.pi),
                        aspect_ratio=self.initial_aspect_ratio,
                        parent_index=0,
                        shrink=False,
                        available_space=True,
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

        # we simulate for num_times time steps
        for i in range(1, num_times + 1):
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
                # Wait for the system to stabilize before deformation
                if i>self.stabilization_time and self.deformation:
                    # we get a permuted copy of the cells list
                    active_cell_indexes = self.rng.permutation(
                        self.active_cell_indexes
                    )

                    # Create a list of the successful deformations
                    deformation_success = []
                    for index in active_cell_indexes:
                        cell = self.cells[index]
                        # if the cell is round
                        if np.isclose(cell.aspect_ratio, 1):
                            # We try to elongate the cell
                            success = self.elongate_from_round(index)
                        else:
                            # We try to shrink the cell
                            success = self.shrink_from_elliptical(index)
                            # if it can´t shrink, we try to elongate it
                            if not success and cell.aspect_ratio < self.aspect_ratio_max:
                                success = self.elongate_from_elliptical(index)
                        deformation_success.append(success)

                # We initialize the change in the position and angle of all cells
                dif_positions = np.zeros((len(self.active_cell_indexes), 3))
                dif_phies = np.zeros(len(self.active_cell_indexes))
                # Calculate the interaction for every cell
                for index in self.active_cell_indexes:
                    dif_position, dif_phi = self.interaction(
                        cell_index=index, delta_t=self.delta_t,
                    )
                    # add the change in position to the matrix
                    dif_positions[index] = dif_position
                    # add the change in angle to the matrix
                    dif_phies[index] = dif_phi
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

        self.output.record_final_state(
            tic=num_times,
            cells=self.cells,
            cell_positions=self.cell_positions,
            active_cell_indexes=self.active_cell_indexes,
        )
