"""Module that contains the classes that handle simulation output."""

import logging
import os
import sqlite3
from abc import ABC, abstractmethod
from typing import List

import pandas as pd
import numpy as np


class TumorsphereOutput(ABC):
    """
    Abstract base class for defining the output interface for the simulation.

    This class provides the methods that need to be implemented by concrete
    output classes in order to record and store the simulation data.
    """

    @abstractmethod
    def begin_culture(
        self,
        prob_stem,
        prob_diff,
        rng_seed,
        simulation_start,
        adjacency_threshold,
        swap_probability,
    ):
        """
        Record the beginning of a simulation.

        This method is called just once at the beginning of the simulation to
        record general culture parameters.
        """
        pass

    @abstractmethod
    def record_stemness(self, cell_index, tic, stemness):
        """
        Record a change in the stemness of a cell.

        This method is calld right after a cell has changed its stemness
        """
        pass

    @abstractmethod
    def record_deactivation(self, cell_index, tic):
        """
        Record the deactivation of a cell.

        This method is called when a cell is deactivated, right after setting
        its available_space attribute to False, and removing it from the list
        of active cells.
        """
        pass

    @abstractmethod
    def record_culture_state(
        self,
        tic,
        cells,
        cell_positions,
        cell_phies,
        active_cell_indexes,
        side,
        cell_area,
    ):
        """
        Record the state of the culture at a given time step.

        This method is called after creating the first cell, with tic = 0, and
        then after each time step, to record the state of the culture at that
        time step.
        """
        pass

    @abstractmethod
    def record_cell(
        self, index, parent, pos_x, pos_y, pos_z, creation_time, is_stem
    ):
        """
        Record the creation of a new cell.

        This method is called when a new cell is created, at the end of the
        cell's __init__ method.
        """
        pass

    @abstractmethod
    def record_final_state(
        self, tic, cells, cell_positions, active_cell_indexes
    ):
        """
        Record the final state of the culture.

        This method is called at the end of the simulation, after the last time
        step, to record the final state of the culture.
        """
        pass

    def should_record_clusters(
        self,
        tic: int,
        final_tic: int,
    ) -> bool:
        """
        Return whether cluster information is required at this timestep.
        """
        return False

    def record_cluster_state(
        self,
        tic,
        final_tic,
        cells,
        cell_positions,
        cell_phies,
        cell_instantaneous_velocities,
        clusters,
        side,
    ):
        """
        Record cluster information for the current culture state.
        """
        pass

    def should_record_local_order(
        self,
        tic: int,
        final_tic: int,
    ) -> bool:
        """
        Return whether local order parameters are required
        at the current timestep.
        """
        return False


    def record_local_order_state(
        self,
        tic,
        final_tic,
        cells,
        cell_positions,
        cell_phies,
        side,
    ):
        """
        Record local order parameters for the current culture state.
        """
        pass

    def should_record_deformation_events(
        self,
        tic: int,
        final_tic: int,
    ) -> bool:
        """
        Return whether deformation events are required at this timestep.
        """
        return False

    def record_deformation_events(
        self,
        tic_start: int,
        tic_end: int,
        final_tic: int,
        event_counts: dict,
    ) -> None:
        """
        Record deformation events accumulated during one time interval.
        """
        pass

class OutputDemux(TumorsphereOutput):
    """Class managing multiple output objects and delegating method calls."""

    def __init__(
        self,
        culture_name: str,
        result_list: List[TumorsphereOutput],
    ):
        self.culture_name = culture_name
        self.result_list = result_list
        # result_list's elements are other TumorsphereOutput objects

    def begin_culture(
        self,
        prob_stem,
        prob_diff,
        rng_seed,
        simulation_start,
        adjacency_threshold,
        swap_probability,
    ):
        """Delegate the call to all output objects in result_list."""
        for result in self.result_list:
            result.begin_culture(
                prob_stem,
                prob_diff,
                rng_seed,
                simulation_start,
                adjacency_threshold,
                swap_probability,
            )

    def record_stemness(self, cell_index, tic, stemness):
        """Delegate the call to all output objects in result_list."""
        for result in self.result_list:
            result.record_stemness(cell_index, tic, stemness)

    def record_deactivation(self, cell_index, tic):
        """Delegate the call to all output objects in result_list."""
        for result in self.result_list:
            result.record_deactivation(cell_index, tic)

    def record_culture_state(
        self,
        tic,
        cells,
        cell_positions,
        cell_phies,
        active_cell_indexes,
        side,
        cell_area,
    ):
        """Delegate the call to all output objects in result_list."""
        for result in self.result_list:
            result.record_culture_state(
                tic,
                cells,
                cell_positions,
                cell_phies,
                active_cell_indexes,
                side,
                cell_area,
            )

    def record_cell(
        self, index, parent, pos_x, pos_y, pos_z, creation_time, is_stem
    ):
        """Delegate the call to all output objects in result_list."""
        for result in self.result_list:
            result.record_cell(
                index, parent, pos_x, pos_y, pos_z, creation_time, is_stem
            )

    def record_final_state(
        self, tic, cells, cell_positions, active_cell_indexes
    ):
        """Delegate the call to all output objects in result_list."""
        for result in self.result_list:
            result.record_final_state(
                tic, cells, cell_positions, active_cell_indexes
            )

    def should_record_clusters(
        self,
        tic: int,
        final_tic: int,
    ) -> bool:
        """
        Return True if at least one output requires cluster data
        at the current timestep.
        """
        return any(
            result.should_record_clusters(
                tic=tic,
                final_tic=final_tic,
            )
            for result in self.result_list
        )

    def record_cluster_state(
        self,
        tic,
        final_tic,
        cells,
        cell_positions,
        cell_phies,
        cell_instantaneous_velocities,
        clusters,
        side,
    ):
        """
        Delegate cluster recording only to outputs that require
        cluster data at the current timestep.
        """
        for result in self.result_list:
            if result.should_record_clusters(tic, final_tic):
                result.record_cluster_state(
                    tic=tic,
                    final_tic=final_tic,
                    cells=cells,
                    cell_positions=cell_positions,
                    cell_phies=cell_phies,
                    cell_instantaneous_velocities=cell_instantaneous_velocities,
                    clusters=clusters,
                    side=side,
                )

    def should_record_local_order(
        self,
        tic: int,
        final_tic: int,
    ) -> bool:
        """
        Return True if at least one output requires local order
        parameters at the current timestep.
        """
        return any(
            result.should_record_local_order(
                tic=tic,
                final_tic=final_tic,
            )
            for result in self.result_list
        )


    def record_local_order_state(
        self,
        tic,
        final_tic,
        cells,
        cell_positions,
        cell_phies,
        side,
    ):
        """
        Delegate local-order recording only to outputs that require
        it at the current timestep.
        """
        for result in self.result_list:
            if result.should_record_local_order(
                tic=tic,
                final_tic=final_tic,
            ):
                result.record_local_order_state(
                    tic=tic,
                    final_tic=final_tic,
                    cells=cells,
                    cell_positions=cell_positions,
                    cell_phies=cell_phies,
                    side=side,
                )

    def should_record_deformation_events(
        self,
        tic: int,
        final_tic: int,
    ) -> bool:
        """
        Return True if at least one output requires deformation-event
        data at the current timestep.
        """
        return any(
            result.should_record_deformation_events(
                tic=tic,
                final_tic=final_tic,
            )
            for result in self.result_list
        )

    def record_deformation_events(
        self,
        tic_start: int,
        tic_end: int,
        final_tic: int,
        event_counts: dict,
    ) -> None:
        """
        Delegate deformation-event recording only to outputs that
        require it at the end of the current interval.
        """
        for result in self.result_list:
            if result.should_record_deformation_events(
                tic=tic_end,
                final_tic=final_tic,
            ):
                result.record_deformation_events(
                    tic_start=tic_start,
                    tic_end=tic_end,
                    final_tic=final_tic,
                    event_counts=event_counts,
                )

class SQLOutput(TumorsphereOutput):
    """Class for handling output to a SQLite database."""

    def __init__(
        self, culture_name, output_dir="."
    ):  # Add output_dir parameter
        self.conn = None
        db_path = (
            f"{output_dir}/{culture_name}.db"  # Use output_dir for db path
        )
        try:
            self.conn = sqlite3.connect(db_path)
        except sqlite3.OperationalError as e:
            logging.error(f"Failed to connect to database at {db_path}: {e}")
            raise

        cursor = self.conn.cursor()

        # Enable foreign key constraints for this connection
        cursor.execute("PRAGMA foreign_keys = ON;")

        # Creating the Culture table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS Cultures (
                culture_id INTEGER PRIMARY KEY AUTOINCREMENT,
                prob_stem REAL NOT NULL,
                prob_diff REAL NOT NULL,
                culture_seed INTEGER NOT NULL,
                simulation_start TIMESTAMP NOT NULL,
                adjacency_threshold REAL NOT NULL,
                swap_probability REAL NOT NULL
            );
            """
        )
        # Creating the Cells table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS Cells (
            _index INTEGER PRIMARY KEY,
            parent_index INTEGER,
            position_x REAL NOT NULL,
            position_y REAL NOT NULL,
            position_z REAL NOT NULL,
            t_creation INTEGER NOT NULL,
            t_deactivation INTEGER,
            culture_id INTEGER,
            FOREIGN KEY(culture_id) REFERENCES Cultures(culture_id)
            );
            """
        )
        # Creating the StemChange table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS StemChanges (
            change_id INTEGER PRIMARY KEY AUTOINCREMENT,
            cell_id INTEGER NOT NULL,
            t_change INTEGER NOT NULL,
            is_stem BOOLEAN NOT NULL,
            FOREIGN KEY(cell_id) REFERENCES Cells(_index)
            );
            """
        )

    def begin_culture(
        self,
        prob_stem,
        prob_diff,
        rng_seed,
        simulation_start,
        adjacency_threshold,
        swap_probability,
    ) -> int:  # Wired annotation, the method returns None
        """Record the beginning of a simulation.

        Insert a new row in the Cultures table with the specified parameters.
        """
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO Cultures (
                    prob_stem, prob_diff, culture_seed, simulation_start,
                    adjacency_threshold, swap_probability
                )
                VALUES (?, ?, ?, ?, ?, ?);
            """,
                (
                    prob_stem,
                    prob_diff,
                    int(rng_seed),
                    simulation_start,
                    adjacency_threshold,
                    swap_probability,
                ),
            )
            self.culture_id = cursor.lastrowid  # Perhaps it'd be better to
            # initialize self.culture_id in the __init__ method

    def record_stemness(self, cell_index, tic, stemness):
        """Record a change in the stemness of a cell.

        Insert a new row in the StemChanges table with the cell_id, the time
        of the change, and the new stemness value from that time on.
        """
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO StemChanges (cell_id, t_change, is_stem)
                VALUES (?, ?, ?);
            """,
                (
                    int(cell_index),
                    tic,
                    stemness,
                ),
            )

    def record_deactivation(self, cell_index, tic):
        """Record the deactivation of a cell.

        Update the t_deactivation value for the specified cell in the Cells
        table.
        """
        with self.conn:
            cursor = self.conn.cursor()

            # Recording (updating) the t_deactivation value for the specified
            # cell
            cursor.execute(
                """
                UPDATE Cells
                SET t_deactivation = ?
                WHERE _index = ?;
                """,
                (tic, int(cell_index)),
            )

    def record_culture_state(
        self,
        tic,
        cells,
        cell_positions,
        cell_phies,
        active_cell_indexes,
        side,
        cell_area,
    ):
        """We do not record the state of the culture, it'd be redundant."""
        pass

    def record_cell(
        self, index, parent, pos_x, pos_y, pos_z, creation_time, is_stem
    ):
        """Record the creation of a new cell.

        Insert a new row in the Cells table with the specified parameters.
        """
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO Cells (
                    _index, parent_index, position_x, position_y, position_z,
                    t_creation, culture_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
                (
                    index,
                    parent,
                    pos_x,
                    pos_y,
                    pos_z,
                    creation_time,
                    self.culture_id,
                ),
            )
            cursor.execute(
                """
                INSERT INTO StemChanges (cell_id, t_change, is_stem)
                VALUES (?, ?, ?);
            """,
                (
                    int(index),
                    creation_time,
                    is_stem,
                ),
            )

    def record_final_state(
        self, tic, cells, cell_positions, active_cell_indexes
    ):
        """Record the final state of the culture.

        We do not record the final state of the culture, it'd be redundant.
        """
        pass


class DatOutput(TumorsphereOutput):
    """Class for handling output to a .dat file."""

    def __init__(self, culture_name, output_dir="."):
        self.filename = f"{output_dir}/{culture_name}.dat"
        with open(self.filename, "w") as datfile:
            datfile.write(
                "total_cells, active_cells, stem_cells, active_stem_cells\n"
            )

    def begin_culture(
        self,
        prob_stem,
        prob_diff,
        rng_seed,
        simulation_start,
        adjacency_threshold,
        swap_probability,
    ):
        """We do not record the beginning of the simulation."""
        pass

    def record_stemness(self, cell_index, tic, stemness):
        """We do not record the individual stemness changes."""
        pass

    def record_deactivation(self, cell_index, tic):
        """We do not record the individual deactivations."""
        pass

    def record_culture_state(
        self,
        tic,
        cells,
        cell_positions,
        cell_phies,
        active_cell_indexes,
        side,
        cell_area,
    ):
        """Record the state of the culture at a given time step.

        We write the total number of cells, the number of active cells, the
        number of stem cells, and the number of active stem cells to the file.
        """
        with open(self.filename, "a") as datfile:
            # we count the total number of cells and active cells
            num_cells = len(cells)
            num_active = len(active_cell_indexes)

            # we count the number of CSCs in this time step
            total_stem_counter = 0
            for cell in cells:
                if cell.is_stem:
                    total_stem_counter = total_stem_counter + 1

            # we count the number of active CSCs in this time step
            active_stem_counter = 0
            for index in active_cell_indexes:
                if cells[index].is_stem:
                    active_stem_counter = active_stem_counter + 1

            # we save the data to the file
            datfile.write(
                (
                    f"{num_cells}, {num_active}, {total_stem_counter},"
                    f" {active_stem_counter} \n"
                )
            )

    def record_cell(
        self, index, parent, pos_x, pos_y, pos_z, creation_time, is_stem
    ):
        """We do not record the individual cell creations."""
        pass

    def record_final_state(
        self, tic, cells, cell_positions, active_cell_indexes
    ):
        """The final state of the culture is already recorded for the type of
        data we are saving.
        """
        pass

class DatOutput_position_aspectratio(TumorsphereOutput):
    def __init__(self, culture_name, output_dir=".", save_step=1):
        self.output_dir = output_dir
        self.save_step = save_step
        self.culture_name = culture_name
     
    def begin_culture(
        self,
        prob_stem,
        prob_diff,
        rng_seed,
        simulation_start,
        adjacency_threshold,
        swap_probability,
    ):
        """We do not record the beginning of the simulation."""
        pass

    def record_stemness(self, cell_index, tic, stemness):
        """We do not record the individual stemness changes."""
        pass

    def record_deactivation(self, cell_index, tic):
        """We do not record the individual deactivations."""
        pass
 
    def calculate_order_parameters(self, cells, cell_phies):
        """
        Calculate the order parameters for all cells in the current step
        """

        # List of elongated cells
        elongated_cells = [cell._index for cell in cells if not np.isclose(cell.aspect_ratio, 1)]
        # Calculation of the number of elongated cells and cells
        num_elongated = len(elongated_cells)
        num_cells = len(cells)

        # Calculate sin(phi), cos(phi), sin(2phi), cos(2phi) for every elongated cell
        sin = np.sin(cell_phies[elongated_cells])
        cos = np.cos(cell_phies[elongated_cells])
        sin_2 = np.sin(2*cell_phies[elongated_cells])
        cos_2 = np.cos(2*cell_phies[elongated_cells])
        # Add them
        sum_sin = sin.sum()
        sum_cos = cos.sum()
        sum_sin_2 = sin_2.sum()
        sum_cos_2 = cos_2.sum()
        # Calculate the parameters     
        if num_elongated != 0:
            nematic = np.sqrt(sum_sin_2**2 + sum_cos_2**2) / num_elongated
            polar = np.sqrt(sum_sin**2 + sum_cos**2) / num_elongated
            nematic_2 = np.sqrt(sum_sin_2**2 + sum_cos_2**2) / num_cells
            polar_2 = np.sqrt(sum_sin**2 + sum_cos**2) / num_cells
        else:
            nematic = 0
            polar = 0
            nematic_2 = 0
            polar_2 = 0
        fraction_elongated = num_cells/num_elongated

        return nematic, polar, nematic_2, polar_2, fraction_elongated

    def record_culture_state(
        self,
        tic,
        cells,
        cell_positions,
        cell_phies,
        active_cell_indexes,
        side,
        cell_area,
    ):
        if np.mod(tic, self.save_step) == 0:
            os.makedirs(f"{self.output_dir}/dat", exist_ok=True)
            filename = (
                f"{self.output_dir}/dat/{self.culture_name}_step={tic:05}.dat"
            )
            with open(filename, "w") as datfile:
                datfile.write(
                    "position_x,position_y,position_z,orientation,aspect_ratio\n"
                )
            for cell in cells:
                with open(filename, "a") as datfile:
                    # we save the positions and the aspect ratio to the file
                    datfile.write(
                        f"{cell_positions[cell._index][0]}, {cell_positions[cell._index][1]}, {cell_positions[cell._index][2]}, {cell_phies[cell._index]}, {cell.aspect_ratio} \n"
                    )

    def record_cell(
        self, index, parent, pos_x, pos_y, pos_z, creation_time, is_stem
    ):
        """We do not record the individual cell creations."""
        pass

    def record_final_state(
        self, tic, cells, cell_positions, active_cell_indexes
    ):
        """The final state of the culture is already recorded for the type of
        data we are saving.
        """
        pass

class DatOutput_order_parameters(TumorsphereOutput):
    def __init__(self, culture_name, output_dir=".", save_step=1):
        self.output_dir = output_dir
        self.save_step = save_step
        self.culture_name = culture_name
     
    def begin_culture(
        self,
        prob_stem,
        prob_diff,
        rng_seed,
        simulation_start,
        adjacency_threshold,
        swap_probability,
    ):
        """We do not record the beginning of the simulation."""
        pass

    def record_stemness(self, cell_index, tic, stemness):
        """We do not record the individual stemness changes."""
        pass

    def record_deactivation(self, cell_index, tic):
        """We do not record the individual deactivations."""
        pass
 
    def calculate_order_parameters(self, cells, cell_phies):
        """
        Calculate the order parameters for all cells in the current step
        """

        # List of elongated cells
        elongated_cells = [cell._index for cell in cells if not np.isclose(cell.aspect_ratio, 1)]
        # Calculation of the number of elongated cells and cells
        num_elongated = len(elongated_cells)
        num_cells = len(cells)

        # Calculate sin(phi), cos(phi), sin(2phi), cos(2phi) for every elongated cell
        sin_phi = np.sin(cell_phies[elongated_cells])
        cos_phi = np.cos(cell_phies[elongated_cells])
        sin_2_phi = np.sin(2*cell_phies[elongated_cells])
        cos_2_phi = np.cos(2*cell_phies[elongated_cells])
        # Add them
        sum_sin = sin_phi.sum()
        sum_cos = cos_phi.sum()
        sum_sin_2 = sin_2_phi.sum()
        sum_cos_2 = cos_2_phi.sum()
        # Calculate the parameters     
        if num_elongated != 0:
            nematic = np.sqrt(sum_sin_2**2 + sum_cos_2**2) / num_elongated
            polar = np.sqrt(sum_sin**2 + sum_cos**2) / num_elongated
            nematic_2 = np.sqrt(sum_sin_2**2 + sum_cos_2**2) / num_cells
            polar_2 = np.sqrt(sum_sin**2 + sum_cos**2) / num_cells
        else:
            nematic = 0
            polar = 0
            nematic_2 = 0
            polar_2 = 0
        fraction_elongated = num_elongated/num_cells

        return nematic, polar, nematic_2, polar_2, fraction_elongated

    def record_culture_state(
        self,
        tic,
        cells,
        cell_positions,
        cell_phies,
        active_cell_indexes,
        side,
        cell_area,
    ):
        if np.mod(tic, self.save_step) == 0:
            os.makedirs(f"{self.output_dir}/dat_order_parameters", exist_ok=True)
            filename = (
                f"{self.output_dir}/dat_order_parameters/op_{self.culture_name}_step={tic:05}.dat"
            )
            nematic, polar, nematic_2, polar_2, fraction_elongated = self.calculate_order_parameters(cells, cell_phies)
            with open(filename, "w") as datfile:
                datfile.write(
                    "nematic,polar,nematic_2,polar_2,fraction_elongated\n"
                )
                # we save the order parameters to the file
                datfile.write(
                    f"{nematic}, {polar}, {nematic_2}, {polar_2}, {fraction_elongated} \n"
                )

    def record_cell(
        self, index, parent, pos_x, pos_y, pos_z, creation_time, is_stem
    ):
        """We do not record the individual cell creations."""
        pass

    def record_final_state(
        self, tic, cells, cell_positions, active_cell_indexes
    ):
        """The final state of the culture is already recorded for the type of
        data we are saving.
        """
        pass

class DatOutput_motion_parameters(TumorsphereOutput):
    def __init__(self, culture_name, output_dir=".", save_step=1):
        self.output_dir = output_dir
        self.save_step = save_step
        self.culture_name = culture_name

        # Store previous wrapped positions and unwrapped trajectories
        self.previous_wrapped_positions = None
        self.unwrapped_positions = None
        self.initial_unwrapped_positions = None

    def begin_culture(
        self,
        prob_stem,
        prob_diff,
        rng_seed,
        simulation_start,
        adjacency_threshold,
        swap_probability,
    ):
        """We do not record the beginning of the simulation."""
        pass

    def record_stemness(self, cell_index, tic, stemness):
        """We do not record the individual stemness changes."""
        pass

    def record_deactivation(self, cell_index, tic):
        """We do not record the individual deactivations."""
        pass

    def calculate_motion_parameters(self, cell_positions, side):
        """
        Calculate motion observables using unwrapped trajectories.

        """
        # convert the wrapped position in an array
        current_wrapped_positions = np.asarray(
            cell_positions,
            dtype=float,
        ).copy()
        
        # First recorded state
        if self.previous_wrapped_positions is None:
            self.previous_wrapped_positions = (
                current_wrapped_positions.copy()
            )
            self.unwrapped_positions = (
                current_wrapped_positions.copy()
            )
            self.initial_unwrapped_positions = (
                current_wrapped_positions.copy()
            )

            return 0.0, 0.0, 0.0, 0.0

        # Wrapped displacement between consecutive simulation steps
        delta_positions = (
            current_wrapped_positions
            - self.previous_wrapped_positions
        )

        # Minimum-image correction (Boundary conditions)
        delta_positions -= (
            side * np.round(delta_positions / side)
        )

        # Update unwrapped positions
        self.unwrapped_positions += delta_positions

        # Magnitude of the displacement of each cell
        step_displacements = np.linalg.norm(
            delta_positions,
            axis=1,
        )

        mean_step_displacement = np.mean(
            step_displacements
        )

        mean_squared_step_displacement = np.mean(
            step_displacements**2
        )

        p95_step_displacement = np.percentile(
            step_displacements,
            95,
        )

        # MSD with respect to the initial state
        displacement_from_initial = (
            self.unwrapped_positions
            - self.initial_unwrapped_positions
        )

        msd_t0 = np.mean(
            np.sum(
                displacement_from_initial**2,
                axis=1,
            )
        )

        # Update previous wrapped positions for next step
        self.previous_wrapped_positions = (
            current_wrapped_positions.copy()
        )

        return (
            mean_step_displacement,
            mean_squared_step_displacement,
            p95_step_displacement,
            msd_t0,
        )


    def record_culture_state(
        self,
        tic,
        cells,
        cell_positions,
        cell_phies,
        active_cell_indexes,
        side,
        cell_area,
    ):
        (
            mean_step_displacement,
            mean_squared_step_displacement,
            p95_step_displacement,
            msd_t0,
        ) = self.calculate_motion_parameters(
            cell_positions,
            side,
        )

        if np.mod(tic, self.save_step) != 0:
            return

        os.makedirs(
            f"{self.output_dir}/dat_motion_parameters",
            exist_ok=True,
        )

        filename = (
            f"{self.output_dir}/dat_motion_parameters/"
            f"motion_{self.culture_name}_step={tic:05}.dat"
        )

        with open(filename, "w") as datfile:
            datfile.write(
                "mean_step_displacement,"
                "mean_squared_step_displacement,"
                "p95_step_displacement,"
                "msd_t0\n"
            )

            datfile.write(
                f"{mean_step_displacement},"
                f"{mean_squared_step_displacement},"
                f"{p95_step_displacement},"
                f"{msd_t0}\n"
            )


    def record_cell(
        self,
        index,
        parent,
        pos_x,
        pos_y,
        pos_z,
        creation_time,
        is_stem,
    ):
        """We do not record the individual cell creations."""
        pass

    def record_final_state(
        self,
        tic,
        cells,
        cell_positions,
        active_cell_indexes,
    ):
        """The final state of the culture is already recorded for the type of
        data we are saving.
        """
        pass

class DatOutput_local_order_parameters(
    TumorsphereOutput
):
    def __init__(
        self,
        culture_name,
        output_dir=".",
        save_step=100,
        raw_save_step=1000,
        local_box_size=10.0,
    ):
        self.culture_name = culture_name
        self.output_dir = output_dir

        # Frequency of the compact local-order summary
        self.summary_save_step = save_step

        # Frequency of the file containing one row per occupied box
        self.raw_save_step = raw_save_step

        # Target side length of the boxes used to calculate
        # the local observables
        self.local_box_size = local_box_size

    def begin_culture(
        self,
        prob_stem,
        prob_diff,
        rng_seed,
        simulation_start,
        adjacency_threshold,
        swap_probability,
    ):
        """We do not record the beginning of the simulation."""
        pass

    def record_stemness(
        self,
        cell_index,
        tic,
        stemness,
    ):
        """We do not record individual stemness changes."""
        pass

    def record_deactivation(
        self,
        cell_index,
        tic,
    ):
        """We do not record individual cell deactivations."""
        pass

    def record_culture_state(
        self,
        tic,
        cells,
        cell_positions,
        cell_phies,
        active_cell_indexes,
        side,
        cell_area,
    ):
        """
        Local order parameters will be recorded through this method
        after connecting this output to the simulation.
        """
        pass

    def record_cell(
        self,
        index,
        parent,
        pos_x,
        pos_y,
        pos_z,
        creation_time,
        is_stem,
    ):
        """We do not record individual cell creations."""
        pass

    def record_final_state(
        self,
        tic,
        cells,
        cell_positions,
        active_cell_indexes,
    ):
        """
        The final local state will be handled when this output is
        connected to the simulation.
        """
        pass

    def should_record_local_order_summary(
        self,
        tic: int,
        final_tic: int,
    ) -> bool:
        """
        Return whether the compact local-order summary must be saved.
        """
        return (
            tic == final_tic
            or np.mod(
                tic,
                self.summary_save_step,
            ) == 0
        )


    def should_record_local_order_raw(
        self,
        tic: int,
        final_tic: int,
    ) -> bool:
        """
        Return whether the complete local grid must be saved.
        """
        return (
            tic == final_tic
            or np.mod(
                tic,
                self.raw_save_step,
            ) == 0
        )


    def should_record_local_order(
        self,
        tic: int,
        final_tic: int,
    ) -> bool:
        """
        Calculate the local grid whenever either output requires it.
        """
        return (
            self.should_record_local_order_summary(
                tic=tic,
                final_tic=final_tic,
            )
            or self.should_record_local_order_raw(
                tic=tic,
                final_tic=final_tic,
            )
        )

    def calculate_local_order_grid(
        self,
        cells,
        cell_positions,
        cell_phies,
        side,
    ):
        """
        Calculate composition and orientational order parameters
        inside square boxes of approximately local_box_size.

        Only occupied boxes are returned.
        """
        positions = np.asarray(
            cell_positions,
            dtype=float,
        )

        phies = np.asarray(
            cell_phies,
            dtype=float,
        )

        # Choose an integer number of boxes along each direction
        # The actual box size is adjusted slightly so that the complete
        # periodic system is covered without leaving a remainder
        number_of_bins = max(
            1,
            int(
                np.round(
                    side / self.local_box_size
                )
            ),
        )

        actual_box_size = (
            side / number_of_bins
        )

        # Apply periodic boundary conditions before assigning cells
        # to the local boxes
        positions_xy = np.mod(
            positions[:, :2],
            side,
        )

        # Two-dimensional box coordinates of every cell
        grid_indices = np.floor(
            positions_xy / actual_box_size
        ).astype(int)

        grid_indices = np.mod(
            grid_indices,
            number_of_bins,
        )

        # Convert the pair (grid_x, grid_y) into one integer index
        flat_grid_indices = (
            grid_indices[:, 1] * number_of_bins
            + grid_indices[:, 0]
        )

        # Identify elongated cells
        elongated_mask = np.asarray(
            [
                not np.isclose(
                    cell.aspect_ratio,
                    1.0,
                )
                for cell in cells
            ],
            dtype=bool,
        )

        # Store the indices of the cells belonging to every occupied box
        # Empty boxes are not included
        cells_by_box = {}

        for cell_index, box_index in enumerate(
            flat_grid_indices
        ):
            cells_by_box.setdefault(
                int(box_index),
                [],
            ).append(
                cell_index
            )

        local_data = []

        # Calculate the local observables independently in each
        # occupied box
        for box_index, box_cell_indices in sorted(
            cells_by_box.items()
        ):
            box_cell_indices = np.asarray(
                box_cell_indices,
                dtype=int,
            )

            box_elongated_mask = elongated_mask[
                box_cell_indices
            ]

            elongated_cell_indices = box_cell_indices[
                box_elongated_mask
            ]

            occupancy = int(
                box_cell_indices.size
            )

            number_elongated = int(
                elongated_cell_indices.size
            )

            number_round = (
                occupancy
                - number_elongated
            )

            fraction_elongated = (
                number_elongated
                / occupancy
            )

            # Orientational order is defined only when the box
            # contains at least one elongated cell
            if number_elongated > 0:
                elongated_phies = phies[
                    elongated_cell_indices
                ]

                # Polar order
                sum_cos = np.sum(
                    np.cos(
                        elongated_phies
                    )
                )

                sum_sin = np.sum(
                    np.sin(
                        elongated_phies
                    )
                )

                # Nematic order
                sum_cos_2 = np.sum(
                    np.cos(
                        2.0 * elongated_phies
                    )
                )

                sum_sin_2 = np.sum(
                    np.sin(
                        2.0 * elongated_phies
                    )
                )

                polar_magnitude = np.sqrt(
                    sum_cos**2
                    + sum_sin**2
                )

                nematic_magnitude = np.sqrt(
                    sum_cos_2**2
                    + sum_sin_2**2
                )

                # Order normalized by the number of elongated cells.
                polar_order = (
                    polar_magnitude
                    / number_elongated
                )

                nematic_order = (
                    nematic_magnitude
                    / number_elongated
                )

                # Order normalized by the total number of cells.
                polar_order_hat = (
                    polar_magnitude
                    / occupancy
                )

                nematic_order_hat = (
                    nematic_magnitude
                    / occupancy
                )

            else:
                # Polar and nematic order are not defined when there
                # are no elongated cells in the box.
                polar_order = np.nan
                nematic_order = np.nan

                # The total-normalized observables are zero because
                # the box contains only round cells.
                polar_order_hat = 0.0
                nematic_order_hat = 0.0

            # Recover the two-dimensional coordinates of the box.
            grid_x = (
                box_index
                % number_of_bins
            )

            grid_y = (
                box_index
                // number_of_bins
            )

            local_data.append(
                {
                    "grid_x": grid_x,
                    "grid_y": grid_y,
                    "center_x": (
                        grid_x + 0.5
                    ) * actual_box_size,
                    "center_y": (
                        grid_y + 0.5
                    ) * actual_box_size,
                    "occupancy": occupancy,
                    "number_round": number_round,
                    "number_elongated": number_elongated,
                    "fraction_elongated": fraction_elongated,
                    "polar_order": polar_order,
                    "nematic_order": nematic_order,
                    "polar_order_hat": polar_order_hat,
                    "nematic_order_hat": nematic_order_hat,
                }
            )

        # One row is returned for every occupied box
        data = pd.DataFrame(
            local_data
        )

        # This information is needed to reconstruct the complete grid,
        # including the boxes that were empty and therefore not stored
        metadata = {
            "side": float(side),
            "number_of_bins": number_of_bins,
            "target_box_size": self.local_box_size,
            "actual_box_size": actual_box_size,
        }

        return data, metadata

    def calculate_local_order_summary(
        self,
        data,
        metadata,
    ):
        """
        Calculate compact statistics from the local-order grid.

        The regular means are calculated over boxes containing at
        least one elongated cell. The min_2 means exclude boxes with
        a single elongated cell.
        """
        # Boxes where the orientational order parameters are defined
        boxes_with_elongated = data[
            data["number_elongated"] > 0
        ]

        # Boxes containing at least two elongated cells
        boxes_with_at_least_2 = data[
            data["number_elongated"] > 1
        ]

        total_number_of_cells = int(
            data["occupancy"].sum()
        )

        number_elongated = int(
            data["number_elongated"].sum()
        )

        fraction_elongated = (
            number_elongated
            / total_number_of_cells
        )

        number_occupied_boxes = len(
            data
        )

        number_boxes_with_elongated = len(
            boxes_with_elongated
        )

        number_boxes_with_at_least_2 = len(
            boxes_with_at_least_2
        )

        # Means over all boxes containing at least one elongated cell
        if number_boxes_with_elongated > 0:
            mean_polar_order = (
                boxes_with_elongated[
                    "polar_order"
                ].mean()
            )

            mean_nematic_order = (
                boxes_with_elongated[
                    "nematic_order"
                ].mean()
            )

            # Weight every local OP by the number of elongated cells
            # contained in its box
            weighted_mean_polar_order = np.average(
                boxes_with_elongated[
                    "polar_order"
                ],
                weights=boxes_with_elongated[
                    "number_elongated"
                ],
            )

            weighted_mean_nematic_order = np.average(
                boxes_with_elongated[
                    "nematic_order"
                ],
                weights=boxes_with_elongated[
                    "number_elongated"
                ],
            )

        else:
            mean_polar_order = np.nan
            mean_nematic_order = np.nan
            weighted_mean_polar_order = np.nan
            weighted_mean_nematic_order = np.nan

        # Collective local order after excluding boxes with only
        # one elongated cell
        if number_boxes_with_at_least_2 > 0:
            mean_polar_order_min_2 = (
                boxes_with_at_least_2[
                    "polar_order"
                ].mean()
            )

            mean_nematic_order_min_2 = (
                boxes_with_at_least_2[
                    "nematic_order"
                ].mean()
            )

        else:
            mean_polar_order_min_2 = np.nan
            mean_nematic_order_min_2 = np.nan

        # The hatted observables are defined in every occupied box
        # Boxes containing only round cells contribute zero
        mean_polar_order_hat = data[
            "polar_order_hat"
        ].mean()

        mean_nematic_order_hat = data[
            "nematic_order_hat"
        ].mean()

        return {
            "number_of_bins": metadata[
                "number_of_bins"
            ],
            "actual_box_size": metadata[
                "actual_box_size"
            ],
            "total_number_of_cells": total_number_of_cells,
            "number_elongated": number_elongated,
            "fraction_elongated": fraction_elongated,
            "number_occupied_boxes": number_occupied_boxes,
            "number_boxes_with_elongated": (
                number_boxes_with_elongated
            ),
            "number_boxes_with_at_least_2_elongated": (
                number_boxes_with_at_least_2
            ),
            "mean_polar_order": mean_polar_order,
            "mean_nematic_order": mean_nematic_order,
            "mean_polar_order_min_2": (
                mean_polar_order_min_2
            ),
            "mean_nematic_order_min_2": (
                mean_nematic_order_min_2
            ),
            "weighted_mean_polar_order": (
                weighted_mean_polar_order
            ),
            "weighted_mean_nematic_order": (
                weighted_mean_nematic_order
            ),
            "mean_polar_order_hat": mean_polar_order_hat,
            "mean_nematic_order_hat": mean_nematic_order_hat,
        }

    def record_local_order_state(
        self,
        tic,
        final_tic,
        cells,
        cell_positions,
        cell_phies,
        side,
    ):
        """
        Record the compact local-order summary and, less frequently,
        the complete grid of occupied boxes.
        """
        record_summary = (
            self.should_record_local_order_summary(
                tic=tic,
                final_tic=final_tic,
            )
        )

        record_raw = (
            self.should_record_local_order_raw(
                tic=tic,
                final_tic=final_tic,
            )
        )

        # Calculate the grid only once, even when both files must
        # be recorded at the current timestep.
        data, metadata = (
            self.calculate_local_order_grid(
                cells=cells,
                cell_positions=cell_positions,
                cell_phies=cell_phies,
                side=side,
            )
        )

        output_folder = os.path.join(
            self.output_dir,
            "dat_local_order_parameters",
        )

        os.makedirs(
            output_folder,
            exist_ok=True,
        )

        raw_filename = os.path.join(
            output_folder,
            (
                f"local_order_grid_{self.culture_name}"
                f"_step={tic:05}.dat"
            ),
        )

        summary_filename = os.path.join(
            output_folder,
            (
                f"local_order_summary_{self.culture_name}"
                f"_step={tic:05}.dat"
            ),
        )

        if record_raw:
            # Save one row for every occupied local box
            with open(raw_filename, "w") as datfile:
                datfile.write(
                    f"# side={metadata['side']},"
                    f"number_of_bins={metadata['number_of_bins']},"
                    f"target_box_size={metadata['target_box_size']},"
                    f"actual_box_size={metadata['actual_box_size']}\n"
                )

                data.to_csv(
                    datfile,
                    index=False,
                )

        if record_summary:
            summary = (
                self.calculate_local_order_summary(
                    data=data,
                    metadata=metadata,
                )
            )

            # Save one row containing the spatially averaged
            # local-order observables
            with open(summary_filename, "w") as datfile:
                datfile.write(
                    "number_of_bins,"
                    "actual_box_size,"
                    "total_number_of_cells,"
                    "number_elongated,"
                    "fraction_elongated,"
                    "number_occupied_boxes,"
                    "number_boxes_with_elongated,"
                    "number_boxes_with_at_least_2_elongated,"
                    "mean_polar_order,"
                    "mean_nematic_order,"
                    "mean_polar_order_min_2,"
                    "mean_nematic_order_min_2,"
                    "weighted_mean_polar_order,"
                    "weighted_mean_nematic_order,"
                    "mean_polar_order_hat,"
                    "mean_nematic_order_hat\n"
                )

                datfile.write(
                    f"{summary['number_of_bins']},"
                    f"{summary['actual_box_size']},"
                    f"{summary['total_number_of_cells']},"
                    f"{summary['number_elongated']},"
                    f"{summary['fraction_elongated']},"
                    f"{summary['number_occupied_boxes']},"
                    f"{summary['number_boxes_with_elongated']},"
                    f"{summary['number_boxes_with_at_least_2_elongated']},"
                    f"{summary['mean_polar_order']},"
                    f"{summary['mean_nematic_order']},"
                    f"{summary['mean_polar_order_min_2']},"
                    f"{summary['mean_nematic_order_min_2']},"
                    f"{summary['weighted_mean_polar_order']},"
                    f"{summary['weighted_mean_nematic_order']},"
                    f"{summary['mean_polar_order_hat']},"
                    f"{summary['mean_nematic_order_hat']}\n"
                )

class DatOutput_cluster_parameters(TumorsphereOutput):
    def __init__(
        self,
        culture_name,
        output_dir=".",
        save_step=100,
        raw_save_step=1000,
    ):
        self.culture_name = culture_name
        self.output_dir = output_dir

        # Frequency of the compact cluster summary
        self.summary_save_step = save_step

        # Frequency of the file containing one row per cluster
        self.raw_save_step = raw_save_step

    def begin_culture(
        self,
        prob_stem,
        prob_diff,
        rng_seed,
        simulation_start,
        adjacency_threshold,
        swap_probability,
    ):
        """We do not record the beginning of the simulation."""
        pass

    def record_stemness(self, cell_index, tic, stemness):
        """We do not record the individual stemness changes."""
        pass

    def record_deactivation(self, cell_index, tic):
        """We do not record the individual deactivations."""
        pass

    def record_culture_state(
        self,
        tic,
        cells,
        cell_positions,
        cell_phies,
        active_cell_indexes,
        side,
        cell_area,
    ):
        """We do not record the culture state."""
        pass

    def record_cell(
        self,
        index,
        parent,
        pos_x,
        pos_y,
        pos_z,
        creation_time,
        is_stem,
    ):
        """We do not record the individual cell creations."""
        pass

    def record_final_state(
        self,
        tic,
        cells,
        cell_positions,
        active_cell_indexes,
    ):
        """The final state of the culture is already recorded for the type of
        data we are saving.
        """
        pass

    def calculate_size_statistics(
        self,
        cluster_list,
    ):
        """
        Calculate size statistics for a collection of clusters.

        The largest cluster is removed only once when calculating the
        observables that exclude it.

        Parameters
        ----------
        cluster_list : list[list[int]]
            List of clusters. Each cluster contains the indices of its cells.

        Returns
        -------
        statistics : dict
            Dictionary containing the raw cluster sizes and their summary
            statistics.
        """
        cluster_sizes = np.asarray(
            [
                len(cluster)
                for cluster in cluster_list
            ],
            dtype=int,
        )

        number_of_clusters = int(
            cluster_sizes.size
        )

        total_number_of_cells = int(
            np.sum(cluster_sizes)
        )

        if number_of_clusters == 0:
            return {
                "sizes": cluster_sizes,
                "total_number_of_cells": 0,
                "number_of_clusters": 0,
                "mean_cluster_size": np.nan,
                "largest_cluster_size": np.nan,
                "number_without_largest": 0,
                "mean_without_largest": np.nan,
            }

        mean_cluster_size = float(
            np.mean(cluster_sizes)
        )

        largest_cluster_size = int(
            np.max(cluster_sizes)
        )

        # Remove exactly one largest cluster, even if several clusters
        # share the maximum size.
        largest_cluster_index = int(
            np.argmax(cluster_sizes)
        )

        cluster_sizes_without_largest = np.delete(
            cluster_sizes,
            largest_cluster_index,
        )

        number_without_largest = int(
            cluster_sizes_without_largest.size
        )

        if number_without_largest == 0:
            mean_without_largest = np.nan
        else:
            mean_without_largest = float(
                np.mean(
                    cluster_sizes_without_largest
                )
            )

        return {
            "sizes": cluster_sizes,
            "total_number_of_cells": total_number_of_cells,
            "number_of_clusters": number_of_clusters,
            "mean_cluster_size": mean_cluster_size,
            "largest_cluster_size": largest_cluster_size,
            "number_without_largest": number_without_largest,
            "mean_without_largest": mean_without_largest,
        }

    def calculate_cluster_order_parameters(
        self,
        cluster,
        cell_phies,
    ):
        """
        Calculate the polar and nematic order parameters of one cluster.

        Parameters
        ----------
        cluster : list[int]
            Indices of the cells belonging to the cluster.
        cell_phies : np.ndarray
            Orientations of all cells in the culture.

        Returns
        -------
        polar_order : float
            Polar order parameter of the cluster.
        nematic_order : float
            Nematic order parameter of the cluster.
        """
        cluster_indices = np.asarray(
            cluster,
            dtype=int,
        )

        number_of_cells = int(
            cluster_indices.size
        )

        if number_of_cells == 0:
            return np.nan, np.nan

        cluster_phies = cell_phies[
            cluster_indices
        ]

        sum_cos = np.sum(
            np.cos(cluster_phies)
        )

        sum_sin = np.sum(
            np.sin(cluster_phies)
        )

        sum_cos_2 = np.sum(
            np.cos(2.0 * cluster_phies)
        )

        sum_sin_2 = np.sum(
            np.sin(2.0 * cluster_phies)
        )

        polar_order = (
            np.sqrt(
                sum_cos**2
                + sum_sin**2
            )
            / number_of_cells
        )

        nematic_order = (
            np.sqrt(
                sum_cos_2**2
                + sum_sin_2**2
            )
            / number_of_cells
        )

        return (
            float(polar_order),
            float(nematic_order),
        )


    def calculate_cluster_order_statistics(
        self,
        cluster_list,
        cell_phies,
    ):
        """
        Calculate polar and nematic order statistics for elongated clusters.

        Singleton clusters are retained in the raw arrays but excluded from
        the mean order parameters because their polar and nematic orders are
        trivially equal to one.

        Parameters
        ----------
        cluster_list : list[list[int]]
            Elongated clusters. Each cluster contains its cell indices.
        cell_phies : np.ndarray
            Orientations of all cells in the culture.

        Returns
        -------
        statistics : dict
            Raw order parameters and summary statistics.
        """
        cluster_sizes = np.asarray(
            [
                len(cluster)
                for cluster in cluster_list
            ],
            dtype=int,
        )

        number_of_clusters = int(
            cluster_sizes.size
        )

        polar_orders = np.full(
            number_of_clusters,
            np.nan,
            dtype=float,
        )

        nematic_orders = np.full(
            number_of_clusters,
            np.nan,
            dtype=float,
        )

        # Calculate the order parameters of every elongated cluster.
        for cluster_index, cluster in enumerate(
            cluster_list
        ):
            (
                polar_orders[cluster_index],
                nematic_orders[cluster_index],
            ) = self.calculate_cluster_order_parameters(
                cluster=cluster,
                cell_phies=cell_phies,
            )

        # Clusters containing a single cell have P = S = 1
        # trivially, so they are excluded from the means.
        non_singleton_mask = (
            cluster_sizes > 1
        )

        number_of_non_singleton_clusters = int(
            np.sum(non_singleton_mask)
        )

        if number_of_non_singleton_clusters == 0:
            mean_polar_order_non_singleton = np.nan
            weighted_mean_polar_order_non_singleton = np.nan

            mean_nematic_order_non_singleton = np.nan
            weighted_mean_nematic_order_non_singleton = np.nan
        else:
            non_singleton_sizes = cluster_sizes[
                non_singleton_mask
            ]

            non_singleton_polar_orders = polar_orders[
                non_singleton_mask
            ]

            non_singleton_nematic_orders = nematic_orders[
                non_singleton_mask
            ]

            mean_polar_order_non_singleton = float(
                np.mean(
                    non_singleton_polar_orders
                )
            )

            weighted_mean_polar_order_non_singleton = float(
                np.average(
                    non_singleton_polar_orders,
                    weights=non_singleton_sizes,
                )
            )

            mean_nematic_order_non_singleton = float(
                np.mean(
                    non_singleton_nematic_orders
                )
            )

            weighted_mean_nematic_order_non_singleton = float(
                np.average(
                    non_singleton_nematic_orders,
                    weights=non_singleton_sizes,
                )
            )

        if number_of_clusters == 0:
            largest_cluster_polar_order = np.nan
            largest_cluster_nematic_order = np.nan
        else:
            # if several clusters share the largest size, take the first one.
            largest_cluster_index = int(
                np.argmax(
                    cluster_sizes
                )
            )

            largest_cluster_polar_order = float(
                polar_orders[
                    largest_cluster_index
                ]
            )

            largest_cluster_nematic_order = float(
                nematic_orders[
                    largest_cluster_index
                ]
            )

        return {
            "polar_orders": polar_orders,
            "nematic_orders": nematic_orders,
            "number_of_non_singleton_clusters": (
                number_of_non_singleton_clusters
            ),
            "mean_polar_order_non_singleton": (
                mean_polar_order_non_singleton
            ),
            "weighted_mean_polar_order_non_singleton": (
                weighted_mean_polar_order_non_singleton
            ),
            "mean_nematic_order_non_singleton": (
                mean_nematic_order_non_singleton
            ),
            "weighted_mean_nematic_order_non_singleton": (
                weighted_mean_nematic_order_non_singleton
            ),
            "largest_cluster_polar_order": (
                largest_cluster_polar_order
            ),
            "largest_cluster_nematic_order": (
                largest_cluster_nematic_order
            ),
        }

    def calculate_cluster_velocity_statistics(
        self,
        cluster_list,
        cell_instantaneous_velocities,
    ):
        """
        Calculate instantaneous velocity statistics for a collection of clusters.

        For each cluster, two different quantities are calculated:

        1. Cluster velocity:
        The vectorial mean of the velocities of its cells.

        2. Mean cell speed:
        The mean of the velocity magnitudes of its cells.

        Singleton clusters are retained in the raw arrays but excluded from
        the summary means.

        Parameters
        ----------
        cluster_list : list[list[int]]
            Clusters represented by their cell indices.
        cell_instantaneous_velocities : np.ndarray
            Instantaneous resultant velocity of every cell.

        Returns
        -------
        statistics : dict
            Raw cluster velocities and summary statistics.
        """
        cluster_sizes = np.asarray(
            [
                len(cluster)
                for cluster in cluster_list
            ],
            dtype=int,
        )

        number_of_clusters = int(
            cluster_sizes.size
        )

        cluster_velocity_x = np.full(
            number_of_clusters,
            np.nan,
            dtype=float,
        )

        cluster_velocity_y = np.full(
            number_of_clusters,
            np.nan,
            dtype=float,
        )

        cluster_speeds = np.full(
            number_of_clusters,
            np.nan,
            dtype=float,
        )

        mean_cell_speeds = np.full(
            number_of_clusters,
            np.nan,
            dtype=float,
        )

        for cluster_index, cluster in enumerate(
            cluster_list
        ):
            cluster_indices = np.asarray(
                cluster,
                dtype=int,
            )

            if cluster_indices.size == 0:
                continue

            # Take the veloicities
            cell_velocities_xy = (
                cell_instantaneous_velocities[
                    cluster_indices,
                    :2,
                ]
            )

            # Vectorial mean: translational velocity of the cluster
            cluster_velocity = np.mean(
                cell_velocities_xy,
                axis=0,
            )

            cluster_velocity_x[cluster_index] = float(
                cluster_velocity[0]
            )

            cluster_velocity_y[cluster_index] = float(
                cluster_velocity[1]
            )

            cluster_speeds[cluster_index] = float(
                np.linalg.norm(
                    cluster_velocity
                )
            )

            # Scalar mean: mean amount of cell movement inside the cluster
            cell_speeds = np.linalg.norm(
                cell_velocities_xy,
                axis=1,
            )

            mean_cell_speeds[cluster_index] = float(
                np.mean(
                    cell_speeds
                )
            )

        non_singleton_mask = (
            cluster_sizes > 1
        )

        number_of_non_singleton_clusters = int(
            np.sum(
                non_singleton_mask
            )
        )

        if number_of_non_singleton_clusters == 0:
            mean_cluster_speed_non_singleton = np.nan
            weighted_mean_cluster_speed_non_singleton = np.nan

            mean_cell_speed_non_singleton = np.nan
            weighted_mean_cell_speed_non_singleton = np.nan
        else:
            non_singleton_sizes = cluster_sizes[
                non_singleton_mask
            ]

            non_singleton_cluster_speeds = cluster_speeds[
                non_singleton_mask
            ]

            non_singleton_mean_cell_speeds = mean_cell_speeds[
                non_singleton_mask
            ]

            # Every cluster contributes with the same weight
            mean_cluster_speed_non_singleton = float(
                np.mean(
                    non_singleton_cluster_speeds
                )
            )

            # Large clusters contribute proportionally to their size
            weighted_mean_cluster_speed_non_singleton = float(
                np.average(
                    non_singleton_cluster_speeds,
                    weights=non_singleton_sizes,
                )
            )

            # Mean of the cluster-level mean cell speeds
            mean_cell_speed_non_singleton = float(
                np.mean(
                    non_singleton_mean_cell_speeds
                )
            )

            # Equivalent to averaging the cell speeds over all cells
            # belonging to non-singleton clusters
            weighted_mean_cell_speed_non_singleton = float(
                np.average(
                    non_singleton_mean_cell_speeds,
                    weights=non_singleton_sizes,
                )
            )

        if number_of_clusters == 0:
            largest_cluster_speed = np.nan
            largest_cluster_mean_cell_speed = np.nan
        else:
            # if several clusters have the maximum size, take the first
            largest_cluster_index = int(
                np.argmax(
                    cluster_sizes
                )
            )

            largest_cluster_speed = float(
                cluster_speeds[
                    largest_cluster_index
                ]
            )

            largest_cluster_mean_cell_speed = float(
                mean_cell_speeds[
                    largest_cluster_index
                ]
            )

        return {
            "cluster_velocity_x": cluster_velocity_x,
            "cluster_velocity_y": cluster_velocity_y,
            "cluster_speeds": cluster_speeds,
            "mean_cell_speeds": mean_cell_speeds,
            "number_of_non_singleton_clusters": (
                number_of_non_singleton_clusters
            ),
            "mean_cluster_speed_non_singleton": (
                mean_cluster_speed_non_singleton
            ),
            "weighted_mean_cluster_speed_non_singleton": (
                weighted_mean_cluster_speed_non_singleton
            ),
            "mean_cell_speed_non_singleton": (
                mean_cell_speed_non_singleton
            ),
            "weighted_mean_cell_speed_non_singleton": (
                weighted_mean_cell_speed_non_singleton
            ),
            "largest_cluster_speed": (
                largest_cluster_speed
            ),
            "largest_cluster_mean_cell_speed": (
                largest_cluster_mean_cell_speed
            ),
        }

    def should_record_cluster_summary(
        self,
        tic: int,
        final_tic: int,
    ) -> bool:
        """
        Return whether the compact cluster summary must be saved.
        """
        return (
            tic == final_tic
            or np.mod(
                tic,
                self.summary_save_step,
            ) == 0
        )


    def should_record_cluster_raw(
        self,
        tic: int,
        final_tic: int,
    ) -> bool:
        """
        Return whether the raw per-cluster information must be saved.
        """
        return (
            tic == final_tic
            or np.mod(
                tic,
                self.raw_save_step,
            ) == 0
        )


    def should_record_clusters(
        self,
        tic: int,
        final_tic: int,
    ) -> bool:
        """
        Calculate clusters whenever either output requires them.
        """
        return (
            self.should_record_cluster_summary(
                tic=tic,
                final_tic=final_tic,
            )
            or self.should_record_cluster_raw(
                tic=tic,
                final_tic=final_tic,
            )
        )

    def record_cluster_state(
        self,
        tic,
        final_tic,
        cells,
        cell_positions,
        cell_phies,
        cell_instantaneous_velocities,
        clusters,
        side,
    ):
        """
        Record raw cluster sizes and their summary statistics.

        Round and elongated clusters are treated independently.
        """
        record_summary = (
            self.should_record_cluster_summary(
                tic=tic,
                final_tic=final_tic,
            )
        )

        record_raw = (
            self.should_record_cluster_raw(
                tic=tic,
                final_tic=final_tic,
            )
        )

        round_statistics = self.calculate_size_statistics(
            clusters["round"],
        )

        elongated_statistics = self.calculate_size_statistics(
            clusters["elongated"],
        )

        elongated_order_statistics = (
            self.calculate_cluster_order_statistics(
                cluster_list=clusters["elongated"],
                cell_phies=cell_phies,
            )
        )

        round_velocity_statistics = (
            self.calculate_cluster_velocity_statistics(
                cluster_list=clusters["round"],
                cell_instantaneous_velocities=(
                    cell_instantaneous_velocities
                ),
            )
        )

        elongated_velocity_statistics = (
            self.calculate_cluster_velocity_statistics(
                cluster_list=clusters["elongated"],
                cell_instantaneous_velocities=(
                    cell_instantaneous_velocities
                ),
            )
        )

        output_folder = os.path.join(
            self.output_dir,
            "dat_cluster_parameters",
        )

        os.makedirs(
            output_folder,
            exist_ok=True,
        )

        raw_filename = os.path.join(
            output_folder,
            (
                f"cluster_sizes_{self.culture_name}"
                f"_step={tic:05}.dat"
            ),
        )

        summary_filename = os.path.join(
            output_folder,
            (
                f"cluster_summary_{self.culture_name}"
                f"_step={tic:05}.dat"
            ),
        )

        if record_raw:
            # Save one row for every individual cluster.
            with open(raw_filename, "w") as datfile:
                datfile.write(
                    "phenotype,"
                    "cluster_id,"
                    "size,"
                    "polar_order,"
                    "nematic_order,"
                    "cluster_velocity_x,"
                    "cluster_velocity_y,"
                    "cluster_speed,"
                    "mean_cell_speed\n"
                )

                for phenotype, statistics in (
                    ("round", round_statistics),
                    ("elongated", elongated_statistics),
                ):
                    if phenotype == "elongated":
                        polar_orders = elongated_order_statistics[
                            "polar_orders"
                        ]

                        nematic_orders = elongated_order_statistics[
                            "nematic_orders"
                        ]

                        velocity_statistics = (
                            elongated_velocity_statistics
                        )
                    else:
                        number_of_clusters = int(
                            statistics["number_of_clusters"]
                        )

                        polar_orders = np.full(
                            number_of_clusters,
                            np.nan,
                            dtype=float,
                        )

                        nematic_orders = np.full(
                            number_of_clusters,
                            np.nan,
                            dtype=float,
                        )

                        velocity_statistics = (
                            round_velocity_statistics
                        )

                    for cluster_id, (
                        cluster_size,
                        polar_order,
                        nematic_order,
                        cluster_velocity_x,
                        cluster_velocity_y,
                        cluster_speed,
                        mean_cell_speed,
                    ) in enumerate(
                        zip(
                            statistics["sizes"],
                            polar_orders,
                            nematic_orders,
                            velocity_statistics[
                                "cluster_velocity_x"
                            ],
                            velocity_statistics[
                                "cluster_velocity_y"
                            ],
                            velocity_statistics[
                                "cluster_speeds"
                            ],
                            velocity_statistics[
                                "mean_cell_speeds"
                            ],
                        )
                    ):
                        datfile.write(
                            f"{phenotype},"
                            f"{cluster_id},"
                            f"{cluster_size},"
                            f"{polar_order},"
                            f"{nematic_order},"
                            f"{cluster_velocity_x},"
                            f"{cluster_velocity_y},"
                            f"{cluster_speed},"
                            f"{mean_cell_speed}\n"
                        )

        if record_summary:
            # Save one summary row for each phenotype.
            with open(summary_filename, "w") as datfile:
                datfile.write(
                    "phenotype,"
                    "total_number_of_cells,"
                    "number_of_clusters,"
                    "mean_cluster_size,"
                    "largest_cluster_size,"
                    "number_without_largest,"
                    "mean_without_largest,"
                    "number_of_non_singleton_clusters,"
                    "mean_polar_order_non_singleton,"
                    "weighted_mean_polar_order_non_singleton,"
                    "mean_nematic_order_non_singleton,"
                    "weighted_mean_nematic_order_non_singleton,"
                    "largest_cluster_polar_order,"
                    "largest_cluster_nematic_order,"
                    "mean_cluster_speed_non_singleton,"
                    "weighted_mean_cluster_speed_non_singleton,"
                    "largest_cluster_speed,"
                    "mean_cell_speed_non_singleton,"
                    "weighted_mean_cell_speed_non_singleton,"
                    "largest_cluster_mean_cell_speed\n"
                )

                for phenotype, statistics in (
                    ("round", round_statistics),
                    ("elongated", elongated_statistics),
                ):
                    if phenotype == "elongated":
                        order_statistics = (
                            elongated_order_statistics
                        )

                        velocity_statistics = (
                            elongated_velocity_statistics
                        )
                    else:
                        # Orientational order is not defined for round cells.
                        order_statistics = {
                            "number_of_non_singleton_clusters": int(
                                np.sum(
                                    statistics["sizes"] > 1
                                )
                            ),
                            "mean_polar_order_non_singleton": np.nan,
                            "weighted_mean_polar_order_non_singleton": np.nan,
                            "mean_nematic_order_non_singleton": np.nan,
                            "weighted_mean_nematic_order_non_singleton": np.nan,
                            "largest_cluster_polar_order": np.nan,
                            "largest_cluster_nematic_order": np.nan,
                        }

                        velocity_statistics = (
                            round_velocity_statistics
                        )

                    datfile.write(
                        f"{phenotype},"
                        f"{statistics['total_number_of_cells']},"
                        f"{statistics['number_of_clusters']},"
                        f"{statistics['mean_cluster_size']},"
                        f"{statistics['largest_cluster_size']},"
                        f"{statistics['number_without_largest']},"
                        f"{statistics['mean_without_largest']},"
                        f"{order_statistics['number_of_non_singleton_clusters']},"
                        f"{order_statistics['mean_polar_order_non_singleton']},"
                        f"{order_statistics['weighted_mean_polar_order_non_singleton']},"
                        f"{order_statistics['mean_nematic_order_non_singleton']},"
                        f"{order_statistics['weighted_mean_nematic_order_non_singleton']},"
                        f"{order_statistics['largest_cluster_polar_order']},"
                        f"{order_statistics['largest_cluster_nematic_order']},"
                        f"{velocity_statistics['mean_cluster_speed_non_singleton']},"
                        f"{velocity_statistics['weighted_mean_cluster_speed_non_singleton']},"
                        f"{velocity_statistics['largest_cluster_speed']},"
                        f"{velocity_statistics['mean_cell_speed_non_singleton']},"
                        f"{velocity_statistics['weighted_mean_cell_speed_non_singleton']},"
                        f"{velocity_statistics['largest_cluster_mean_cell_speed']}\n"
                    )

class DatOutput_deformation_parameters(
    TumorsphereOutput
):
    def __init__(
        self,
        culture_name,
        output_dir=".",
        save_step=100,
    ):
        self.culture_name = culture_name
        self.output_dir = output_dir
        self.save_step = save_step

    def begin_culture(
        self,
        prob_stem,
        prob_diff,
        rng_seed,
        simulation_start,
        adjacency_threshold,
        swap_probability,
    ):
        """We do not record the beginning of the simulation."""
        pass

    def record_stemness(
        self,
        cell_index,
        tic,
        stemness,
    ):
        """We do not record individual stemness changes."""
        pass

    def record_deactivation(
        self,
        cell_index,
        tic,
    ):
        """We do not record individual deactivations."""
        pass

    def record_culture_state(
        self,
        tic,
        cells,
        cell_positions,
        cell_phies,
        active_cell_indexes,
        side,
        cell_area,
    ):
        """
        We do not record instantaneous culture states.

        Deformation events are accumulated inside Culture and are
        recorded through record_deformation_events().
        """
        pass

    def record_cell(
        self,
        index,
        parent,
        pos_x,
        pos_y,
        pos_z,
        creation_time,
        is_stem,
    ):
        """We do not record individual cell creations."""
        pass

    def record_final_state(
        self,
        tic,
        cells,
        cell_positions,
        active_cell_indexes,
    ):
        """
        The final deformation interval is recorded separately through
        record_deformation_events().
        """
        pass

    def should_record_deformation_events(
        self,
        tic: int,
        final_tic: int,
    ) -> bool:
        """
        Return True at the selected recording frequency and at the
        final timestep.
        """
        return (
            tic > 0
            and (
                np.mod(
                    tic,
                    self.save_step,
                ) == 0
                or tic == final_tic
            )
        )

    def record_deformation_events(
        self,
        tic_start: int,
        tic_end: int,
        final_tic: int,
        event_counts: dict,
    ) -> None:
        """
        Record all deformation events accumulated during one interval.
        """
        output_folder = os.path.join(
            self.output_dir,
            "dat_deformation_parameters",
        )

        os.makedirs(
            output_folder,
            exist_ok=True,
        )

        filename = os.path.join(
            output_folder,
            (
                f"deformation_events_"
                f"{self.culture_name}"
                f"_step={tic_end:05}.dat"
            ),
        )

        number_of_steps = (
            tic_end
            - tic_start
            + 1
        )

        with open(
            filename,
            "w",
        ) as datfile:
            datfile.write(
                "tic_start,"
                "tic_end,"
                "number_of_steps,"
                "round_elongation_attempts,"
                "round_elongation_successes,"
                "elliptical_elongation_attempts,"
                "elliptical_elongation_successes,"
                "contraction_events,"
                "contraction_to_round_events,"
                "contraction_overlap_rejections\n"
            )

            datfile.write(
                f"{tic_start},"
                f"{tic_end},"
                f"{number_of_steps},"
                f"{event_counts['round_elongation_attempts']},"
                f"{event_counts['round_elongation_successes']},"
                f"{event_counts['elliptical_elongation_attempts']},"
                f"{event_counts['elliptical_elongation_successes']},"
                f"{event_counts['contraction_events']},"
                f"{event_counts['contraction_to_round_events']},"
                f"{event_counts.get('contraction_overlap_rejections', 0)}\n"
            )

class OvitoOutput(TumorsphereOutput):
    """Class for handling output to a file for visualization in Ovito."""

    def __init__(self, culture_name, output_dir=".", save_step=1):
        self.output_dir = output_dir
        self.culture_name = culture_name
        self.save_step = save_step

    def begin_culture(
        self,
        prob_stem,
        prob_diff,
        rng_seed,
        simulation_start,
        adjacency_threshold,
        swap_probability,
    ):
        """We do not record the beginning of the simulation."""
        pass

    def record_stemness(self, cell_index, tic, stemness):
        """We do not record the individual stemness changes."""
        pass

    def record_deactivation(self, cell_index, tic):
        """We do not record the individual deactivations."""
        pass

    def record_culture_state(
        self,
        tic,
        cells,
        cell_positions,
        cell_phies,
        active_cell_indexes,
        side,
        cell_area,

    ):
        """Writes the data file in path for ovito, for time step t of self.

        Auxiliar function for simulate_with_ovito_data.
        """
        # we save the ovito if tic is multiple of the save_step or in some special situations
        # in order to see the deformation
        if (
            np.mod(tic, self.save_step) == 0
        ):
            path_folder = os.path.join(self.output_dir, "ovito")
            os.makedirs(path_folder, exist_ok=True)

            path_to_write = os.path.join(
                path_folder, f"ovito_data_{self.culture_name}.{tic:05}"
            )

            with open(path_to_write, "w") as file_to_write:
                file_to_write.write(str(len(cells)) + "\n")
                file_to_write.write(
                    ' Lattice="'
                    + str(side)
                    + " 0.0 0.0 0.0 "
                    + str(side)
                    + ' 0.0 0.0 0.0 1.0"Properties=species:S:1:pos:R:3:aspherical_shape:R:3:orientation:R:4:Color:R:1'
                    + "\n"
                )
                for cell in cells: # csc activas
                    
                    if cell.is_stem and cell.available_space:
                        phi = cell_phies[cell._index]
                        # Color condition in aspect ratio
                        aspect_ratio_condition = (cell.aspect_ratio - 1) / (cell.culture.aspect_ratio_max - 1)

                        if phi is None:
                            color_value = 1
                        elif np.isclose(aspect_ratio_condition, 0):
                            color_value = 0
                        else:
                            color_value = phi % (2 * np.pi)
                        line = (
                            (
                                "active_stem "
                                if cell_phies[cell._index] is None
                                else "cell "
                            )
                            + str(cell_positions[cell._index][0])
                            + " "
                            + str(cell_positions[cell._index][1])
                            + " "
                            + str(cell_positions[cell._index][2])
                            + " "
                            + f"{1 if phi is None else np.sqrt((cell_area*cell.aspect_ratio)/np.pi)}"  # aspherical shape x
                            + " "
                            + f"{1 if phi is None else np.sqrt(cell_area/(np.pi*cell.aspect_ratio))}"  # aspherical shape y
                            + " "
                            + "1"  # aspherical shape z
                            + " "
                            + "0"  # X orientation, str(0*np.sin((phi)/2))
                            + " "
                            + "0"  # Y orientation, str(0*np.sin((phi)/2))
                            + " "
                            + f"{0 if phi is None else np.sin(phi / 2)}"  # Z orientation
                            + " "
                            + f"{0 if phi is None else np.cos(phi / 2)}"  # W orientation
                            + " "
                            + f"{color_value}"  # color
                            + "\n"
                        )
                        file_to_write.write(line)

                for cell in cells:  # csc quiesc
                    if cell.is_stem and (not cell.available_space):
                        line = (
                            "quiesc_stem "
                            + str(cell_positions[cell._index][0])
                            + " "
                            + str(cell_positions[cell._index][1])
                            + " "
                            + str(cell_positions[cell._index][2])
                            + " "
                            + "1"  # aspherical shape x
                            + " "
                            + "1"  # aspherical shape y
                            + " "
                            + "1"  # aspherical shape z
                            + " "
                            + "0"  # X orientation, str(0*np.sin((phi)/2))
                            + " "
                            + "0"  # Y orientation, str(0*np.sin((phi)/2))
                            + " "
                            + "0"  # Z orientation
                            + " "
                            + "0"  # W orientation
                            + " "
                            + "2"
                            + "\n"
                        )
                        file_to_write.write(line)

                for cell in cells:  # dcc activas
                    if (not cell.is_stem) and cell.available_space:
                        line = (
                            "active_diff "
                            + str(cell_positions[cell._index][0])
                            + " "
                            + str(cell_positions[cell._index][1])
                            + " "
                            + str(cell_positions[cell._index][2])
                            + " "
                            + "1"  # aspherical shape x
                            + " "
                            + "1"  # aspherical shape y
                            + " "
                            + "1"  # aspherical shape z
                            + " "
                            + "0"  # X orientation, str(0*np.sin((phi)/2))
                            + " "
                            + "0"  # Y orientation, str(0*np.sin((phi)/2))
                            + " "
                            + "0"  # Z orientation
                            + " "
                            + "0"  # W orientation
                            + " "
                            + "3"
                            + "\n"
                        )
                        file_to_write.write(line)

                for cell in cells:  # dcc quiesc
                    if not (cell.is_stem or cell.available_space):
                        line = (
                            "quiesc_diff "
                            + str(cell_positions[cell._index][0])
                            + " "
                            + str(cell_positions[cell._index][1])
                            + " "
                            + str(cell_positions[cell._index][2])
                            + " "
                            + "1"  # aspherical shape x
                            + " "
                            + "1"  # aspherical shape y
                            + " "
                            + "1"  # aspherical shape z
                            + " "
                            + "0"  # X orientation, str(0*np.sin((phi)/2))
                            + " "
                            + "0"  # Y orientation, str(0*np.sin((phi)/2))
                            + " "
                            + "0"  # Z orientation
                            + " "
                            + "0"  # W orientation
                            + " "
                            + "4"
                            + "\n"
                        )
                        file_to_write.write(line)

    def record_cell(
        self, index, parent, pos_x, pos_y, pos_z, creation_time, is_stem
    ):
        """We do not record the individual cell creations."""
        pass

    def record_final_state(
        self, tic, cells, cell_positions, active_cell_indexes
    ):
        """We already recorded the final state of the culture."""
        pass


class DfOutput(TumorsphereOutput):
    """Class for saving only the final state of the culture to a DataFrame."""

    def __init__(self, culture_name, output_dir="."):
        self.output_dir = output_dir
        self.culture_name = culture_name

    def begin_culture(
        self,
        prob_stem,
        prob_diff,
        rng_seed,
        simulation_start,
        adjacency_threshold,
        swap_probability,
    ):
        """
        Record the beginning of a simulation. We do nothing in this case.
        """
        pass

    def record_stemness(self, cell_index, tic, stemness):
        """
        Record a change in the stemness of a cell. We do nothing in this case.
        """
        pass

    def record_deactivation(self, cell_index, tic):
        """
        Record the deactivation of a cell. We do nothing in this case.
        """
        pass

    def record_culture_state(
        self,
        tic,
        cells,
        cell_positions,
        active_cell_indexes,
    ):
        """
        Record the state of the culture at a given time step. We do nothing in
        this case.
        """
        pass

    def record_cell(
        self, index, parent, pos_x, pos_y, pos_z, creation_time, is_stem
    ):
        """
        Record the creation of a new cell. We do nothing in this case.
        """
        pass

    def record_final_state(
        self, tic, cells, cell_positions, active_cell_indexes
    ):
        """
        Record the final state of the culture.

        This method is called at the end of the simulation, after the last time
        step, to record the final state of the culture. We record the position
        norm, the stemness, and the activity status.
        """
        # susceptibility = self.rng.random(size=len(self.cells))
        norms = np.linalg.norm(cell_positions, axis=1)
        data = {
            "position_norm": norms,
            "stemness": [],
            "active": [],
            # "susceptibility": [],  # susceptibility,
        }

        # we get the stemness and activity status of the cells
        for cell in cells:
            data["stemness"].append(cell.is_stem)
            data["active"].append(cell._index in active_cell_indexes)
            assert (
                cell._index in active_cell_indexes
            ) == cell.available_space

        # we make the dataframe
        df = pd.DataFrame(data)

        # we save the dataframe to a file
        filename = (
            f"{self.output_dir}/final_state_t={tic}_{self.culture_name}.csv"
        )
        df.to_csv(filename, index=False)


def create_output_demux(
    culture_name: str,
    requested_outputs: list[str],
    output_dir: str = ".",
    save_step_dat_pos_ar: int = 1,
    save_step_dat_order_par: int = 1,
    save_step_dat_motion_par: int = 1,
    save_step_dat_cluster_summary: int = 100,
    save_step_dat_cluster_raw: int = 1000,
    save_step_dat_deformation_par: int = 100,
    save_step_dat_local_order_summary: int = 100,
    save_step_dat_local_order_raw: int = 1000,
    save_step_ovito: int = 1,
):
    """Create an OutputDemux object with the requested output types."""
    output_types = {
        "sql": SQLOutput,
        "dat": DatOutput,
        "dat_pos_ar": DatOutput_position_aspectratio,
        "dat_order_par": DatOutput_order_parameters,
        "dat_motion_par": DatOutput_motion_parameters,
        "dat_cluster_par": DatOutput_cluster_parameters,
        "dat_deformation_par": DatOutput_deformation_parameters,
        "dat_local_order_par": DatOutput_local_order_parameters,
        "ovito": OvitoOutput,
        "df": DfOutput,
    }
    outputs = []
    for out in requested_outputs:
        if out in output_types:
            if out == "dat_pos_ar":
                outputs.append(
                    output_types[out](
                        culture_name, 
                        output_dir, 
                        save_step_dat_pos_ar
                    )
                )
            elif out == "dat_order_par":
                outputs.append(
                    output_types[out](
                        culture_name,
                        output_dir,
                        save_step_dat_order_par
                    )
                )
            elif out == "dat_motion_par":
                outputs.append(
                    output_types[out](
                        culture_name,
                        output_dir,
                        save_step_dat_motion_par,
                    )
                )
            elif out == "dat_cluster_par":
                outputs.append(
                    output_types[out](
                        culture_name=culture_name,
                        output_dir=output_dir,
                        save_step=save_step_dat_cluster_summary,
                        raw_save_step=save_step_dat_cluster_raw,
                    )
                )
            elif out == "dat_deformation_par":
                outputs.append(
                    output_types[out](
                        culture_name,
                        output_dir,
                        save_step_dat_deformation_par,
                    )
                )
            elif out == "dat_local_order_par":
                outputs.append(
                    output_types[out](
                        culture_name=culture_name,
                        output_dir=output_dir,
                        save_step=save_step_dat_local_order_summary,
                        raw_save_step=save_step_dat_local_order_raw,
                    )
                )
            elif out == "ovito":
                outputs.append(
                    output_types[out](
                        culture_name,
                        output_dir,
                        save_step_ovito
                    )
                )
            else:
                outputs.append(
                    output_types[out](
                        culture_name,
                        output_dir
                    )
                )
        else:
            logging.warning(f"Invalid output {out} requested")
    return OutputDemux(culture_name, outputs)
