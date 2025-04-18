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
            filename = (
                f"{self.output_dir}/{self.culture_name}_step={tic:05}.dat"
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
            path_to_write = os.path.join(
                self.output_dir, f"ovito_data_{self.culture_name}.{tic:05}"
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
                        phi = cell.culture.cell_phies[cell._index]
                        line = (
                            (
                                "active_stem "
                                if cell.culture.cell_phies[cell._index] is None
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
                            + f"{1 if phi is None else phi % (2 * np.pi)}"  # color
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
    save_step_ovito: int = 1,
):
    """Create an OutputDemux object with the requested output types."""
    output_types = {
        "sql": SQLOutput,
        "dat": DatOutput,
        "dat_pos_ar": DatOutput_position_aspectratio,
        "ovito": OvitoOutput,
        "df": DfOutput,
    }
    outputs = []
    for out in requested_outputs:
        if out in output_types:
            if out == "dat_pos_ar":
                outputs.append(output_types[out](culture_name, output_dir, save_step_dat_pos_ar))
            elif out == "ovito":
                outputs.append(output_types[out](culture_name, output_dir, save_step_ovito))
            else:
                outputs.append(output_types[out](culture_name, output_dir))
        else:
            logging.warning(f"Invalid output {out} requested")
    return OutputDemux(culture_name, outputs)
