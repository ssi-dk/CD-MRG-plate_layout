import csv
import glob
import itertools
import os
import string
import tomli

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

# from pl_logger import logger
# TODO have to change import depending on if we package the module or use
# it locally with cli...
from .logger import logger

# parameters governing how numpy arrays are printed to console
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

# TODO
# The well size in figures should dynamically be set depending on
# the number of wells in the plate. Currently it is an optional argument to
# the plot function.

# TODO
# Try out Sphinx to generate automatic(?) docs for the module


# TODO ?
# Add @dataclass decorator to Well
# - we can then get access to methods such as JSON conversion,
# nice representation, make immutable, __init__, __le__, __eq__, __ge__ etc
# could be useful to have nice syntax comparing well position

class Well:
    """
    A class for keeping track of a well in a plate. 
    """
    name: str
    coordinate: tuple[int, int]
    metadata: dict

    def __init__(self,
                 name="",
                 coordinate=(int, int),
                 index=None,
                 plate_id=None,
                 metadata=None,
                 color=None) -> None:

        self.name = name
        self.coordinate = coordinate

        if metadata is None:
            metadata = {"index": index,
                        "plate_id": plate_id,
                        "color": color}

        self.metadata = metadata

    def __str__(self) -> str:
        return f"name: {self.name}\ncoordinate: {self.coordinate}\nmetadata: {self.metadata}"

    def __repr__(self) -> str:
        return f"Well(name={self.name}, coordinate={self.coordinate})"


class Plate:

    """_summary_
    A class to represent a multiwell plate. 
    
    """

    # private default variables
    _specimen_code: str = "S"
    _specimen_base_name: str = "Specimen"
    _colormap: str = "tab20"  # for plotting using matplotlib
    _NaN_color: tuple = (1, 1, 1)
    _default_n_rows: int = 8
    _default_n_columns: int = 12

    # "public" variables
    plate_id: int
    rows: list
    columns: list
    wells: list  # list of well objects
    layout: object
    metadata: list  # available metadata in wells

    # "private" variables
    _coordinates: list  # list of tuples with a pair of ints (row, column)
    # list of strings for canonical naming of plate coordnates, i.e A1, A2, A3, ..., B1, B2, etc
    _alphanumerical_coordinates: list
    _coordinates2index: dict  # map well index to well coordinate
    _row_labels: list  # alphabet labels for rows
    _name2index: dict  # map well index to well name
    _itercount: int  # for use in __iter__
    _metadata_map: dict  # key = metadata, value= well index for wells containing the metadata key
    _n_rows: int
    _n_columns: int
    _specimen_capacity: int

    def __init__(self, plate_dim=None, plate_id=1):

        if plate_dim is None:
            logger.info(
                f"Setting up a default {self._default_n_rows*self._default_n_columns}-well plate.")
            self._n_rows = self._default_n_rows
            self._n_columns = self._default_n_columns

        if isinstance(plate_dim, tuple) or isinstance(plate_dim, list):
            if len(plate_dim) == 2:
                self._n_rows = plate_dim[0]
                self._n_columns = plate_dim[1]
            else:
                raise ValueError(
                    f"Unsupported plate format: {plate_dim}. Must be given as a tuple '(#rows, #columns)', list '[#rows, #columns]' or an integer '#wells in total'")

        elif isinstance(plate_dim, dict):
            self._n_rows = plate_dim.get("rows", self._default_n_rows)
            self._n_columns = plate_dim.get("columns", self._default_n_columns)

        elif isinstance(plate_dim, int):
            # design a plate in format 2 : 3 as in
            # 6, 12, 24, 48, 96, 384 or 1536 plates
            x = np.sqrt(plate_dim * 25 / 6)
            self._n_rows = int(np.round(2/5*x))
            self._n_columns = int(np.round(3/5*x))

        else:
            raise ValueError(
                f"Unsupported plate format: {plate_dim}. Must be given as a tuple '(#rows, #columns)', list '[#rows, #columns]' or an integer '#wells in total'")

        self.plate_id = plate_id
        self.rows = list(range(0, self._n_rows))
        self.columns = list(range(0, self._n_columns))
        self.capacity = self._n_rows * self._n_columns
        self._specimen_capacity = self.capacity

        self._coordinates = Plate.create_index_coordinates(self.rows, self.columns)
        self._row_labels, self._alphanumerical_coordinates = Plate.create_alphanumerical_coordinates(
            self.rows, self.columns)

        self.define_empty_wells()
        self.create_plate_layout()

        logger.info(f"Created a plate template with {len(self)} wells:")
        logger.debug(f"Canonical well coordinates:\n{self}")
        # logger.debug(f"Well index coordinates:\n{self.to_numpy_array(self._coordinates)}")

    @staticmethod
    def create_index_coordinates(rows, columns) -> list:
        # count from left to right, starting at well in top left
        return list(itertools.product(
            range(len(rows)-1, -1, -1),
            range(0, len(columns))
        )
        )

    @staticmethod
    def create_alphanumerical_coordinates(rows, columns) -> list:
        alphabet = list(string.ascii_uppercase)

        row_labels = alphabet
        number_of_repeats = 1

        while len(rows) > len(row_labels):
            row_labels = list(itertools.product(alphabet, repeat=number_of_repeats))
            number_of_repeats += 1

        alphanumerical_coordinates = []
        for r_i in rows:
            for c_i in columns:
                # an_crd = row_crds[r_i], column
                r_str = str()
                for r in row_labels[r_i]:
                    r_str = r_str.join(r)

                alphanumerical_coordinates.append(f"{r_str}_{c_i+1}")

        return row_labels[0:len(rows)], alphanumerical_coordinates

    @staticmethod
    def load_config_file(config_file: str = None) -> dict:

        # READ CONFIG FILE
        if config_file is None:

            logger.warning(
                "No config file specified. Trying to find a toml file in current folder.")

            config_file_search = glob.glob("*.toml")

            if config_file_search:
                config_file = config_file_search[0]
                logger.info(f"Using toml file '{config_file}'")

        try:
            with open(config_file, mode="rb") as fp:
                config = tomli.load(fp)

            logger.info(f"Successfully loaded config file {config_file}")
            logger.debug(f"{config}")

            return config

        except FileNotFoundError:
            logger.error(f"Could not find/open config file {config_file}")

            raise FileExistsError(config_file)

    def define_empty_wells(self):

        wells = []
        well_crd_to_index_map = {}
        well_name_to_index_map = {}

        for i, crd in enumerate(
            zip(self._coordinates, self._alphanumerical_coordinates)
        ):

            index_crd = crd[0]
            name_crd = crd[1]

            wells.append(
                Well(coordinate=index_crd,
                     name=name_crd,
                     index=i,
                     plate_id=self.plate_id,
                     color=self._NaN_color)
            )

            well_crd_to_index_map[index_crd] = i
            well_name_to_index_map[name_crd] = i

        self.wells = wells
        self._coordinates2index = well_crd_to_index_map
        self._name2index = well_name_to_index_map

    # LENGTH
    # len(plate) shuld return the number of wells the plate has defined = plate capacity
    def __len__(self) -> int:
        return len(self.wells)

    # ITERATOR
    # make class to work as an iteror, eg we can iterate over the wells in the class
    # "for well in plate|
    def __iter__(self) -> list:
        self._itercount = 0
        return self

    def __next__(self):
        if self._itercount < self.capacity:
            well_to_return = self.wells[self._itercount]
            self._itercount += 1
            return well_to_return
        else:
            raise StopIteration

    # IN
    # check if well is contained in plate
    def __contains__(self):
        return NotImplemented

    # Plate[index/coordinate/name] should return the well from plate
    def __getitem__(self, key) -> object:

        key_type = type(key)

        if key_type is str:
            index = self._name2index[key]
        elif key_type is int:
            index = key
        elif key_type is tuple:
            index = self._coordinates2index[key]
        else:
            raise KeyError(key)

        return self.wells[index]

    # Allow to set a new well object to Plate using Plate[] = Well(..)
    def __setitem__(self, key, well_object) -> None:

        key_type = type(key)

        if key_type is str:
            index = self._name2index[key]
        elif key_type is int:
            index = key
        elif key_type is tuple:
            index = self._coordinates2index[key]
        else:
            raise KeyError(key)

        position_different = self.wells[index].coordinate != well_object.coordinate

        self.wells[index] = well_object
        if position_different:
            self.update_well_position(index)

    def __delitem__(self):
        # TODO
        return NotImplemented

    # print(plate)
    def __str__(self):
        return f"{self.to_numpy_array(self._alphanumerical_coordinates)}"

    # plate
    def __repr__(self):
        return f"Plate(({len(self.rows)},{len(self.columns)}), plate_id={self.plate_id})"

    def update_well_position(self, index):
        self.wells[index].name = self._alphanumerical_coordinates[index]
        self.wells[index].coordinate = self._coordinates[index]
        self.wells[index].metadata["index"] = index

    def to_numpy_array(self, data: list) -> object:

        plate_array = np.empty((len(self.rows), len(self.columns)),
                               dtype=object)

        if len(data) > self.capacity:
            Warning(
                f"Number of wells in plate is {self.capacity}, but data contains {len(data)} values")
            return plate_array

        for i, well in enumerate(self._coordinates):
            r, c = well[0], well[1]
            plate_array[r][c] = data[i]

        return np.flipud(plate_array)

    def create_plate_layout(self):
        # set metadata for wells that are for specimen samples

        for i in range(0, self.capacity):
            self.wells[i].metadata["QC"] = False
            self.wells[i].metadata["sample_code"] = self._specimen_code
            self.wells[i].metadata["sample_type"] = self._specimen_base_name
            self.wells[i].metadata["sample_name"] = f"{self._specimen_code}{i+1}"

    @property
    def layout(self):
        return self.to_numpy_array(self.get("sample_name"))

    @property
    def metadata(self):
        return list(self._metadata_map.keys())

    @property
    def _metadata_map(self):
        metadata = {}
        for well in self:
            for key in well.metadata.keys():
                metadata.setdefault(key, [])
                metadata[key].append(well.metadata["index"])

        return metadata

    def get(self, metadata_key) -> list:

        if metadata_key == "names":
            return [well.name for well in self]
        elif metadata_key == "coordinates":
            return [well.coordinate for well in self]
        else:
            return [well.metadata.get(metadata_key, "NaN") for well in self]

    def get_metadata_as_numpy_array(self, metadata_key: str) -> object:
        metadata = self.get(metadata_key)
        return self.to_numpy_array(metadata)

    def define_metadata_colors(self, metadata_key: str, colormap: str) -> dict:
        # get number of colors needed == number of (discrete) values represented in wells with metadata_key
        metadata_categories = np.unique(self.get(metadata_key))

        N_colors = len(metadata_categories)

        RGB_colors = {}
        for i, s in enumerate(metadata_categories):
            # get RGB values for i-th color in colormap
            col = mpl.colormaps[colormap](i)[0:3]
            if (s != "NaN") and (pd.notnull(s)):
                RGB_colors.setdefault(str(s), col)
            else:
                RGB_colors.setdefault("NaN", self._NaN_color)

        logger.debug(
            f"Metadata '{metadata_key}' has {N_colors} values: {metadata_categories}.")
        logger.debug(
            f"Assigning {N_colors} colors from colormap {colormap} to metadata '{metadata_key}' as:")

        key_length = np.max(list(map(len, metadata_categories)))

        for key, value in RGB_colors.items():
            logger.debug(
                f"{key:{key_length}}: ({value[0]:.2f}, {value[1]:.2f}, {value[2]:.2f})")

        return RGB_colors

    def assign_well_color(self, metadata_key: str, colormap: str) -> dict:

        RGB_colors = self.define_metadata_colors(metadata_key, colormap)

        # assign well color for each well according to color scheme defined above
        for well in self:
            key = well.metadata.get(metadata_key, "NaN")
            if pd.isnull(key):
                key = "NaN"
            well.metadata["color"] = RGB_colors[str(key)]

        return RGB_colors

    def to_figure(self, annotation_metadata_key=None,
                  color_metadata_key=None,
                  fontsize: int = 8,
                  rotation: int = 0,
                  step=10,
                  title_str=None,
                  alpha=0.7,
                  well_size=1200,
                  fig_width=11.69,
                  fig_height=8.27,
                  dpi=100,
                  plt_style="bmh",
                  grid_color=(1, 1, 1),
                  edge_color=(0.5, 0.5, 0.5),
                  legend_bb=(0.15, -0.2, 0.7, 1.3),
                  legend_n_columns=6,
                  colormap=None,
                  ) -> object:

        if colormap is None:
            colormap = self._colormap

        # Define title
        if title_str is None:
            title_str = f"Plate {self.plate_id}, showing {annotation_metadata_key} colored by {color_metadata_key}"

        # DEFINE COLORS FOR METADATA VALUES
        RGB_colors = self.assign_well_color(color_metadata_key, colormap)

        # DEFINE GRID FOR WELLS
        # 1 - define the lower and upper limits
        minX, maxX, minY, maxY = 0, len(self.columns)*step, 0, len(self.rows)*step
        # 2 - create one-dimensional arrays for x and y
        x = np.arange(minX, maxX, step)
        y = np.arange(minY, maxY, step)
        # 3 - create a mesh based on these arrays
        Xgrid, Ygrid = np.meshgrid(x, y)

        # Well size array
        size_grid = np.ones_like(Xgrid) * well_size

        # Get colors for each well based on chosen metadata key (from assign_well_color)
        well_colors = np.ravel(np.flipud(self.get_metadata_as_numpy_array("color")))

        # PLOT WELLS AS SCATTER PLOT
        # Style and size
        fig = plt.figure(dpi=dpi)
        plt.style.use(plt_style)
        fig.set_size_inches(fig_width, fig_height)
        ax = fig.add_subplot(111)

        # Create scatter plot
        ax.scatter(Xgrid, Ygrid,
                   s=size_grid,
                   c=well_colors,
                   alpha=alpha,
                   edgecolors=edge_color)

        # Create annotations
        for well in self:
            x_i = Xgrid[well.coordinate]
            y_i = Ygrid[well.coordinate]

            annotation_label = well.metadata.get(annotation_metadata_key, "NaN")

            ax.annotate(annotation_label, (x_i, y_i),
                        horizontalalignment='center',
                        verticalalignment='center',
                        rotation=rotation,
                        fontsize=fontsize)

        # LEGENDS
        # Create dummy plot to map legends, save plot handles to list
        lh = []
        for key, color in RGB_colors.items():
            lh.append(ax.scatter([], [], well_size*0.8,
                                 color=color, label=key,
                                 alpha=alpha,
                                 edgecolors=edge_color))

            # Add a legend
        # Adjust position depending on number of legend keys to show
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0*2, pos.width, pos.height*0.8])
        ax.legend(
            handles=lh,
            bbox_to_anchor=legend_bb,
            loc='lower center',
            frameon=False,
            labelspacing=1,
            ncol=legend_n_columns
        )

        # FIG PROPERTIES
        # X axis
        ax.set_xticks(x)
        ax.set_xticklabels(self.columns)
        ax.xaxis.grid(color=grid_color, linestyle='dashed', linewidth=1)
        ax.set_xlim(-1*x.max()*0.05, x.max()*1.05)

        # Y axis
        ax.set_yticks(y)
        ax.set_yticklabels(self._row_labels[::-1])
        ax.yaxis.grid(color=grid_color, linestyle='dashed', linewidth=1)
        ax.set_ylim(-1*y.max()*0.07, y.max()*1.07)

        # Hide grid behind graph elements
        ax.set_axisbelow(True)

        ax.set_title(title_str)

        return fig

    def to_file(self, file_path: str = None,
                file_format: str = "txt",
                metadata_keys: list = []) -> None:

        if file_path is None:
            file_name = f"Plate_{self.plate_id}.{file_format}"
            file_path = os.path.join(os.getcwd(), file_name)

        # file_path is a directory
        if os.path.isdir(file_path):
            file_name = f"Plate_{self.plate_id}.{file_format}"
            file_path = os.path.join(file_path, file_name)
        else:
            file_extension = os.path.splitext(file_path)[1]

            if not file_extension:
                file_path += f".{file_format}"
            else:  # deduce file format to use from file extenstion
                file_format = file_extension[1::]  # don't include . in name

        logger.info(f"Writing to file:\n\t{file_path}")

        if file_format != "txt":

            dialect = 'unix'
            delimiter = ","
            if file_format == "tsv":
                delimiter = "\t"
            elif file_format == "xlsx" or "xls":
                dialect = "excel"
            else:
                raise RuntimeWarning(file_format)

            with open(file_path, "w", newline="") as file:
                writer = csv.writer(file,
                                    delimiter=delimiter,
                                    lineterminator="\n",
                                    dialect=dialect,
                                    quoting=csv.QUOTE_NONE,)

                # Create and write column headers
                to_write = ["well", "sample_name"]
                for key in metadata_keys:
                    to_write.append(key)
                writer.writerow(to_write)

                # Write rows
                for well in self:
                    well_name = str().join(well.name.split("_"))
                    to_write = [well_name, well.metadata["sample_name"]]

                    for key in metadata_keys:
                        to_write.append(well.metadata.get(key, "NaN"))

                    writer.writerow(to_write)
        else:  # default: write to text file
            width = 20
            width2 = 10
            with open(file_path, "w", newline="\n") as file:

                # Create and write column headers
                to_write = f"{'well':<{width2}}{'sample name':<{width}}"
                for key in metadata_keys:
                    to_write += f"{key:<{width}}"
                file.write(to_write+"\n")

                # Write rows
                for well in self:
                    well_name = str().join(well.name.split("_"))
                    to_write = f"{well_name:<{width2}}{well.metadata['sample_name']:<{width}}"

                    for key in metadata_keys:
                        to_write += f"{str(well.metadata.get(key,'NaN')):<{width}}"

                    file.write(to_write+"\n")


# A plate with QC samples is a subclass of a Plate class
class QCPlate(Plate):
    """_summary_
    Class that represents a multiwell plate where some wells can 
    contain quality control samples according to the scheme defined 
    in QC_config; either a <config_file.toml> file or a dict following the same structure

    Args:
        Plate (_type_): _description_
    """

    config: dict
    _QC_unique_seq: list
    _well_inds_QC: list
    _well_inds_specimens: list

    def __init__(self, QC_config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if QC_config is not None:

            if isinstance(QC_config, dict):
                self.config = QC_config
            else:
                self.config = Plate.load_config_file(QC_config)

            if self.config is not None:
                self.create_QC_plate_layout()

            else:
                logger.error(f"No scheme for QC samples provided.")

    def __repr__(self):
        return f"{self.__class__.__name__}(({len(self.rows)},{len(self.columns)}), plate_id={self.plate_id})"

    def define_unique_QC_sequences(self):
        """_summary_

        """

        # Check if QC scheme defined in config file
        logger.debug("Setting up QC scheme from config file")

        self.N_specimens_between_QC = self.config['QC']['run_QC_after_n_specimens']
        QCscheme = self.config['QC']['scheme']

        self.N_QC_samples_per_round = np.max(
            [QCscheme[key]['position_in_round'] for key in QCscheme.keys()]
        )

        self.N_unique_QCrounds = np.max(
            [QCscheme[key]['every_n_rounds'] for key in QCscheme.keys()]
        )

        # Generate QC sequence list in each unique QC round
        QC_unique_seq = []
        for m in range(0, self.N_unique_QCrounds):

            m_seq = []

            for key in QCscheme:
                order = QCscheme[key]['position_in_round']

                if QCscheme[key]["introduce_in_round"] == m + 1:
                    m_seq.append((key, order))

                if QCscheme[key]["introduce_in_round"] == 0:
                    m_seq.append((key, order))

            m_seq.sort(key=lambda x: x[1])
            QC_unique_seq.append(m_seq)

        logger.debug(f"{len(QC_unique_seq)} unique QC sequences defined: ")
        for i, qc_seq in enumerate(QC_unique_seq):
            logger.debug(f"\t{i+1}) {qc_seq}")

        self._QC_unique_seq = QC_unique_seq

    def define_QC_rounds(self):

        # Number of QC rounds per plate
        N_QC_rounds = self.capacity // (self.N_QC_samples_per_round +
                                        self.N_specimens_between_QC - 1)

        logger.debug(f"Assigning {N_QC_rounds} QC rounds per plate")

        # Create dict with key = QC round;
        # value = a dict with well indices and associated sequence of QC samples

        QC_rounds = {}
        well_count_start = 0
        round_count = 0
        QC_well_indices = []

        while round_count < N_QC_rounds:

            if not self.config["QC"]["start_with_QC_round"]:
                well_count_start += self.N_specimens_between_QC

            for sequence in self._QC_unique_seq:
                well_indices = list(range(well_count_start,
                                          well_count_start + len(sequence)))
                round_count += 1

                QC_rounds[round_count] = {"sequence": sequence,
                                          "well_index": well_indices}
                QC_well_indices += well_indices

                well_count_start += len(sequence)
                well_count_start += self.N_specimens_between_QC

                logger.debug(
                    f"\t Round {round_count}: {sequence}; wells {well_indices}")

        self._QC_rounds = QC_rounds
        self._well_inds_QC = QC_well_indices
        self._well_inds_specimens = [w for w in range(
            0, self.capacity) if w not in QC_well_indices]
        self._specimen_capacity = len(self._well_inds_specimens)

    def create_QC_plate_layout(self):

        logger.info(f"Creating plate layout with QC samples.")

        self.define_unique_QC_sequences()
        self.define_QC_rounds()

        counts = {}
        for qc_type in self.config["QC"]["names"].keys():
            counts.setdefault(qc_type, 0)

        # set metadata for wells that are for QC samples
        for val in self._QC_rounds.values():
            for sample, index in zip(val["sequence"], val["well_index"]):
                self.wells[index].metadata["QC"] = True

                sample_code = sample[0]
                counts[sample_code] += 1
                self.wells[index].metadata["sample_code"] = sample_code

                sample_type = self.config["QC"]["names"][sample[0]]
                self.wells[index].metadata["sample_type"] = sample_type
                self.wells[index].metadata["sample_name"] = f"{sample[0]}{counts[sample_code]}"

        # set metadata for wells that are for specimen samples
        for i, index in enumerate(self._well_inds_specimens):
            self.wells[index].metadata["QC"] = False
            self.wells[index].metadata["sample_code"] = self._specimen_code
            self.wells[index].metadata["sample_type"] = self._specimen_base_name
            self.wells[index].metadata["sample_name"] = f"{self._specimen_code}{i+1}"
