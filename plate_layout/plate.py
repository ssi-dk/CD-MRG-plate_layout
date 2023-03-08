import csv
import glob
import itertools
import logging
import os
import string
import tomli

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

# logging.basicConfig(level=logging.INFO, format='[%(name)s %(levelname)s] --- %(message)s')
logger = logging.getLogger(__name__)

# parameters governing how numpy arrays are printed to console
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

# TODO
# Try out Sphinx to generate automatic(?) docs for the module

# TODO
# Write "solution" class/namedtuple to store data pertaining to a solution such as name and volume and concentration 

# Add methods to add solutions to well keep track on volumes
# and amount of sample/reagents


class Well:
    """
    A class for keeping track of a well in a plate.
    """
    name: str
    coordinate: tuple[int, int]
    index : int
    color : tuple[float, float, float]
    plate_id : int
    metadata: dict

    def __init__(self, name, coordinate,
                 index=None, plate_id=None,
                 color=None, size=None, metadata=None):

        self.name = name
        self.coordinate = coordinate
        self.index = index
        self.plate_id = plate_id
        self.color = color
        self.size = size

        # We do this because of Python's treatment of mutable default arguments
        if metadata is None:
            metadata = {}

        self.metadata = metadata

    def __str__(self) -> str:
        return f"name: {self.name}\ncoordinate: {self.coordinate}\
            \nmetadata: {self.metadata}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, \
            coordinate={self.coordinate})"


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
                f"Setting up a default {self._default_n_rows*self._default_n_columns}\
                    -well plate.")
            self._n_rows = self._default_n_rows
            self._n_columns = self._default_n_columns

        if isinstance(plate_dim, tuple) or isinstance(plate_dim, list):
            if len(plate_dim) == 2:
                self._n_rows = plate_dim[0]
                self._n_columns = plate_dim[1]
            else:
                raise ValueError(
                    f"Unsupported plate format: {plate_dim}. Must be given as a tuple \
                        '(#rows, #columns)', list '[#rows, #columns]' or an integer \
                            '#wells in total'")

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
                f"Unsupported plate format: {plate_dim}. Must be given as a tuple \
                    '(#rows, #columns)', list '[#rows, #columns]' or an integer \
                        '#wells in total'")

        self.plate_id = plate_id
        self.rows = list(range(0, self._n_rows))
        self.columns = list(range(0, self._n_columns))
        self.capacity = self._n_rows * self._n_columns
        self._specimen_capacity = self.capacity

        self._coordinates = Plate.create_index_coordinates(self.rows, self.columns)
        self._row_labels, self._alphanumerical_coordinates = \
            Plate.create_alphanumerical_coordinates(self.rows, self.columns)

        self.define_empty_wells()
        self.create_plate_layout()

        logger.info(f"Created a plate template with {len(self)} wells:")
        logger.debug(f"Canonical well coordinates:\n{self}")
        # logger.debug(f"Well index coordinates:
        # \n{self.to_numpy_array(self._coordinates)}")

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

            logger.warning("No config file specified. \
                Trying to find a toml file in current folder.")

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
        well_name_to_index_map = {}\

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
        return f"{self.__class__.__name__}(({len(self.rows)},\
                        {len(self.columns)}),\
                            plate_id={self.plate_id})"

    def update_well_position(self, index):
        self.wells[index].name = self._alphanumerical_coordinates[index]
        self.wells[index].coordinate = self._coordinates[index]
        self.wells[index].metadata["index"] = index

    def to_numpy_array(self, data: list) -> object:

        plate_array = np.empty((len(self.rows), len(self.columns)),
                               dtype=object)

        if len(data) > self.capacity:
            Warning(
                f"Number of wells in plate is {self.capacity}, \
                    but data contains {len(data)} values")
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
                metadata[key].append(well.index)

        return metadata

    def get(self, metadata_key) -> list:

        if metadata_key == "names":
            return [well.name for well in self]
        elif metadata_key == "coordinates":
            return [well.coordinate for well in self]
        elif metadata_key == "color":
            return [well.color for well in self]
        else:
            return [well.metadata.get(metadata_key, "NaN") for well in self]

    def get_metadata_as_numpy_array(self, metadata_key: str) -> object:
        metadata = self.get(metadata_key)
        return self.to_numpy_array(metadata)

    def define_metadata_colors(self, metadata_key: str, colormap: str) -> dict:
        # get number of colors needed == number of (discrete) values \
        # represented in wells with metadata_key
        metadata_categories = np.unique(self.get(metadata_key))

        # convert to strings in case metadata is numeric
        metadata_categories = list(map(str, metadata_categories))

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
            f"Assigning {N_colors} colors from colormap {colormap} to metadata \
                '{metadata_key}' as:")

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
            well.color = RGB_colors[str(key)]

        return RGB_colors

    def to_figure(self, annotation_metadata_key=None,
                  color_metadata_key=None,
                  rotation: int = 45,
                  step=10,
                  title_str=None,
                  font_size=7,
                  alpha=0.6,
                  fig_height=12,
                  dpi=300,
                  plt_style="bmh",
                  grid_color=(1, 1, 1),
                  edge_color=(0.99, 0.99, 0.99),
                  legend_n_columns=5,
                  colormap=None,
                  marker='o',
                  filepath=None,
                  label_color=(0.5, 0.5, 0.5),
                  save=False,
                  ) -> object:

        if colormap is None:
            colormap = self._colormap

        # Define title
        if title_str is None:
            title_str = f"Plate {self.plate_id}, showing {annotation_metadata_key} \
                colored by {color_metadata_key}"

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

        # Get colors for each well
        well_colors = np.ravel(np.flipud(self.get_metadata_as_numpy_array("color")))
        well_colors = [mpl.colors.colorConverter.to_rgba(col, alpha=alpha) for col in well_colors]
        
        # PLOT WELLS AS SCATTER PLOT
        # Style and size
        #fontsize = 3/96 * self.capacity

        s = fig_height / 2
        height = 2.1*s
        width = 3*s
        cm = 1/2.54
        fig = plt.figure(figsize=(width*cm, height*cm), dpi=300)
        plt.style.use(plt_style)
        ax = fig.add_subplot(111)

        # Well size array
        # Size in pixels
        # get bounding box of ax
        width_px, height_px = ax.get_tightbbox().bounds[2:4]  # pixels
        well_size_px = np.power(width_px * height_px / self.capacity, 0.68)
        # well_width = np.sqrt(well_size_px)
        
        # size in grid
   
        well_width_grid = np.sqrt(x.max() * y.max() / self.capacity)

        size_grid = np.ones_like(Xgrid) * well_size_px

        # Create scatter plot
        ax.scatter(Xgrid, Ygrid,
                   s=size_grid,
                   c=well_colors,
                   edgecolors=edge_color,
                   linewidths=0.3*well_width_grid,
                   marker=marker)

        # Create annotations
        for well in self:
            x_i = Xgrid[well.coordinate]
            y_i = Ygrid[well.coordinate]

            annotation_label = well.metadata.get(annotation_metadata_key, "")

            ax.annotate(annotation_label, (x_i, y_i),
                        horizontalalignment='center',
                        verticalalignment='center',
                        rotation=rotation,
                        fontsize=font_size)

        # LEGENDS
        # Create dummy plot to map legends, save plot handles to list
        lh = []
        for key, color in RGB_colors.items():

            if key == "NaN":
                key = ""

            lh.append(ax.scatter([], [], well_size_px*0.4,
                                 color=color, label=key,
                                 alpha=alpha,
                                 edgecolors=edge_color,))

            # Add a legend
        legend_bb = (0.15, -0.1, 0.7, 0.7)
        # Adjust position depending on number of legend keys to show
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0*1.2, pos.width, pos.height])
        ax.legend(
            handles=lh,
            bbox_to_anchor=legend_bb,
            loc='lower center',
            frameon=False,
            labelspacing=1,
            ncol=legend_n_columns,
            fontsize=font_size
        )

        # FIG PROPERTIES
        crd_pad = 1.1*well_width_grid
        pad = 0.6*well_width_grid
        # X axis
        Xticklabels = [e+1 for e in self.columns]
        ax.set_xticks(x)
        ax.set_xticklabels(Xticklabels, fontsize=font_size, weight="bold", color=label_color)
        ax.xaxis.grid(color=grid_color, linestyle='None', linewidth=0.5)
        ax.set_xlim(x.min()-crd_pad, x.max()+pad)

        # Y axis
        ax.set_yticks(y)
        ax.set_yticklabels(self._row_labels[::-1], fontsize=font_size,  weight="bold", color=label_color)
        ax.yaxis.grid(color=grid_color, linestyle='None', linewidth=0.5)
        ax.set_ylim(y.min()-pad, y.max()+crd_pad)
        
        ax.tick_params(axis=u'both', which=u'both', length=0, direction="in")
        ax.xaxis.set_ticks_position('top')
        X_TICK_PADDING = -step*1.2
        Y_TICK_PADDING = -step*1.0

        xticks = [*ax.xaxis.get_major_ticks(), *ax.xaxis.get_minor_ticks()]
        yticks = [*ax.yaxis.get_major_ticks(), *ax.yaxis.get_minor_ticks()]

        for tick in xticks:
            tick.set_pad(X_TICK_PADDING)

        for tick in yticks:
            tick.set_pad(Y_TICK_PADDING)

        # Hide grid behind graph elements
        ax.set_axisbelow(True)

        ax.set_title(title_str, fontsize=12)
        
        # folder_path = os.getcwd()
        # file_format = "png"
        # file_name = f"plate_name_{self.plate_id}_{annotation_metadata_key}_{color_metadata_key}.{file_format}"

        # file_path = os.path.join(folder_path, file_name)

        # # Define title
        # title_str = f" Plate {self.plate_id}, showing {annotation_metadata_key} colored by {color_metadata_key}"

        # plt.gcf().set_size_inches(18*cm, 12*cm)
        
        # plt.savefig(file_path, format="png")
        # plt.tight_layout()
        if save:
            logger.info(f"Saving plate figure to {filepath}")
            plt.savefig(filepath, format="pdf")
            plt.close(fig)
        else:
            return fig

    def to_file(self, metadata_keys: list = None,
                file_path: str = None,
                file_format: str = "txt",
                ) -> None:

        if metadata_keys is None:
            metadata_keys = []

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
                to_write = f"{'well':{width2}}{'sample name':{width}}"
                for key in metadata_keys:
                    to_write += f"{key:<{width}}"
                file.write(to_write+"\n")

                # Write rows
                for well in self:
                    well_name = str().join(well.name.split("_"))
                    to_write = f"{well_name:{width2}}{well.metadata['sample_name']:{width}}"

                    for key in metadata_keys:
                        to_write += f"{str(well.metadata.get(key,'NaN')):<{width}}"

                    file.write(to_write+"\n")


# A plate with QC samples is a subclass of a Plate class
class QCPlate(Plate):
    """_summary_
    Class that represents a multiwell plate where some wells can
    contain quality control samples according to the scheme defined
    in QC_config; either a <config_file.toml> file or a dict following
    the same structure

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
                logger.error("No scheme for QC samples provided.")

    def __repr__(self):
        return f"{self.__class__.__name__}(({len(self.rows)},{len(self.columns)}), \
            plate_id={self.plate_id})"

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

        logger.info("Creating plate layout with QC samples.")

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

                sample_type = self.config["QC"]["names"][sample_code]
                self.wells[index].metadata["sample_type"] = sample_type
                self.wells[index].metadata["sample_name"] = \
                    f"{sample_code}{counts[sample_code]}"

        # set metadata for wells that are for specimen samples
        for i, index in enumerate(self._well_inds_specimens):
            self.wells[index].metadata["QC"] = False
            self.wells[index].metadata["sample_code"] = self._specimen_code
            self.wells[index].metadata["sample_type"] = self._specimen_base_name
            self.wells[index].metadata["sample_name"] = f"{self._specimen_code}{i+1}"
