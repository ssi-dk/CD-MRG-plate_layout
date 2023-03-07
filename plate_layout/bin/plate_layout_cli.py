
from plate_layout.study import Study
from plate_layout.plate import Plate
from plate_layout.plate import QCPlate

from datetime import date

import logging
import argparse
import os

logging.basicConfig(level=logging.INFO, format='[%(name)s %(levelname)8s] --- %(message)s')
logger = logging.getLogger(__name__)


def setup_option_parser(parser):

    parser.add_argument("--study-file",
                        required=True,
                        help="csv/excel file with study specimen samples")

    parser.add_argument("--randomize",
                        default="no",
                        choices=['yes', 'no'],
                        help="randomize specimen order (if sample groups \
                            defined in study file, the order of _the \
                                groups_ will be randomized.")

    parser.add_argument("--plate-size",
                        default='96',
                        help="#wells, e.g. '6', '12', '24', '48', 96, \
                            384, 1536,... . The number of wells will \
                                be changed if it does not fit in a 2:3 format")

    parser.add_argument("--qc-file",
                        help="<path> to a 'toml' file with QC scheme parameters: \
                            <path/qc_config.toml>")

    parser.add_argument("--name",
                        help="name to include in output files.")

    parser.add_argument("--output-folder",
                        help="folder for saving layout files and figures. ")

    parser.add_argument("--log-level",
                        choices=['info', 'debug'],
                        default='debug',
                        help="level of information printed to console")

    parser.add_argument("--export-data",
                        choices=['all', 'lists', 'figures', 'off'],
                        default='all',
                        help="set which data should be exported to file")

    parser.add_argument("--metadata", help="Column names (metadata) from \
                        the study input file that will be printed to lists.\
                        The last two will be used when rendering figures, \
                        where the second last annotates the plate by color \
                        and the last annotates by value.", nargs='+')

    parser.add_argument("--file-format",
                        choices=['txt', 'csv', 'xlsx'],
                        default='text',
                        help="file format for plate layout lists"
                        )

    parser.add_argument("--figure-format",
                        choices=['pdf', 'png'],
                        default='pdf',
                        help="file format for plate layout figures"
                        )

    return parser.parse_args()


def main():
    """Entry point for the plate layout CLI."""

    description = """

    Create (specimen and/or QC) sample layout lists/figures for multiwell
    plates from a csv/excel with specimen data.

    Specimens assigned to groups will automatically be placed together on\
    the plate.

    """

    epilog = """

    EXAMPLE
plate_layout
        --study-file sample_input/fake_case_control_Npairs_523_Ngroups_5.csv \
        --randomize yes \
        --name my_fake_study \
        --qc-file sample_input/plate_config.toml \
        --metadata organ barcode

    """

    parser = argparse.ArgumentParser(description=description,
                                     epilog=epilog,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    op = setup_option_parser(parser)

    print(op)
    if op.log_level == 'debug':
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.debug("Running plate_layout in CLI mode")

    if not op.name:
        op.name = f"Study_{date.today()}"
        
    if not op.metadata:
        op.metadata = (None, None)

    study = Study(op.name)
    study.load_specimen_records(op.study_file)

    if op.randomize == "yes":
        study.randomize_order()

    if op.qc_file:
        plate = QCPlate(op.qc_file, int(op.plate_size))
    else:
        plate = Plate(int(op.plate_size))

    study.create_batches(plate)

    if not op.output_folder:
        op.output_folder = os.path.join(os.getcwd(), "plate_layouts/")
        if not os.path.exists(op.output_folder):
            logger.info("Creating folder ")
            os.mkdir(op.output_folder)

    study.to_layout_lists(metadata_keys=op.metadata, folder_path=op.output_folder)
    study.to_layout_figures(
        op.metadata[1], op.metadata[0], folder_path=op.output_folder)