import pytest
import os

from plate_layout.plate import Plate
from plate_layout.plate import Well
from plate_layout.plate import QCPlate

config_folder = os.path.abspath(os.path.join(os.getcwd(), "sample_input/"))
config_file = "plate_config.toml"
config_path = os.path.join(config_folder, config_file)


# @pytest.fixture


# Well CLASS
def test_should_create_well():
    well = Well("A0", (7, 0))

    assert well.name == "A0"
    assert well.coordinate == (7, 0)


# Plate CLASS
def test_should_create_plate_from_tuple():
    plate_96 = Plate((8, 12))

    assert plate_96.plate_id == 1
    assert len(plate_96.rows) == 8
    assert len(plate_96.columns) == 12


def test_should_create_plate_from_dict_args():
    plate_dim = {"rows": 8,
                 "columns": 12}

    plate_96 = Plate(plate_dim)

    assert len(plate_96.rows) == 8
    assert len(plate_96.columns) == 12


def test_should_create_plate_id_from_key_word_args():
    plate_dim = {"rows": 8,
                 "columns": 12}

    plate_96 = Plate(plate_dim, plate_id=5)
    assert plate_96.plate_id == 5


def test_should_create_plate_index_coordinates():
    plate_96 = Plate((8, 12))

    plate_96._coordinates = Plate.create_index_coordinates(
        plate_96.rows, plate_96.columns)

    assert plate_96._coordinates[0] == (7, 0)
    assert plate_96._coordinates[11] == (7, 11)
    assert plate_96._coordinates[84] == (0, 0)
    assert plate_96._coordinates[95] == (0, 11)


def test_should_create_plate_alphanumerical_coordinates():
    plate_96 = Plate((8, 12))

    plate_96._alphanumerical_coordinates = Plate.create_alphanumerical_coordinates(
        plate_96.rows, plate_96.columns)[1]

    assert plate_96._alphanumerical_coordinates[0] == "A_1"
    assert plate_96._alphanumerical_coordinates[11] == "A_12"
    assert plate_96._alphanumerical_coordinates[84] == "H_1"
    assert plate_96._alphanumerical_coordinates[95] == "H_12"


@pytest.mark.dependancy(depends=[test_should_create_well,
                                 test_should_create_plate_index_coordinates,
                                 test_should_create_plate_alphanumerical_coordinates])
def test_should_create_well_objects_in_plate():
    plate_96 = Plate((8, 12))

    plate_96._coordinates = Plate.create_index_coordinates(
        plate_96.rows, plate_96.columns)
    plate_96._alphanumerical_coordinates = Plate.create_alphanumerical_coordinates(
        plate_96.rows, plate_96.columns)[1]
    Plate.define_empty_wells(plate_96)

    well = plate_96.wells[0]
    assert well.name == "A_1"
    assert well.index == 0
    assert well.coordinate == (7, 0)

    well = plate_96.wells[95]
    assert well.name == "H_12"
    assert well.index == 95
    assert well.coordinate == (0, 11)


def test_should_return_plate_capacity():
    plate_96 = Plate((8, 12))
    assert plate_96.capacity == 96


@pytest.mark.dependancy(depends=[test_should_create_well_objects_in_plate])
def test_should_return_length_of_plate():
    plate_96 = Plate((8, 12))
    assert len(plate_96) == 96


def test_should_get_well_by_name():
    plate_96 = Plate((8, 12))

    assert plate_96['A_1'].name == plate_96.wells[0].name
    assert plate_96['A_1'].coordinate == plate_96.wells[0].coordinate

    assert plate_96['B_12'].name == plate_96.wells[23].name
    assert plate_96['B_12'].coordinate == plate_96.wells[23].coordinate

    assert plate_96['H_1'].name == plate_96.wells[84].name
    assert plate_96['H_1'].coordinate == plate_96.wells[84].coordinate

    assert plate_96['H_12'].name == plate_96.wells[95].name
    assert plate_96['H_12'].coordinate == plate_96.wells[95].coordinate


def test_should_get_well_by_coordinate():
    plate_96 = Plate((8, 12))

    assert plate_96[(7, 0)].name == plate_96[0].name
    assert plate_96[(7, 0)].coordinate == plate_96[0].coordinate

    assert plate_96[(6, 11)].name == plate_96[23].name
    assert plate_96[(6, 11)].coordinate == plate_96[23].coordinate

    assert plate_96[(0, 0)].name == plate_96[84].name
    assert plate_96[(0, 0)].coordinate == plate_96[84].coordinate

    assert plate_96[(0, 11)].name == plate_96[95].name
    assert plate_96[(0, 11)].coordinate == plate_96[95].coordinate


def test_should_get_well_by_index():
    plate_96 = Plate((8, 12))

    assert plate_96[0].name == plate_96[0].name
    assert plate_96[0].coordinate == plate_96[0].coordinate

    assert plate_96[23].name == plate_96[23].name
    assert plate_96[23].coordinate == plate_96[23].coordinate

    assert plate_96[84].name == plate_96[84].name
    assert plate_96[84].coordinate == plate_96[84].coordinate

    assert plate_96[95].name == plate_96[95].name
    assert plate_96[95].coordinate == plate_96[95].coordinate


def test_should_set_new_well_by_name():
    plate_96 = Plate((8, 12))
    test_well = Well("A_12", (7, 11))
    test_well.metadata["barcode"] = 1666

    assert plate_96[11].metadata.get("barcode", 0) != 1666

    plate_96["A_12"] = test_well

    assert plate_96[11].name == "A_12"
    assert plate_96[11].metadata["barcode"] == 1666


def test_should_set_new_well_by_coordinate():
    plate_96 = Plate((8, 12))
    test_well = Well("A_12", (7, 11))
    test_well.metadata["barcode"] = 1666

    assert plate_96[11].metadata.get("barcode", 0) != 1666

    plate_96[(7, 11)] = test_well

    assert plate_96[11].name == "A_12"
    assert plate_96[11].metadata["barcode"] == 1666


def test_should_set_new_well_by_index_and_update_well_position():
    plate_96 = Plate((8, 12))
    test_well = Well("G_1", (15, 15))

    plate_96[11] = test_well

    assert plate_96[11].name == "A_12"
    assert plate_96[(7, 11)].coordinate == (7, 11)


def test_should_get_well_names():
    plate_96 = Plate((8, 12))

    well_names = plate_96.get("names")

    assert well_names[0] == "A_1"
    assert well_names[11] == "A_12"
    assert well_names[84] == "H_1"
    assert well_names[95] == "H_12"


def test_should_get_well_coordinates():
    plate_96 = Plate((8, 12))

    well_crd = plate_96.get("coordinates")

    assert well_crd[0] == (7, 0)
    assert well_crd[11] == (7, 11)
    assert well_crd[84] == (0, 0)
    assert well_crd[95] == (0, 11)


def test_should_get_well_metadata():
    assert 1 == 2, "Not implemented"


def test_should_show_nan_in_get_for_missing_well_keys():
    assert 1 == 2, "Not implemented"


# QCplate SUBCLASS
def test_should_raise_error_on_missing_config_file():

    with pytest.raises(FileExistsError) as exception_info:
        print(exception_info)
        qcplate = QCPlate("my_config", (8, 12), plate_id=3)

    assert exception_info.value.args[0] == "my_config"


def test_should_create_QCplate():
    qcplate = QCPlate(config_path, (8, 12), plate_id=3)

    assert qcplate.config["QC"]["run_QC_after_n_specimens"] == 11
    assert qcplate.plate_id == 3
    assert qcplate._QC_rounds[1]["sequence"][0] == ("PB", 1)
    assert qcplate._QC_rounds[1]["sequence"][1] == ("EC", 2)


def test_should_get_well_QC_info_from_well_name():
    qcplate = QCPlate(config_path, (8, 12), plate_id=3)

    assert qcplate["A_1"].metadata["QC"] is True


def test_should_get_well_QC_info_from_well_coord():
    qcplate = QCPlate(config_path, (8, 12), plate_id=3)

    assert qcplate[(0, 7)].metadata["QC"] is True
