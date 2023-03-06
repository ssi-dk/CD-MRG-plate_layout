import os
import pytest


from plate_layout.study import Study
from plate_layout.plate import QCPlate

config_folder = os.path.abspath(os.path.join(os.getcwd(), "sample_input/"))
config_file = "plate_config.toml"
config_path = os.path.join(config_folder, config_file)

records_folder = os.path.abspath(os.path.join(os.getcwd(), "sample_input/"))
records_file_xlsx = "fake_case_control_Npairs_523_Ngroups_5.xlsx"
records_path_xlsx = os.path.join(records_folder, records_file_xlsx)
records_file_csv = "fake_case_control_Npairs_523_Ngroups_5.csv"
records_path_csv = os.path.join(records_folder, records_file_csv)


# study CLASS
@pytest.fixture
def an_empty_study():
    study_name = "CNS_tumor_study"
    return Study(study_name)
    
    
@pytest.fixture
def my_study_with_records(an_empty_study):
    an_empty_study.load_specimen_records(records_path_xlsx)
    return an_empty_study
    
    
@pytest.fixture
def my_study(my_study_with_records):
    my_study_with_records.create_batches(QCPlate(config_path, (8, 12)))
    return my_study_with_records
    
    
def test_should_create_empty_study(an_empty_study):
    study_name = "CNS_tumor_study"
    my_study = Study(study_name)

    assert my_study.name == study_name


def test_should_import_study_records_csv():
    study_name = "CNS_tumor_study"
    my_study = Study(study_name)

    my_study.load_specimen_records(records_path_csv)
    
    assert my_study.specimen_records_df.empty is False
    
    
def test_should_import_study_records_xlsx():
    study_name = "CNS_tumor_study"
    my_study = Study(study_name)
    
    my_study.load_specimen_records(records_path_xlsx)

    assert my_study.specimen_records_df.empty is False
    
      
def test_should_fail_import_study_records():
    study_name = "CNS_tumor_study"
    my_study = Study(study_name)

    with pytest.raises(FileExistsError) as exception_info:
        my_study.load_specimen_records("missing_file")

    assert exception_info.value.args[0] == "missing_file"


def test_should_create_batches(my_study):
    assert len(my_study.batches) == 14


def test_should_get_plate(my_study):
    
    assert my_study[0].plate_id == 1
    assert my_study[4].plate_id == 5
    assert my_study[13].plate_id == 14
    
    
def test_should_assign_colors_to_wells_by_metadata(my_study):

    plate = my_study[0]
    rgbs = plate.define_metadata_colors(metadata_key="organ", colormap=plate._colormap)

    plate.assign_well_color(metadata_key="organ", colormap=plate._colormap)

    assert plate[0].color == rgbs["NaN"]
    assert plate[2].color == rgbs["Parotid glands"]
    assert plate[95].color == rgbs["Tendons"]


def test_should_write_plate_layout_to_txt_file(my_study):
    plate = my_study[0]
    plate.to_file()
    file_path = "Plate_1.txt"

    assert os.path.exists(file_path) is True


def test_should_write_plate_layout_to_format_from_file_extension(my_study):
    plate = my_study[0] 
    plate.to_file("Plate_1.csv")
    file_path = "Plate_1.csv"
    
    assert os.path.exists(file_path) is True
    

def test_should_write_plate_layout_to_file_given_metadata(my_study): 
    plate = my_study[0]
    metadata = ["QC", "pair_ID", "organ", "barcode"]
    file_path = "Plate_1.csv"
    plate.to_file(file_path=file_path, metadata_keys=metadata)
   
    assert os.path.exists(file_path) is True
    
#     #os.remove(file_path)