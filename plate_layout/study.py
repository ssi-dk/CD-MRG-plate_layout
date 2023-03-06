import datetime
import os


import numpy as np
import pandas as pd

import logger


class Study:
    """_summary_

    Raises:
        StopIteration: _description_
        FileExistsError: _description_

    Returns:
        _type_: _description_
    """
    
    name = str
    plate_layout = object
    QC_config_file = str
    batches : list = []
    
    specimen_records_df : object = pd.DataFrame()
    
    _batch_count : int = 0
    _iter_count : int = 0
    _seed = 1234 # seed number for the random number generator in case randomization of specimens should be reproducible
    _N_permutations : int = 0
    _column_with_group_index : str = ""
    
    def __init__(self, 
                 study_name=None, 
                 ):
        
        if study_name is None:
            study_name = f"Study_{datetime.date}"
            
        self.name = study_name
        
        
    def __iter__(self) -> None:
        self._iter_count = 0
        return self
        
        
    def __next__(self) -> object:
        if self._iter_count < self.N_batches:
            plate_to_return = self.batches[self._iter_count]
            self._iter_count += 1
        else:
            raise StopIteration
        
        return plate_to_return
    
    
    def __len__(self):
        return len(self.batches)
        
    
    def __repr__(self):
        return f"Study({self.name})"


    def __str__(self):
        return f"{self.name}\n {self.study_specimens} on {self.N_batches}"

    
    def __getitem__(self, index):
        return self.batches[index]
    
    
    def load_specimen_records(self, records_file : str):
        
        self.records_file_path = records_file
        
        logger.debug(f"Loading records file: {records_file}")
        extension = os.path.splitext(records_file)[1]
        
        if not os.path.exists(records_file):
            logger.error(f"Could not find file{records_file}")
            raise FileExistsError(records_file)
        
        if extension == ".xlsx" or extension == ".xls":
            logger.debug(f"Importing Excel file.")
            records = pd.read_excel(records_file, )
        elif extension == ".csv":
            logger.debug(f"Importing csv file.")
            records = pd.read_csv(records_file, )
        else:
            logger.error(f"File extension not recognized")
            records = pd.DataFrame()
               
        self._column_with_group_index = Study.find_column_with_group_index(records)
        
        logger.debug(f"{records.shape[0]} specimens in file")
        logger.info("Metadata in file:")
        for col in records.columns:
            logger.info(f"\t{col}")
        
        
        if self._column_with_group_index:
            logger.debug(f"Sorting records in ascending order based on column '{self._column_with_group_index}'")
            records = records.sort_values(by=[self._column_with_group_index])
            
        self.specimen_records_df = records
             
    
    def add_specimens_to_plate(self, study_plate: object, specimen_samples_df: object) -> object:
        
        logger.debug(f"Adding samples to plate {study_plate.plate_id}")
        columns = specimen_samples_df.columns
        
        # keep track on how many wells we should use per batch
        N_specimens_left = len(specimen_samples_df)
        plate_specimen_count = 0
        
        for i, well in enumerate(study_plate):
            
            if well.metadata["sample_code"] == "S": 
                # add metadata key (and values) for each column in dataframe
                for col in columns:
                    well.metadata[col] = specimen_samples_df[col][plate_specimen_count]
                    
                plate_specimen_count += 1
                
            else:
                # add metadata key and nan value for each column in dataframe
                for col in columns:
                    well.metadata[col] = "NaN"
                    
            study_plate[i] = well
            
            if plate_specimen_count >= N_specimens_left:
                    logger.debug(f"\t -> Done. Last specimen placed in {well.name}")
                    break
                
        return study_plate
                
        # --- END OF FOOR LOOP ---
    
    
    def to_layout_lists(self, metadata_keys: list = None, 
                        file_format : str = "txt",
                        folder_path : str = None,
                        plate_name : str = "Plate") -> None:
        
        if folder_path is None: 
            folder_path = os.getcwd()
        
        for plate in self:
            file_name = f"{self.name}_{plate_name}_{plate.plate_id}"
            file_path = os.path.join(folder_path, file_name)
            
            plate.to_file(file_path=file_path,
                          file_format=file_format,
                          metadata_keys=metadata_keys)
    
    
    def to_layout_figures(self,
                          annotation_metadata_key : str,
                          color_metadata_key : str,
                        file_format : str = "pdf",
                        folder_path : str = None,
                        plate_name : str = "Plate", **kwargs) -> None:
        
        if folder_path is None: 
            folder_path = os.getcwd()
            
        for plate in self:
            file_name = f"{self.name}_{plate_name}_{plate.plate_id}_{annotation_metadata_key}_{color_metadata_key}.{file_format}"
            file_path = os.path.join(folder_path, file_name)
            
            # Define title        
            title_str = f"{self.name}: Plate {plate.plate_id}, showing {annotation_metadata_key} colored by {color_metadata_key}"
           
            fig = plate.to_figure(annotation_metadata_key, color_metadata_key, title_str=title_str, **kwargs)
    
            logger.info(f"Saving plate figure to {file_path}")
            
            plt.savefig(file_path)
    
    
    def create_batches(self, plate_layout : object) -> None: 
            
        batch_count = 1
        batches = []
        
        # get specimen data from study list
        specimen_df_copy = self.specimen_records_df.copy()
        
        while specimen_df_copy.shape[0] > 0:

            study_plate = copy.deepcopy(plate_layout)
            study_plate.plate_id = batch_count
                            
            # extract max specimen samples that will fit on plate; select from top and remove them from original DF
            sel = specimen_df_copy.head(study_plate._specimen_capacity)
            specimen_df_copy.drop(index=sel.index, inplace=True) 
            
            # reset index to so that rows always start with index 0
            sel.reset_index(inplace=True, drop=True)
            specimen_df_copy.reset_index(inplace=True, drop=True)

            # add specimen to plate
            study_plate = self.add_specimens_to_plate(study_plate, sel)
            
            batches.append(study_plate)
            
            batch_count += 1

        # --- END OF WHILE LOOP ---
        
        self.batches = batches
        self.N_batches = batch_count - 1

        logger.info(f"Finished distributing samples to plates; {self.N_batches} batches created.")
       
        
    @staticmethod
    def find_column_with_group_index(specimen_records_df) -> str:
        # Select columns that are integers; currently we can only identify groups based on pair _numbers_ 
        int_cols = specimen_records_df.select_dtypes("int")
                    
        logger.debug(f"Looking for group index of study pairs in the following table columns:")
        
        for col_name in int_cols.columns:
            
            logger.debug(f"\t\t{col_name}")
            
            # sort in ascending order
            int_col = int_cols[col_name].sort_values()
            # compute difference: n_1 - n_2, n_2 - n_3, ...
            int_diffs = np.diff(int_col)
            # count instances were numbers were the same, i.e. diff == 0
            n_zeros = np.sum(list(map(lambda x: x==0, int_diffs)))
            # we assume column contains pairs if #pairs == #samples / 2
            column_have_pairs = n_zeros == (int_col.shape[0]//2)

            if column_have_pairs:# we found a column so let's assume it is the correct one
                logger.info(f"Found group index in column {col_name}")
                return col_name
            
        return "" 
    
    
    def randomize_order(self, case_control : bool = None, reproducible=True):
        
        if not len(self.specimen_records_df) > 0:
            logger.error("There are no study records loaded. Use 'load_specimen_records' method to import study records.")
            return
        
        if case_control is None:
            if self._column_with_group_index:
                case_control = True
            else:
                case_control = False
        
        specimen_records_df_copy = self.specimen_records_df.copy()
        
        if case_control:
            column_with_group_index = self._column_with_group_index
                        
            logger.info(f"Randomly permuting group order (samples within group unchanged) using variable '{column_with_group_index}'")
            logger.debug("Creating multiindex dataframe")
            specimen_records_df_copy = specimen_records_df_copy.set_index([column_with_group_index, specimen_records_df_copy.index])
            drop = False
        else:
            logger.info(f"Randomly permuting sample order.")
            specimen_records_df_copy = specimen_records_df_copy.set_index([specimen_records_df_copy.index, specimen_records_df_copy.index])
            column_with_group_index = 0
            drop = True
            
            
        group_IDs = np.unique(specimen_records_df_copy.index.get_level_values(0))

        # Permute order in table
        if reproducible:
            logger.info("Using a fixed seed to random number generator for reproducibility; \
                running this method will always give the same result.")
            logger.debug(f"Using class-determined seed {self._seed} for random number generator")
            np.random.seed(self._seed)

        permutation_order = np.random.permutation(group_IDs)
        
        prev_index_str = "index_before_permutation"
        
        # if multiple randomization rounds, remove old column = prev_index_str 
        if prev_index_str in specimen_records_df_copy.columns:
            specimen_records_df_copy = specimen_records_df_copy.drop(columns=prev_index_str)
        
        specimen_records_df_copy = specimen_records_df_copy \
                                    .loc[permutation_order]\
                                    .reset_index(level=column_with_group_index, drop=drop)\
                                    .reset_index(drop=False)
        
        specimen_records_df_copy = specimen_records_df_copy.rename(columns = {"index": "index_before_permutation"})

        self._N_permutations += 1
        self.specimen_records_df = specimen_records_df_copy.copy()
       