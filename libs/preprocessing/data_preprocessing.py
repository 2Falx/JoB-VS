import os
from pathlib import Path

from global_features import Global_features, Global_features_general
from task_preprocessing import Preprocess_datasets
from image_sizes import Calculate_sizes

data_root = '/home/falcetta/0_PhD/JOB-VS/JoB-VS/datasets/TOPCOW'
json_out_filepath = os.path.join(data_root, 'dataset.json')

out_directory = os.path.join(data_root, 'processed')
Path(out_directory).mkdir(parents=True, exist_ok=True)

#num_workers = 50 # Default value: -1 (use all available cores)

# Change to false if want to avoid the processing on data that has been 
# already processed.
process_again=True

# Calculate the statistics of the original dataset
#Global_features(data_root, num_workers)
Global_features_general(json_out_filepath)

# # Preprocess the datasets
Preprocess_datasets(out_directory, data_root, remake=process_again)

# # Calculate the dataset sizes (used to plan the experiments)
Calculate_sizes(out_directory, remake=process_again)

