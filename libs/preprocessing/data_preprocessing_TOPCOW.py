import argparse
import os
from pathlib import Path

from global_features import Global_features_general
from task_preprocessing import Preprocess_datasets_general
from image_sizes import Calculate_sizes_general

def main(args):

    data_root = args.data_root
    assert os.path.exists(data_root), f"{data_root} folder not found"
    json_out_filepath = os.path.join(data_root, 'dataset.json')

    out_directory = os.path.join(data_root, 'processed')
    Path(out_directory).mkdir(parents=True, exist_ok=True)
    
    #num_workers = 50 # Default value: -1 (use all available cores)

    # Change to false if want to avoid the processing on data that has been 
    # already processed.
    process_again=True

    # Calculate the statistics of the original dataset
    #Global_features(data_root, num_workers)
    print(f'Calculating global features for {json_out_filepath}')
    Global_features_general(json_out_filepath)

    # # Preprocess the datasets
    Preprocess_datasets_general(out_directory, data_root, remake=process_again)

    # # Calculate the dataset sizes (used to plan the experiments)
    print(f'Calculating sizes for {out_directory}')
    Calculate_sizes_general(out_directory, remake=process_again)



if __name__ == '__main__':
    # SET THE PARAMETERS
    parser = argparse.ArgumentParser()
    # EXPERIMENT DETAILS
    parser.add_argument('--data_root', type=str, default=os.path.join(os.getcwd(),'datasets/TOPCOW'))
    
    main(parser.parse_args())
