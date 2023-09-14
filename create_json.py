import json, os
from pathlib import Path
from libs.preprocessing.global_features import Global_features_general




def create_json_file(description, labels, modality, name, root, train_img, train_label, val_img, val_label, test_img, test_label, output_file):
    # Define the data structure
    data = {
        "description": description,
        "labels": labels,
        "modality": modality,
        "name": name,
        "root": root,
        "training": [],
        "validation": [],
        "test": []
    }

    # Populate the training data
    populate_json(train_img, train_label, data, "training")

    # Populate the validation data
    populate_json(val_img, val_label, data, "validation")

    # Populate the test data
    populate_json(test_img, test_label, data, "test")
    
    # Create a JSON file and write the data to it
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"JSON file '{output_file}' created successfully!")

def populate_json(img_list, label_list, data, data_type):
    for i in range(len(img_list)):
        entry = {
            "image": img_list[i],
            "label": label_list[i],
            "monai_name": img_list[i].replace("/", "_").replace("_ToF", "")
        }
        data[data_type].append(entry)


def get_id(filepath):
    filename = os.path.basename(filepath)
    id = filename.split("_")[0].split(".")[0] # remove extension and split by "_"
    return id

def test_get_id():
    assert get_id("~/Job-VS-TOPCOW/ORIGINAL/train/001.jpg") == "001", "get_id function not working"
    assert get_id("001_ToF.png") == "001", "get_id function not working"
    assert get_id("001_vessel_aaa.jpg") == "001", "get_id function not working"
    print("get_id function working")
    

def extract_paired_list(dataset_complete_path, dataset_name=''):
    
    assert os.path.exists(dataset_complete_path), f"{dataset_name} folder not found"
    assert dataset_name in ['train', 'test'], "dataset_name must be 'train' or 'test'"
    
    data_list = os.listdir(dataset_complete_path)
    data_list = [os.path.join(dataset_name, x) for x in data_list]
    data_img_list = sorted([x for x in data_list if 'ToF' in x])
    data_label_list = sorted([x for x in data_list if 'vessel' in x])

    assert len(data_img_list) == len(data_label_list), "Images and labels have different length"
    print(f"Dataset ({dataset_name}) size: {len(data_label_list)}") 


    for img, label in zip(data_img_list, data_label_list):
        assert get_id(img) == get_id(label), "Images and labels are not aligned"

    return data_img_list, data_label_list


#main
def main(args):
    
    test_get_id()
    
    dataset_path = os.path.normpath(args.dataset_path)
    print(f"Dataset path: {dataset_path}")
    json_dir = os.path.normpath(args.json_dir)
    print(f"JSON directory: {json_dir}")
    
    json_out_filepath = os.path.join(json_dir, 'dataset.json')
    Path(json_dir).mkdir(parents=True, exist_ok=True)

    train_folder = os.path.join(dataset_path, "train")
    test_folder = os.path.join(dataset_path, "test")

    assert os.path.exists(train_folder), "Train folder not found"
    assert os.path.exists(test_folder), "Test folder not found"

    train_img_list, train_label_list = extract_paired_list(train_folder,'train')
    test_img_list, test_label_list = extract_paired_list(test_folder, 'test')
        
    val_img_list = []
    val_label_list = []
    
    # Take n RANDOM samples from the training set (n images + n labels)
    import random
    random.seed(42)

    n = len(test_img_list)

    if len(val_img_list) == 0:
        for i in range(n):
            idx = random.randint(0, len(train_img_list)-1)
            val_img_list.append(train_img_list.pop(idx))
            val_label_list.append(train_label_list.pop(idx))
    else:
        print("Validation set already created\n")

    print(f"Validation set size: {len(val_img_list)}\n")

    print(f"Val images: {val_img_list}")
    print(f"Val labels: {val_label_list}")

    for img, label in zip(val_img_list, val_label_list):
        assert get_id(img) == get_id(label), "Images and labels are not aligned"
        
    num_train, num_val, num_test = len(train_img_list), len(val_img_list), len(test_img_list)
    print(f"Train set size: {num_train} ({num_train/(num_train+num_val+num_test)*100:.2f}%)")
    print(f"Validation set size: {num_val} ({num_val/(num_train+num_val+num_test)*100:.2f}%)")
    print(f"Test set size: {num_test} ({num_test/(num_train+num_val+num_test)*100:.2f}%)")

    description = "TOPCOW Dataset - 3D TOF MRA images of the Circle of Willis"
    labels = {
        "0": "background",
        "1": "vessels"
    }
    modality = {
        "0": "tof"
    }
    name = "TOPCOW Dataset"
    root = dataset_path


    create_json_file(description, labels,
                     modality, name, root,
                     train_img_list, train_label_list,
                     val_img_list, val_label_list,
                     test_img_list, test_label_list,
                     json_out_filepath)
    
    Global_features_general(json_out_filepath)



#main
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/data/falcetta/Job-VS-TOPCOW/ORIGINAL/')
    parser.add_argument('--json_dir', type=str, default='/home/falcetta/0_PhD/JOB-VS/JoB-VS/datasets/TOPCOW')
    args = parser.parse_args()
    main(args)