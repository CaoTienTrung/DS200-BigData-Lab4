import kagglehub
import shutil
import os
import argparse

def download_dataset(dataset_name, save_dir):
    path = kagglehub.dataset_download(dataset_name)
    shutil.copytree(path, save_dir, dirs_exist_ok=True)
    print("Dataset saved to:", save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a Kaggle dataset using kagglehub.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Kaggle dataset name (e.g. username/dataset-name)")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the dataset")

    args = parser.parse_args()
    download_dataset(dataset_name=args.dataset_name, save_dir=args.save_dir)
