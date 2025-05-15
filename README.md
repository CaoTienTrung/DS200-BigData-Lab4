```
python src\download_dataset.py\
    --dataset_name mohamedhanyyy/chest-ctscan-images\
    --save_dir "Dataset"
```

```
python src\Producer\stream.py\
    --directory "Dataset\Data"\
    --split "train"\
    --batch_size 8\
    --endless True\
    --label_mode "int"\
    --color_mode "grayscale"\
    --image_size 256 256\
    --interpolation "bilinear"\
    --sleep 5

python src\Producer\stream.py --directory "Dataset\Data" --split "train" --batch_size 8 --endless False --label_mode "int" --color_mode "rgb" --image_size 32 32 --interpolation "bilinear" --sleep 5
```

python src\Consumer\main.py --mode "train" --save_path "Model\random_forest.pkl"