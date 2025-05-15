

# Medical Chest Image Classification with Spark Streaming

This project demonstrates a real-time medical image classification pipeline for chest X-ray images using Apache Spark Streaming. The architecture consists of two main components:

* Producer: Reads images from a local dataset, preprocesses them into mini-batches, and streams JSON payloads over a TCP socket.

* Consumer: Uses Spark Streaming to listen on the socket, parse incoming batches into DataFrames, extract features, and train a classification model on each mini-batch.
    * `Feature extractor` methods include `HOG`, `SIFT`, `HTFs`, `LBP`, `Gabor Filter`.
    * `Classification` model include `Random Forest`, `Support Vector Machine`.

## 🧾Dataset
[Dataset link][https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images]

## 🏗 Architecture
```
+-------------+          TCP          +--------------+
|             |  Producer (Python)   |              |
|  Image      |  socket.sendall()    | Spark        |
|  Dataset    | ------------------>  | Streaming    |
|  (Structured|                      | Consumer     |
|   Folders)  |                      +--------------+
|             |                             |
+-------------+                             |
       ^                                    |
       |                                    v
       |                              +--------------+
       |                              | Model        |
       |                              | Training per |
       |                              | Mini-batch   |
       |                              +--------------+
       v
+-------------+
|  Custom     |
|  DataLoader |
+-------------+
```
## 📦 Prerequisites

* Python 3.9+
* Java 8+
* Apache Spark 3.5.5
* Conda

**Python Dependencies**
```
conda create -n medical_env python=3.9
conda activate medical_env
!pip install -r requirements.txt
```
## 📁 Project Structure
```
Lab5/
├── src/
|   ├── download_dataset.py      # downloading with kagglehub
│   ├── Producer/
│   │   ├── stream.py            # Producer: stream images via TCP
│   │   ├── custom_dataset.py    # Dataset loader
│   └── Consumer/
│       ├── main.py              # Entrypoint for train/predict
│       ├── trainer.py           # training model setup
│       ├── dataloader.py        # parse_stream and data loader
│       ├── models/              # Model & feature-extractor 
|       └── config.py            # SparkConfig parameters
│   
├── Dataset/Data/                # Chest X-ray images (train, dev, test splits)
└── Model/                       # Saved models
```
## 🚀 Usage
```
!git clone https://github.com/CaoTienTrung/DS200-BigData-Lab4
```

### 1. Training mode
#### 1. Open a terminal and run Producer
```
# Activate environment
conda activate medical_env

# Run producer
python src\Producer\stream.py\
    --directory "Dataset\Data"\
    --split "train"\
    --batch_size 8\
    --label_mode "int"\
    --color_mode "rgb"\
    --image_size 32 32\
    --interpolation "bilinear"\
    --sleep 5           

```
*This will start a TCP server at localhost:6100 and send mini-batches as JSON payloads.*

#### 2. Open another terminal and run Consumer
```
# Activate environment
conda activate medical_env

# Run producer
python src\Consumer\main.py\
    --mode "train"\
    --model "rf"\
    --save_path "Model\random_forest.pkl"        

```
*This will connect to localhost:6100, parse each batch as a Spark DataFrame, extract features, train the model and save to a pickle file.*

### 2. Testing mode
#### 1. Open a terminal and run Producer
```
# Activate environment
conda activate medical_env

# Run producer
python src\Producer\stream.py\
    --directory "Dataset\Data"\
    --split "test"\
    --batch_size 8\
    --label_mode "int"\
    --color_mode "rgb"\
    --image_size 32 32\
    --interpolation "bilinear"\
    --sleep 5           

```
*This will start a TCP server at localhost:6100 and send mini-batches as JSON payloads.*

#### 2. Open another terminal and run Consumer
```
# Activate environment
conda activate medical_env

# Run producer
python src\Consumer\main.py\
    --mode "predict"\
    --model "rf"\
    --save_path "Model\random_forest.pkl"        

```
*This will connect to localhost:6100, parse each batch as a Spark DataFrame, extract features, load the saved model and start predicting and evaluate.*

## ⚙️ Configuration
* Edit src/config.py to adjust Spark settings:
```
class SparkConfig:
    appName        = "MedialImageClassificationApp"
    receivers      = 2                                    # Number of streaming receivers
    host           = "local"                              # Spark master
    stream_host    = "localhost"
    port           = 6100                                 # TCP port
    batch_interval = 5                                    # Seconds per batch
```
## 🛠️ Customization
* Model: adjust between `RandomForest` and `SupportVectorMachine`.
* Feature Extraction: adjusting between `HOG`, `SIFT`, `HTFs`, `LBP`, `Gabor Filter`.

## 📈 Evaluation
* Inspect console logs for per-batch accuracy, precision, recall, and F1-score.
* After finishing prediction mode, review final aggregate metrics.

