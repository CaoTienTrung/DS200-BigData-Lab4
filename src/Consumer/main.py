import argparse
# import findspark
# findspark.init()
from pyspark import SparkContext
from dataloader import StreamingDataLoader
from models.model import *
from models.feature_extractor import *
from trainer import Trainer
from config import SparkConfig

import pyspark
from pyspark.context import SparkContext
from pyspark.streaming.context import StreamingContext
from pyspark.sql.context import SQLContext

parser = argparse.ArgumentParser(
    description="Streaming Trainer"
)
parser.add_argument("--mode", type=str, choices=["train", "predict"], required=True, 
                    help="Mode to run: train or predict")
parser.add_argument("--model", type=str, choices=["rf", "svm"], required=True, 
                    help="Choosing model: SVM for RF")
parser.add_argument("--save_path", type=str, required=True, 
                    help="Path to save/load model")
args = parser.parse_args()


if __name__ == "__main__":
    spark_config = SparkConfig()
    model = None
    if (args.model == 'rf'):
        model = RandomForest()
    else: 
        model = SVM()
    trainer = Trainer(model, spark_config, args.save_path)
    # stream = trainer.dl.parse_stream()
    # stream.pprint()
    # stream.foreachRDD(trainer.__train__)
    # trainer.ssc.start()
    # trainer.ssc.awaitTermination()
    if (args.mode == 'train'):
        trainer.train()
    else:
        trainer.predict()
