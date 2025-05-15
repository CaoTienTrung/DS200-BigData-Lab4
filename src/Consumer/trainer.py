import os
import pyspark
from pyspark.context import SparkContext
from pyspark.streaming.context import StreamingContext
from pyspark.sql.context import SQLContext
from pyspark.sql.types import IntegerType, StructField, StructType
from pyspark.ml.linalg import VectorUDT
from dataloader import StreamingDataLoader

from models.model import *
from models.feature_extractor import *
from config import SparkConfig

class Trainer:
    def __init__(self, model, spark_config: SparkConfig, save_path: str):
        self.model         = model
        self.sparkConf     = spark_config
        self.save_path    = save_path 

        # Spark contexts
        self.sc   = SparkContext(f"{self.sparkConf.host}[{self.sparkConf.receivers}]", self.sparkConf.appName)
        self.ssc  = StreamingContext(self.sc, self.sparkConf.batch_interval)
        self.sqlC = SQLContext(self.sc)

        self.dl = StreamingDataLoader(self.sc, self.ssc, self.sqlC, self.sparkConf)

        self.total_batches   = 0

    def train(self):
        stream = self.dl.parse_stream()
        stream.foreachRDD(self.__train__)
        self.ssc.start()
        self.ssc.awaitTermination()

    def __train__(self, timestamp, rdd: pyspark.RDD):
        if not rdd.isEmpty():
            schema = StructType([
                StructField("image", VectorUDT(), True),
                StructField("label", IntegerType(), True)
            ])
            df = self.sqlC.createDataFrame(rdd, schema)
            # print(df)

            preds, acc, prec, rec, f1 = self.model.train(df)

            print("="*10)
            print(f"[TRAIN] acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}")
            print("="*10)

            # Save model after a mini-batch
            self.model.save(self.save_path)
            print(f"Model saved to {self.save_path}")

        print("[TRAIN] Batch size received:", rdd.count())

    def predict(self):
        print(f"Loading model from {self.save_path} ...")
        self.model.load(self.save_path)

        self.empty_batches = 0
        self.y_preds = []
        self.y_trues = []
        stream = self.dl.parse_stream()
        stream.foreachRDD(self.__predict__)
        self.ssc.start()
        self.ssc.awaitTermination()

    def __predict__(self, timestamp, rdd: pyspark.RDD):
        if not rdd.isEmpty():
            schema = StructType([
                StructField("image", VectorUDT(), True),
                StructField("label", IntegerType(), True)
            ])
            df = self.sqlC.createDataFrame(rdd, schema)

            trues, preds, acc, prec, rec, f1 = self.model.predict(df)
            self.y_preds.extend(preds.tolist())
            self.y_trues.extend(trues.tolist())
            self.total_batches += 1
            

            print("="*20)
            print(f"[PREDICT] batch#{self.total_batches}")
            print(f" acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}")

            print("="*20)
        else:
            self.empty_batches += 1
            if self.empty_batches >= 3:
                os.system('cls')
                print("="*20)
                print(f"[FINAL REPORT] Total batches: {self.total_batches}")
                print(f" Final Accuracy : {accuracy_score(self.y_trues, self.y_preds):.4f}")
                print(f" Final Precision: {precision_score(self.y_trues, self.y_preds, average='macro'):.4f}")
                print(f" Final Recall   : {recall_score(self.y_trues, self.y_preds, average='macro'):.4f}")
                print(f" Final F1-score : {f1_score(self.y_trues, self.y_preds, average='macro'):.4f}")
                print("="*20)
                self.ssc.stop(stopSparkContext=True, stopGraceFully=True)



