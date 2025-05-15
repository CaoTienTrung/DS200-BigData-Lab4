import numpy as np
from pyspark.context import SparkContext
from pyspark.sql.context import SQLContext
from pyspark.streaming.context import StreamingContext
from pyspark.streaming.dstream import DStream
from pyspark.ml.linalg import DenseVector

from config import SparkConfig

import json
class StreamingDataLoader:
    def __init__(self, 
                 sparkContext: SparkContext, 
                 sparkStreamingContext: StreamingContext, 
                 sqlContext: SQLContext,
                 sparkConf: SparkConfig) -> None:
        
        self.sc = sparkContext
        self.ssc = sparkStreamingContext
        self.sql_context = sqlContext
        self.sparkConf = sparkConf

        self.stream = self.ssc.socketTextStream(
            hostname=self.sparkConf.stream_host,
            port=self.sparkConf.port
        )

    def parse_stream(self) -> DStream:
        record_arr = self.stream.map(lambda line: json.loads(line))\
                  .flatMap(lambda d: d.values())\
                  .map(lambda record: list(record.values()))\
                  .map(lambda x: [
                      np.array(x[:-1])\
                      .reshape(3, 32, 32)\
                      .transpose(1,2, 0)\
                      .astype(np.uint8),
                      int(x[-1])
                  ])
        return self.preprocess(record_arr)
        
    @staticmethod
    def preprocess(stream: DStream) -> DStream:
        stream = stream.map(lambda x: [x[0].reshape(-1).tolist(),x[1]])
        stream = stream.map(lambda x: [DenseVector(x[0]), x[1]])
        
        return stream
        




