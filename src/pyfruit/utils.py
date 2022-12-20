import numpy as np
import os, sys
import boto3, botocore.session
from io import BytesIO
import cv2


from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

def array_to_string(my_list):
    return '[' + ','.join([str(elem) for elem in my_list]) + ']'
array_to_string_udf = udf(array_to_string, StringType())

def load_filepath(spark, datapath):
    from pyspark.sql.types import Row
    if 's3://' not in datapath : 
        listdir = os.listdir(datapath)
        filepath_list = []
        for dir in listdir :
            for file in os.listdir(os.path.join(datapath, dir)) :
                filepath_list.append(Row(os.path.abspath(os.path.join(datapath, dir, file))))
        df = spark.createDataFrame(filepath_list).toDF('filepath')
        return df
    else :
        session = botocore.session.get_session()
        s3_client = boto3.client(
            's3',
            aws_access_key_id = session.get_credentials().access_key,
            aws_secret_access_key = session.get_credentials().secret_key
        )
        s3 = boto3.resource('s3')
        bucket_name = datapath.split('/')[-1]
        bucket = s3.Bucket(bucket_name)
        filepath_list = []
        for my_bucket_object in bucket.objects.filter(Prefix='data') :
            if 'jpg' in my_bucket_object.key :
                filename = str(my_bucket_object.key)
                filepath_list.append(Row(datapath + '/' + filename))
        df = spark.createDataFrame(filepath_list).toDF('filepath')
        return df

def load_image_preprocess(filepath, resize_shape=[20,20]) :
    if 's3://' not in filepath :
        img = cv2.imread(filepath)
    else :
        session = botocore.session.get_session()
        s3_client = boto3.client(
            's3',
            aws_access_key_id = session.get_credentials().access_key,
            aws_secret_access_key = session.get_credentials().secret_key
        )
        #s3 = boto3.resource('s3')
        #return '0'
        bucket_name = filepath.split('/')[2]
        filename = filepath.split(bucket_name + '/')[1]
        #return filename
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        obj = bucket.Object(filename)
        
        file_stream = BytesIO()
        obj.download_fileobj(file_stream)
        np_1d_array = np.frombuffer(file_stream.getbuffer(), dtype="uint8")
        img = cv2.imdecode(np_1d_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, resize_shape)
    return Vectors.dense(img.flatten())


get_dirname_udf = udf(lambda x : x.split('/')[-2], StringType())
get_name_udf = udf(lambda x : x.split('/')[-1], StringType())


img2vec = udf(lambda img: Vectors.dense(np.array(img).flatten()), VectorUDT())