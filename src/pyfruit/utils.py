import numpy as np
import pandas as pd
import os, sys
import boto3, botocore.session
from io import BytesIO
import cv2
from PIL import Image


from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql.functions import udf, pandas_udf, PandasUDFType
from pyspark.sql.types import StringType

from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import MobileNet_V2_Weights
import torchvision
import torch

def array_to_string(my_list):
    return '[' + ','.join([str(elem) for elem in my_list]) + ']'
array_to_string_udf = udf(array_to_string, StringType())

def model_fn():
    """
    Returns a MobileNetV2 model with top layer removed 
    and broadcasted pretrained weights.
    """
    model = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model = create_feature_extractor(model, return_nodes=['flatten'])
    model.eval()
    return model

def featurize_series(model, content_series):
    """
    Featurize a pd.Series of raw images using the input model.
    :return: a pd.Series of image features
    """
    # NMAX : max number of extracted Features
    NMAX = 509
    input_data = []
    session = None

    # Load a s3 aws session
    if 's3://' in content_series[0] :
        session = botocore.session.get_session()
        s3_client = boto3.client(
            's3',
            aws_access_key_id = session.get_credentials().access_key,
            aws_secret_access_key = session.get_credentials().secret_key
        )
        s3 = boto3.resource('s3')
    
    
    for content in content_series :
        if 's3://' in content :
            bucket_name = content.split('/')[2]
            filename = content.split(bucket_name + '/')[1]
            #return filename
            s3 = boto3.resource('s3')
            bucket = s3.Bucket(bucket_name)
            obj = bucket.Object(filename)
            file_stream = obj.get()['Body']
            img = Image.open(file_stream).resize([224, 224])
        else :
            img = Image.open(content).resize([224, 224])
        arr = np.array(img)
        input_data.append(arr)
        del arr, img
        
    input_data = np.transpose(input_data, (0,3,1,2))
    with torch.no_grad():
        preds = model(torch.Tensor(input_data))['flatten']
    del input_data
    
    output = [np.array(p)[:NMAX] for p in preds]
    del preds
    return pd.Series(output)

@pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
def featurize_udf(content_series_iter):
    '''
    This method is a Scalar Iterator pandas UDF wrapping our featurization function.
    The decorator specifies that this returns a Spark DataFrame column of type ArrayType(FloatType).

    :param content_series_iter: This argument is an iterator over batches of data, where each batch
                              is a pandas Series of image data.
    '''
    # With Scalar Iterator pandas UDFs, we can load the model once and then re-use it
    # for multiple data batches.  This amortizes the overhead of loading big models.
    model = model_fn()
    for content_series in content_series_iter:
        yield featurize_series(model, content_series)
        
# Vectorize the extracted fearures array to be used in PCA
vectorize_udf = udf(lambda p : Vectors.dense(p), VectorUDT())

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
        bucket_name = filepath.split('/')[2]
        filename = filepath.split(bucket_name + '/')[1]
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