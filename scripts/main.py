from pyspark.sql import SQLContext, SparkSession
from pyspark import SparkFiles
import pyfruit
from pyfruit.process import process
from pyfruit.utils import array_to_string_udf
import os, sys
import numpy as np

if __name__ == '__main__' :
    config = {
        '-i' : '',
        '-master_url' : "local[*]",
        '-resize_shape' : [20,20],
        '-nc' : 2,    
        '-v' : 1,
        '-o' : 'apple_dimred_3.csv'
    }
    for i in range(1,len(sys.argv), 2) :
        name = sys.argv[i]
        if name not in config.keys() :
            raise Exception('%s not available in pyspark script parameters' %(name))
        if name in ['-resize_shape'] :
            config[name] = np.int0(sys.argv[i+1].split(','))
        elif name in ['-nc', '-v'] :
            config[name] = int(sys.argv[i+1])
        else :
            config[name] = sys.argv[i+1]

    if config['-v'] == 1 :
        print('Used parameters:')
        for k in config.keys() :
            print(' %s : %s' %(k, str(config[k])))


    #spark = SparkSession.builder.master(config['-master_url']).config("spark.driver.bindAddress", "127.0.0.1").getOrCreate()
    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    #site_package = sys.path[-1]
    #sc.addFile(os.path.join(site_package,'pyfruit', 'process.py'))
    #sc.addFile(os.path.join(site_package,'pyfruit', 'utils.py'))
    #sys.path.insert(0,SparkFiles.getRootDirectory())

    df = process(spark,config['-i'],config['-nc'], config['-resize_shape'], config['-v'])

    if config['-v'] == 1 :
        print('Writing in %s...' %(config['-o']), end=' ')
    #df.withColumn('image', array_to_string_udf(df["image"])).withColumn('std', array_to_string_udf(df["std"])).withColumn('pca', array_to_string_udf(df["pca"])).write.mode('overwrite').csv(os.path.abspath(config['-o']))
    df.withColumn('image', array_to_string_udf(df["image"])).withColumn('std', array_to_string_udf(df["std"])).withColumn('pca', array_to_string_udf(df["pca"])).coalesce(1).write.mode('overwrite').csv(os.path.abspath(config['-o']))
    if config['-v'] == 1 :
        print('Done !')