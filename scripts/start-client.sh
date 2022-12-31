#!/bin/bash

echo "SPARK_LOCAL_IP="$SPARK_LOCAL_IP
echo "PYFRUIT_MASTER="$PYFRUIT_MASTER
echo "PYFRUIT_BLOCKMANAGER_PORT="$PYFRUIT_BLOCKMANAGER_PORT
echo "PYFRUIT_DRIVER_PORT="$PYFRUIT_DRIVER_PORT
echo "PYFRUIT_DRIVER_HOST="$PYFRUIT_DRIVER_HOST
echo "PYFRUIT_LOCAL_IP="$PYFRUIT_LOCAL_IP
echo "PYFRUIT_INPUT_PATH="$PYFRUIT_INPUT_PATH
echo "PYFRUIT_OUTPUT_PATH="$PYFRUIT_OUTPUT_PATH
echo "PYFRUIT_NC="$PYFRUIT_NC
echo "PYFRUIT_RESIZE="$PYFRUIT_RESIZE

./spark-3.3.1-bin-hadoop3/bin/spark-submit --master $PYFRUIT_MASTER \
    --conf spark.blockManager.port=$PYFRUIT_BLOCKMANAGER_PORT \
    --conf spark.driver.port=$PYFRUIT_DRIVER_PORT \
    --conf spark.driver.host=$PYFRUIT_DRIVER_HOST \
    --conf spark.driver.bindAddress=$PYFRUIT_LOCAL_IP \
    --conf spark.jars.packages=org.apache.hadoop:hadoop-aws:3.2.2 \
    --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem \
    --conf spark.hadoop.fs.s3a.aws.credentials.provider=com.amazonaws.auth.DefaultAWSCredentialsProviderChain \
    main.py \
        -i $PYFRUIT_INPUT_PATH \
        -o $PYFRUIT_OUTPUT_PATH \
        -resize_shape $PYFRUIT_RESIZE \
        -nc $PYFRUIT_NC