#!/bin/bash

export SPARK_MASTER_HOST=`hostname -I`
./spark-3.3.1-bin-hadoop3/bin/spark-class org.apache.spark.deploy.master.Master --ip $SPARK_MASTER_HOST --port $SPARK_MASTER_PORT --webui-port $SPARK_MASTER_WEBUI_PORT