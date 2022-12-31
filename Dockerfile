FROM docker.io/library/python:3.9

WORKDIR /app

# Install the required libraries for spark
RUN apt update
RUN apt install libsm6 libxext6  -y
RUN apt install default-jdk -y
RUN apt install libatlas3-base libopenblas-base -y
RUN apt remove curl -y
RUN apt install curl -y

# Download and extract spark
RUN curl -SL https://dlcdn.apache.org/spark/spark-3.3.1/spark-3.3.1-bin-hadoop3.tgz | tar -xz --no-same-owner

# Copy spark config
COPY ./spark_conf/log4j2.properties spark-3.3.1-bin-hadoop3/conf/.

# Install pyfruit package
COPY ./dist/pyfruit-0.1-py3-none-any.whl .
RUN pip install pyfruit-0.1-py3-none-any.whl
RUN pip install jupyter 

# Copy running scripts
COPY ./scripts/* .
COPY ./spark-env.sh.template /app/spark-3.3.1-bin-hadoop3/conf/spark-env.sh
ENV SPARK_LOCAL_IP=0.0.0.0
ENV SPARK_MASTER_PORT=7077
ENV SPARK_MASTER_WEBUI_PORT=9090
CMD ["/bin/bash"]
