# PyFruit

PyFruit is a Python library based on spark to handle the first preprocessing steps of fruits images treatments in an aws environment.
The fruit image database is available on [https://www.kaggle.com/datasets/moltean/fruits](https://www.kaggle.com/datasets/moltean/fruits).

This repository permits also to create a docker container image handling the creation of a master, workers and client spark instance to handle the preprocessing app in a aws eenvironment.

## Installation

### pyfruit

Download dist/pyfruit-0.1-py3-none-any.whl and use the package manager [pip](https://pip.pypa.io/en/stable/) in the same directory.

```bash
pip install ./dist/pyfruit-0.1-py3-none-any.whl
```

### Create docker image

Download the repository and run a podman build in the same directory (must work with docker). 

```bash
podman build -t pyfruit_spark .
```

## Usage


### Docker container

There is 3 exemples of docker-compose.yml to run a container for a spark-master (docker-compose_master.yml), spark-worker (docker-compose_worker.yml) and jupyter client (docker-compose_jupyter.yml) and docker-compose.yml to create 3 containers : master / worker / jupyter client.

Modify the docker-compose.yml according to your needs : 
- Set SPARK_MASTER to your master location ip address.
- Set PYFRUIT_DRIVER_HOST to your client location ip address.
- Set dir_to_aws_cred to you aws crediential location (for exemple /home/user/.aws).

Then run podman-compose :
```bash
podman-compose -f docker-compose.yml up -d
```
