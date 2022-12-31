#!/bin/bash
sudo apt update
sudo apt install podman -y
sudo apt install python3-pip -y
pip install podman-compose
export PATH=$PATH:/home/ubuntu/.local/bin