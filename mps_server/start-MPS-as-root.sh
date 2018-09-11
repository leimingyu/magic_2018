#!/bin/bash 
export CUDA_VISIBLE_DEVICES="6"
nvidia-smi -i 6 -c DEFAULT 
nvidia-cuda-mps-control -d
