#!/bin/bash
#nvidia-smi --query-gpu=timestamp,utilization.gpu --format=csv   -l 1 


#nvidia-smi --query-gpu=timestamp,utilization.gpu --format=csv   -lms 500   -f monitor_log_0.5s
nvidia-smi --query-gpu=timestamp,utilization.gpu --format=csv   -l 1   -f monitor_log_1s.csv
#nvidia-smi --query-gpu=timestamp,utilization.gpu --format=csv   -lms 100   -f monitor_log_0.1s


