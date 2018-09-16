#!/usr/bin/env python
"""Scripts to preprocess the cuda applications."""
#import os
import subprocess
import pandas as pd
import sys

METRICS_COLS = ['Device', 'Kernel', 'Invocations', 'Metric Name', 'Metric Description',
                'Min', 'Max', 'Avg']


Percentage2decimal_Metrics = ['sm_efficiency', 'branch_efficiency', 
        'warp_execution_efficiency', 'warp_nonpred_execution_efficiency', 
        'issue_slot_utilization', 'global_hit_rate', 'local_hit_rate',
        'gld_efficiency', 'gst_efficiency', 'shared_efficiency', 
        'stall_inst_fetch', 'stall_exec_dependency', 'stall_memory_dependency',
        'stall_texture', 'stall_sync', 'stall_other',
        'stall_constant_memory_dependency', 'stall_pipe_busy',
        'stall_memory_throttle', 'stall_not_selected', 'local_memory_overhead', 
        'tex_cache_hit_rate','l2_tex_read_hit_rate', 'l2_tex_write_hit_rate', 
        'flop_sp_efficiency', 'flop_dp_efficiency']


Throughput_Metrics = ['gld_requested_throughput', 'gst_requested_throughput', 
        'gld_throughput', 'gst_throughput', 'dram_read_throughput', 
        'dram_write_throughput', 'tex_cache_throughput', 'local_load_throughput', 
        'local_store_throughput', 'shared_load_throughput', 
        'shared_store_throughput', 'l2_tex_read_throughput', 
        'l2_tex_write_throughput', 'l2_read_throughput', 'l2_write_throughput', 
        'sysmem_read_throughput', 'sysmem_write_throughput', 
        'l2_atomic_throughput', 'ecc_throughput']

Utilization2decimal_Metrics = ['cf_fu_utilization', 'tex_fu_utilization', 
        'ldst_fu_utilization', 'double_precision_fu_utilization', 
        'special_fu_utilization', 'single_precision_fu_utilization', 
        'dram_utilization', 'tex_utilization', 'shared_utilization', 
        'l2_utilization', 'sysmem_utilization', 'sysmem_read_utilization', 
        'sysmem_write_utilization']

Cols2norm_Full = ['ipc',
        'issued_ipc',
        'inst_per_warp',
        'inst_replay_overhead',
        'shared_load_transactions_per_request',
        'shared_store_transactions_per_request',
        'local_load_transactions_per_request',
        'local_store_transactions_per_request',
        'gld_transactions_per_request',
        'gst_transactions_per_request',
        'shared_store_transactions',
        'shared_load_transactions',
        'local_load_transactions',
        'local_store_transactions',
        'gld_transactions',
        'gst_transactions',
        'dram_read_transactions',
        'dram_write_transactions',
        'gld_requested_throughput',
        'gst_requested_throughput',
        'gld_throughput',
        'gst_throughput',
        'dram_read_throughput',
        'dram_write_throughput',
        'tex_cache_throughput',
        'local_load_throughput',
        'local_store_throughput',
        'shared_load_throughput',
        'shared_store_throughput',
        'tex_cache_transactions',
        'flop_count_dp',
        'flop_count_dp_add',
        'flop_count_dp_fma',
        'flop_count_dp_mul',
        'flop_count_sp',
        'flop_count_sp_add',
        'flop_count_sp_fma',
        'flop_count_sp_mul',
        'flop_count_sp_special',
        'inst_executed',
        'inst_issued',
        'inst_fp_32',
        'inst_fp_64',
        'inst_integer',
        'inst_bit_convert',
        'inst_control',
        'inst_compute_ld_st',
        'inst_misc',
        'inst_inter_thread_communication',
        'issue_slots',
        'cf_issued',
        'cf_executed',
        'ldst_issued',
        'ldst_executed',
        'atomic_transactions',
        'atomic_transactions_per_request',
        'sysmem_read_transactions',
        'sysmem_write_transactions',
        'l2_read_transactions',
        'l2_write_transactions',
        'ecc_transactions',
        'l2_tex_read_throughput',
        'l2_tex_write_throughput',
        'l2_tex_read_transactions',
        'l2_tex_write_transactions',
        'l2_read_throughput',
        'l2_write_throughput',
        'sysmem_read_throughput',
        'sysmem_write_throughput',
        'l2_atomic_throughput',
        'l2_atomic_transactions',
        'ecc_throughput',
        'eligible_warps_per_cycle']


ColsMaxOne_Full = ['sm_efficiency',
        'achieved_occupancy',
        'branch_efficiency',
        'warp_execution_efficiency',
        'warp_nonpred_execution_efficiency',
        'issue_slot_utilization',
        'global_hit_rate',
        'local_hit_rate',
        'gld_efficiency',
        'gst_efficiency',
        'cf_fu_utilization',
        'tex_fu_utilization',
        'ldst_fu_utilization',
        'double_precision_fu_utilization',
        'special_fu_utilization',
        'single_precision_fu_utilization',
        'dram_utilization',
        'tex_utilization',
        'shared_efficiency',
        'shared_utilization',
        'stall_inst_fetch',
        'stall_exec_dependency',
        'stall_memory_dependency',
        'stall_texture',
        'stall_sync',
        'stall_other',
        'stall_constant_memory_dependency',
        'stall_pipe_busy',
        'stall_memory_throttle',
        'stall_not_selected',
        'local_memory_overhead',
        'tex_cache_hit_rate',
        'l2_tex_read_hit_rate',
        'l2_tex_write_hit_rate',
        'l2_utilization',
        'sysmem_utilization',
        'sysmem_read_utilization',
        'sysmem_write_utilization',
        'flop_sp_efficiency',
        'flop_dp_efficiency']


def cuprof(appcmd, target_metrics=None, ofile=""):
    """Profile input cuda application."""
    # if target_metrics are specified, do partial profiling
    # else do full metrics profiling
    metrics_opt = "--metrics "
    if not target_metrics:
        metrics_opt += "all"
    else:
        for index, metric in enumerate(target_metrics):
            if index == len(target_metrics) - 1:
                metrics_opt += str(metric)
            else:
                metrics_opt += str(metric) + ","

    if not ofile:
        raise Exception('Output file is not specified!')

    log_opt = "--csv --log-file " + str(ofile)

    # if len(str(logFileName)):
    ##    raise Exception('Output metrics file is not specified!')

    run_nvprof = "nvprof" + " " + metrics_opt + " " + log_opt + " " + appcmd
    # print runNvprof

    pipe = subprocess.Popen(run_nvprof, shell=True)


def read_trace(logfile_csv):
    """Read profiling metrics in csv to dataframe in pandas."""
    df_trace = pd.read_csv(logfile_csv, names=METRICS_COLS, engine='python')

    # find out the number of rows to skip
    rows2skip = 0
    for index, row in df_trace.iterrows():
        if "Metric result" in str(row['Device']):
            rows2skip = index + 1  # index starting from 0
            break
    # print rows2skip

    df_trace = pd.read_csv(logfile_csv, names=METRICS_COLS, skiprows=rows2skip + 1)

    return df_trace


#------------------------------------------------------------------------------
# convert metric from original format to the target presentation 
#------------------------------------------------------------------------------
def adjust_metric(metricName, metricValue):
    """
    Adjust metric from dataframe containing nvprof results.
    """ 
    metricValue_str= str(metricValue)
    #adjustedV = float(metricValue)  # apply float as default

    adjustedV = None

    #
    # update the value if the metric belongs to the following 3 groups
    # 

    # convert % to decimal
    if metricName in Percentage2decimal_Metrics: 
        adjustedV = float(metricValue_str[:-1]) * 0.01
    # scale to GB/s
    elif metricName in Throughput_Metrics: 
        if "GB/s" in metricValue_str:
            adjustedV = float(metricValue_str[:-4])
        elif "MB/s" in metricValue_str:
            adjustedV = float(metricValue_str[:-4]) * 1e-3
        elif "KB/s" in metricValue_str:
            adjustedV = float(metricValue_str[:-4]) * 1e-6
        elif "B/s" in metricValue_str:
            adjustedV = float(metricValue_str[:-3]) * 1e-9
        else:
            print "Error: unknow throughtput unit!"
            sys.exit(0)
    # convert util() to decimal
    elif metricName in Utilization2decimal_Metrics:
        adjustedV = float(metricValue_str[metricValue_str.find('(') + 1 : metricValue_str.rfind(')')]) * 0.1
        #print('{} in Utilization2decimal_Metrics.'.format(local_metric_value))

    else:
        adjustedV = float(metricValue)


    if adjustedV == None:
        print "Error: adjustedV is None!"
        sys.exit(1)

    #print('{} : {}'.format(metricName, metricValue))
    #print('{} : {}'.format(metricName, adjustedV))

    return adjustedV


#------------------------------------------------------------------------------
# 
#------------------------------------------------------------------------------
def convert_metrics_with_max(df_app, targetmetric):
    """Convert metrics in the dataframe trace.
    If the dataframe has multiple rows, apply max() to the target metric.
    """

    df_metric = df_app.loc[df_app['Metric Name'] == targetmetric]

    maxV = None
    rowcount = 0

    for _, row in df_metric.iterrows():
        adjustedV = adjust_metric(targetmetric, row['Avg'])
        # update maxV
        if rowcount == 0:
            maxV = adjustedV
        else:
            if adjustedV > maxV:
                maxV = adjustedV

        rowcount += 1

    print('For {}, the max value is {}.'.format(targetmetric, maxV))

    return maxV

