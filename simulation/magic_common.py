#!/usr/bin/env python

import os,sys
import random
import time

from subprocess import check_call, STDOUT, CalledProcessError
DEVNULL = open(os.devnull, 'wb', 0)  # no std out


import pandas as pd
import numpy as np
#from math import *
#import operator


#-----------------------------------------------------------------------------#
# shuffle the input app order 
#-----------------------------------------------------------------------------#
def genRandSeq(applist, seed = 99999999):
    random.seed(seed)
    appNum = len(applist)
    idx = [ i for i in xrange(0, appNum)]
    random.shuffle(idx)
    applist_new = [applist[i] for i in idx] 
    return applist_new

#-----------------------------------------------------------------------------#
# go to target dir for each process 
#-----------------------------------------------------------------------------#
class cd:
    """
    Context manager for changing the current working directory
    """

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

#-----------------------------------------------------------------------------#
# Run incoming workload
#-----------------------------------------------------------------------------#
def run_remote(app_dir, devid=0):
    cmd_str = "./run.sh " + str(devid)

    startT = time.time()
    with cd(app_dir):
        # print os.getcwd()
        # print app_dir
        # print cmd_str
        try:
            check_call(cmd_str, stdout=DEVNULL, stderr=STDOUT, shell=True)
        except CalledProcessError as e:
            raise RuntimeError(
                "command '{}' return with error (code {}): {} ({})".format(
                    e.cmd, e.returncode, e.output, app_dir))

    endT = time.time()

    return [startT, endT]

#-----------------------------------------------------------------------------#
# GPU Job Table 
#-----------------------------------------------------------------------------#
def PrintGpuJobTable(GpuJobTable, total_jobs, id2name, saveFile=None):
    """Print application trace."""

    #
    # gpujobtable cols:   
    #            jobid / gpu / status / startT / endT
    #

    # To save, jobID / jobName / start / end / duration  (5 columns)
    if saveFile:
        traceCols = ['jobID', 'appName', 'start', 'end', 'duration (s)']
        df_trace = pd.DataFrame(index=np.arange(0, total_jobs),columns=traceCols)

    print("JobID\tStart\tEnd\tDuration\tAppName")
    start_list = []
    end_list = []
    for row in xrange(total_jobs):
        jobid    = GpuJobTable[row,0]
        startT   = GpuJobTable[row,3]
        endT     = GpuJobTable[row,4]
        duration = endT - startT 
        appName  = id2name[jobid] 

        if saveFile:
            df_trace.loc[row, 'jobID']         = jobid
            df_trace.loc[row, 'appName']       = appName 
            df_trace.loc[row, 'start']         = startT 
            df_trace.loc[row, 'end']           = endT 
            df_trace.loc[row, 'duration (s)']  = duration 

        print("{}\t{}\t{}\t{}\t{}".format(jobid, startT, endT, duration, appName))

        start_list.append(startT)
        end_list.append(endT)

    total_runtime = max(end_list) - min(start_list) 
    print("total runtime = {} (s)".format(total_runtime))

    if saveFile:
        df_trace.to_csv(saveFile, index=False, encoding='utf-8')




#NVPROF_TRACE_COLS = [
#    "Start",
#    "Duration",
#    "Grid X",
#    "Grid Y",
#    "Grid Z",
#    "Block X",
#    "Block Y",
#    "Block Z",
#    "Registers Per Thread",
#    "Static SMem",
#    "Dynamic SMem",
#    "Size",
#    "Throughput",
#    "Device",
#    "Context",
#    "Stream",
#    "Name"]
#
#
#def sort_dict_by_val(inputDict):
#    """
#    Sort dictionary by the value
#    """
#    return sorted(inputDict.items(), key=operator.itemgetter(1))
#
#
#def read_nvprof_trace(traceFile):
#    df_trace = pd.read_csv(traceFile, names=NVPROF_TRACE_COLS, engine='python')
#    rows_to_skip = 0
#    # find out the number of rows to skip
#    for index, row in df_trace.iterrows():
#        if row['Start'] == 'Start':
#            rows_to_skip = index
#            break
#    # read the input csv again
#    df_trace = pd.read_csv(traceFile, skiprows=rows_to_skip)
#    return df_trace
#
#
#def parse_nvprof_trace(df_trace, rowinfoVer=0):
#    """
#    Read dataframe of the nvprof trace
#    Return a list of trace
#    """
#    #-----------------------------------
#    # start parsing the data frame
#    #-----------------------------------
#    start_coef, duration_coef = time_coef_ms(df_trace)  # time unit : ms
#    # print start_coef, duration_coef
#
#    trans_coef = trans_coef_kb(df_trace)  # normalize the transfer size to KB
#    # print trans_coef
#
#    ssm_coef, dsm_coef = sm_coef_bytes(df_trace)  # normalize the size to Bytes
#    # print ssm_coef, dsm_coef
#
#    #-----
#    # read each row for  api type, its timing and transfer size
#    #-----
#    appTrace = []
#    for rowID in xrange(1, df_trace.shape[0]):
#
#        if rowinfoVer == 0:
#            # apiType, starT, endT, Grid, Block, Reg, SharedMem
#            CurrentApi = getRowInfo(
#                df_trace.iloc[[rowID]], start_coef, duration_coef, trans_coef, ssm_coef, dsm_coef)
#        elif rowinfoVer == 1:
#            CurrentApi = getRowInfo_v1(
#                df_trace.iloc[[rowID]], start_coef, duration_coef, trans_coef, ssm_coef, dsm_coef)
#        else:
#            print "something is wrong."
#            sys.exit(1)
#
#        # print CurrentApi
#        appTrace.append(CurrentApi)
#
#    return appTrace
#
#
#def getruntime(appTraceList):
#    """
#    Return the difference between 1st api start and last api end.
#    """
#    # print appTrace1[0][1], appTrace1[-1][2]
#    #print("app runtime (ms) : {}".format(appTms))
#    return appTraceList[-1][2] - appTraceList[0][1]
#
#
#class transfer():
#    def __init__(self, start=0.0, end=0.0, trans_size=0.0):
#        self.start_time_ms = start
#        self.end_time_ms = end
#        self.size = trans_size
#
#
#class streams():
#    def __init__(self):
#        self.h2d = []
#        self.d2h = []
#        self.d2d = []
#        self.memset = []
#        self.kernel = []
#        self.kernel_info = []
#
##------------------------------------------------------------------------------
## Use ms for timing
##------------------------------------------------------------------------------
#def time_coef_ms(df_trace):
#    rows, cols = df_trace.shape
#
#    start_unit = df_trace['Start'].iloc[0]
#    duration_unit = df_trace['Duration'].iloc[0]
#
#    start_coef = 1.0
#    if start_unit == 's':
#        start_coef = 1e3
#    if start_unit == 'us':
#        start_coef = 1e-3
#
#    duration_coef = 1.0
#    if duration_unit == 's':
#        duration_coef = 1e3
#    if duration_unit == 'us':
#        duration_coef = 1e-3
#
#    return start_coef, duration_coef
#
#
##------------------------------------------------------------------------------
## Use bytes for shared memory
##------------------------------------------------------------------------------
#def sm_coef_bytes(df_trace):
#    ssm_unit = df_trace['Static SMem'].iloc[0]
#    dsm_unit = df_trace['Dynamic SMem'].iloc[0]
#
#    ssm_coef = 1.0
#    if ssm_unit == 'KB':
#        ssm_coef = 1e3
#    if ssm_unit == 'MB':
#        ssm_coef = 1e6
#
#    dsm_coef = 1.0
#    if dsm_unit == 'KB':
#        dsm_coef = 1e3
#    if dsm_unit == 'MB':
#        dsm_coef = 1e6
#
#    return ssm_coef, dsm_coef
#
#
##------------------------------------------------------------------------------
## Use KB for data transfer
##------------------------------------------------------------------------------
#def trans_coef_kb(df_trace):
#    size_unit = df_trace['Size'].iloc[0]
#
#    coef = 0.0  # KB
#
#    if size_unit == 'B':
#        coef = 1e-3
#    elif size_unit == 'MB':
#        coef = 1e3
#    elif size_unit == 'GB':
#        coef = 1e6
#    elif size_unit == 'KB':
#        coef = 1.0  # KB
#    else:
#        sys.stderr.write('Unknown Size Unit.\n')
#
#    return coef
#
##------------------------------------------------------------------------------
##  Read current row of the dataframe, return timing and transfer_size
##------------------------------------------------------------------------------
#def getRowInfo(df_row, start_coef_ms, duration_coef_ms,
#               trans_coef, ssm_coef, dsm_coef):
#    """
#    Read the current row for the dataframe, extracting timing only.
#
#    :param df_row:              row of dataframe
#    :param start_coef_ms        start time coef in ms
#    :param duration_coef_m      duration time coef in ms
#
#    :return api_type:           api type : h2d, d2h, memset, kern
#    :return start_time_ms:      the starting time for current api
#    :return end_time_ms:        the ending time for current api
#    :return trans_kb:           the transfer size in KB
#    """
#    # parameters
#    api_name = df_row['Name'].to_string()
#    startT = float(df_row['Start']) * start_coef_ms
#    endT = startT + float(df_row['Duration']) * duration_coef_ms
#    tranSize = 0.0
#    Grid, Block, Reg, SharedMem = 0., 0., 0., 0.
#
#    if "DtoH" in api_name or "DtoA" in api_name:
#        apiType = 'd2h'
#        tranSize = float(df_row.Size) * trans_coef  # d2h size in KB
#    elif "HtoD" in api_name or "HtoA" in api_name:   
#        # NOTE: cudaMemcpyToArray = HtoA
#        apiType = 'h2d'
#        tranSize = float(df_row.Size) * trans_coef  # h2d size in KB
#    elif "DtoD" in api_name:
#        apiType = 'd2d'
#        tranSize = float(df_row.Size) * trans_coef  # d2d size in KB
#    elif "memset" in api_name:
#        apiType = 'memset'
#        tranSize = 0      # no transfer over pcie
#    else:
#        apiType = 'kern'
#        Grid = float(df_row['Grid X']) * \
#            float(df_row['Grid Y']) * float(df_row['Grid Z'])
#        Block = float(df_row['Block X']) * \
#            float(df_row['Block Y']) * float(df_row['Block Z'])
#        Reg = float(df_row['Registers Per Thread'])
#        SharedMem = float(df_row['Static SMem']) * ssm_coef + \
#            float(df_row['Dynamic SMem']) * dsm_coef
#        # print Grid, Block, Reg, SharedMem
#
#    CurrentApi = [apiType, startT, endT, Grid, Block, Reg, SharedMem]
#    return CurrentApi
#
#def getRowInfo_v1(df_row, start_coef_ms, duration_coef_ms,
#               trans_coef, ssm_coef, dsm_coef):
#    """
#    Read the current row for the dataframe, extracting timing only.
#
#    :param df_row:              row of dataframe
#    :param start_coef_ms        start time coef in ms
#    :param duration_coef_m      duration time coef in ms
#
#    :return api_type:           api type : h2d, d2h, memset, kern
#    :return start_time_ms:      the starting time for current api
#    :return end_time_ms:        the ending time for current api
#    :return trans_kb:           the transfer size in KB
#    """
#    # parameters
#    api_name = df_row['Name'].to_string()
#    startT = float(df_row['Start']) * start_coef_ms
#    endT = startT + float(df_row['Duration']) * duration_coef_ms
#    tranSize = 0.0  # transfer siez in unit of KB
#    Grid, Block, Reg, SharedMem = 0., 0., 0., 0.
#
#    if "DtoH" in api_name or "DtoA" in api_name:
#        apiType = 'd2h'
#        tranSize = float(df_row.Size) * trans_coef  # d2h size in KB
#    elif "HtoD" in api_name or "HtoA" in api_name:   
#        # NOTE: cudaMemcpyToArray = HtoA
#        apiType = 'h2d'
#        tranSize = float(df_row.Size) * trans_coef  # h2d size in KB
#    elif "DtoD" in api_name:
#        apiType = 'd2d'
#        tranSize = float(df_row.Size) * trans_coef  # d2d size in KB
#    elif "memset" in api_name:
#        apiType = 'memset'
#        tranSize = 0.      # no transfer over pcie
#    else:
#        apiType = 'kern'
#        Grid = float(df_row['Grid X']) * \
#            float(df_row['Grid Y']) * float(df_row['Grid Z'])
#        Block = float(df_row['Block X']) * \
#            float(df_row['Block Y']) * float(df_row['Block Z'])
#        Reg = float(df_row['Registers Per Thread'])
#        SharedMem = float(df_row['Static SMem']) * ssm_coef + \
#            float(df_row['Dynamic SMem']) * dsm_coef
#        # print Grid, Block, Reg, SharedMem
#
#    CurrentApi = [apiType, startT, endT, Grid, Block, Reg, SharedMem, tranSize]
#    return CurrentApi
#
##------------------------------------------------------------------------------
## Generate Features for NN 
##------------------------------------------------------------------------------
#
## as default, take the 1st 5 api calls , if less than 5, fill in the blanks with 0., 
## if more than 5 calls, leave the others alone
#
## h2d: 1, d2h:2, d2d : 3 , memset 4, kern: 5
## for each api call:  [apitype, startT, endT, A, B, C, D]
## for transfer apis, A-D are zeros
## for kernel, A = gridsize, B = blocksize, C = register usage, D =  shared memory usage
## Therefore, there are 7 floats for each API
#
## Since we use 5 (as default) Api call, we need to preallocate 35 floats per application
#
## Besides, we include 10 feats : 
## 1) app startT, 2) endT, 3) appDur, 
## 4) kernel Calls 5) trans calls 
## 6) kerns_time_ratio 7) trans_time_ratio
## 8) avgKernsTime 9) avgTransTime 
## 10) avgApiGap
#
## Sum up, there are 45 features per applications
#
#def genNNFeat(app_trace, ApiNum = 5):
#    # pre-allocate array
#    feat_num = ApiNum * 7 + 10
#    feat_array = np.zeros(feat_num, dtype=float)
#    
#    transT = 0.
#    kernsT = 0.
#    
#    # 
#    startT, endT = app_trace[0][1], app_trace[-1][2]
#    appDur = endT - startT
#    
#    i = 0
#    gapSum = 0.
#    
#    transCalls = 0.
#    kernelCalls = 0.
#
#    READ_TRACE = True
#    for app in app_trace:
#        if app[0] in ['h2d', 'd2h', 'd2d', 'memset']:
#            transT = transT + (app[2] - app[1])
#            transCalls = transCalls + 1.
#            if READ_TRACE:
#                startpos = i * 7
#                
#                # e.g., ['h2d', 566.138617, 566.141849, 0.0, 0.0, 0.0, 0.0]
#                if app[0] == 'h2d':
#                    app_type_idx = 1
#                elif app[0] == 'd2h':
#                    app_type_idx = 2
#                elif app[0] == 'd2d':
#                    app_type_idx = 3
#                elif app[0] == 'memset':
#                    app_type_idx = 4
#                else:
#                    print("Error! Unknow transfer types!")
#                    sys.exit(1)
#                    
#                    
#                feat_array[startpos] = app_type_idx
#                feat_array[startpos + 1] = app[1] 
#                feat_array[startpos + 2] = app[2] 
#                feat_array[startpos + 3] = app[3] 
#                feat_array[startpos + 4] = app[4] 
#                feat_array[startpos + 5] = app[5] 
#                feat_array[startpos + 6] = app[6] 
#
#                #print feat_array[startpos], feat_array[startpos+1], feat_array[startpos+2], feat_array[startpos+3], feat_array[startpos+4], feat_array[startpos+5], feat_array[startpos+6]
#                
#
#        if app[0] == "kern":
#            app_type_idx = 5
#            kernsT = kernsT + (app[2] - app[1])
#            kernelCalls = kernelCalls + 1.
#            
#            if READ_TRACE:
#                startpos = i * 7
#                feat_array[startpos] = app_type_idx
#                feat_array[startpos + 1] = app[1] 
#                feat_array[startpos + 2] = app[2] 
#                feat_array[startpos + 3] = app[3] 
#                feat_array[startpos + 4] = app[4] 
#                feat_array[startpos + 5] = app[5] 
#                feat_array[startpos + 6] = app[6] 
#                
#                #print feat_array[startpos], feat_array[startpos+1], feat_array[startpos+2], feat_array[startpos+3], feat_array[startpos+4], feat_array[startpos+5], feat_array[startpos+6]
#        
#        if i>0:
#            gapSum = gapSum + (app_trace[i][1] - app_trace[i-1][2])
#            
#        i = i + 1
#        
#        if i > ApiNum:
#            READ_TRACE = False
#    
#    kerns_time_ratio = kernsT / appDur   # sum(kernels_runtime) / app_runtime
#    trans_time_ratio = transT / appDur   # sum(transfer_runtime) / app_runtime
#    
#    avgKernsTime = kernsT / kernelCalls
#    avgTransTime = transT / transCalls
#    
#    # avg interval between two consecutive apis
#    avgApiGap = gapSum / (len(app_trace) - 1)
#    
#    
#    # add the rest
#    startpos = ApiNum * 7
#    feat_array[startpos]     = startT
#    feat_array[startpos + 1] = endT
#    feat_array[startpos + 2] = appDur
#    feat_array[startpos + 3] = kernelCalls
#    feat_array[startpos + 4] = transCalls
#    feat_array[startpos + 5] = kerns_time_ratio
#    feat_array[startpos + 6] = trans_time_ratio
#    feat_array[startpos + 7] = avgKernsTime
#    feat_array[startpos + 8] = avgTransTime
#    feat_array[startpos + 9] = avgApiGap
#    
#    #print feat_array[startpos], feat_array[startpos+1], feat_array[startpos+2], feat_array[startpos+3], feat_array[startpos+4], feat_array[startpos+5], feat_array[startpos+6]
#
#    #print feat_array
#
#    return feat_array
#
#
#def testFeat(app_trace):
#    # pre-allocate array
#    # 
#    startT, endT = app_trace[0][1], app_trace[-1][2]
#    appDur = endT # ret 1
#    gpuDur = endT - startT # ret2
#
#
#
#    gapSum = 0.
#    
#    transT = 0.
#    transCalls = 0.
#
#    kernsT = 0.
#    kernelCalls = 0.
#    KernelTime_list = [] # to compute min/max/mean/std
#
#    for i, app in enumerate(app_trace):
#        if app[0] in ['h2d', 'd2h', 'd2d', 'memset']:
#            transT = transT + (app[2] - app[1])
#            transCalls = transCalls + 1.
#
#        if app[0] == "kern":
#            myKernT = app[2] - app[1] 
#            kernsT = kernsT + myKernT 
#            kernelCalls = kernelCalls + 1.
#            KernelTime_list.append(myKernT)
#                
#        if i>0:
#            gapSum = gapSum + (app_trace[i][1] - app_trace[i-1][2])
#    
#    kerns_time_ratio = kernsT / gpuDur   # sum(kernels_runtime) / app_runtime
#    trans_time_ratio = transT / gpuDur   # sum(transfer_runtime) / app_runtime
#    
#    avgKernsTime = kernsT / kernelCalls
#    avgTransTime = transT / transCalls
#    
#    # avg interval between two consecutive apis
#    avgApiGap = gapSum / float(len(app_trace) - 1)
#    
#    ## add the rest
#    #feat_array[startpos]     = startT
#    #feat_array[startpos + 1] = endT
#    #feat_array[startpos + 2] = appDur
#    #feat_array[startpos + 3] = kernelCalls
#    #feat_array[startpos + 4] = transCalls
#    #feat_array[startpos + 5] = kerns_time_ratio
#    #feat_array[startpos + 6] = trans_time_ratio
#    #feat_array[startpos + 7] = avgKernsTime
#    #feat_array[startpos + 8] = avgTransTime
#    #feat_array[startpos + 9] = avgApiGap
#
#    apiNum = len(app_trace)
#
#    KernelTime_array = np.array(KernelTime_list)
#
#    MinKernT = np.amin(KernelTime_array)
#    MaxKernT = np.amax(KernelTime_array)
#    StdKernT = np.std(KernelTime_array)
#    MeanKernT = np.mean(KernelTime_array)
#    MedianKernT = np.median(KernelTime_array)
#
#
#
#    return appDur, gpuDur, apiNum,transCalls, kernelCalls, kerns_time_ratio, trans_time_ratio, avgKernsTime, avgTransTime, avgApiGap, MinKernT, MaxKernT, MeanKernT, StdKernT, MedianKernT 
#
#
#def testFeat_v1(app_trace):
#    # pre-allocate array
#    # 
#    startT, endT = app_trace[0][1], app_trace[-1][2]
#
#    appDur = endT
#    cpuTime = startT
#    gpuTime = endT - startT 
#
#    # [apiType, startT, endT, Grid, Block, Reg, SharedMem, tranSize]
#    kern_threads_list = []
#    kern_reg_list = []
#    kern_sm_list = []
#    trans_list = []
#    for traceinfo in app_trace:
#        [apiType, _, _, Grid, Block, Reg, SharedMem, tranSize] = traceinfo
#
#        if apiType == 'kern':
#            kern_threads_list.append(Grid * Block)
#            kern_reg_list.append(Reg)
#            kern_sm_list.append(SharedMem)
#        else:
#            trans_list.append(tranSize)
#
#    kern_threads_mat = np.matrix(kern_threads_list)
#    kern_reg_mat     = np.matrix(kern_reg_list)
#    kern_sm_mat      = np.matrix(kern_sm_list)
#    trans_mat        = np.matrix(trans_list)
#
#    threads_max = np.amax(kern_threads_mat)
#    threads_avg = np.mean(kern_threads_mat)
#
#    reg_max = np.amax(kern_reg_mat)
#    reg_avg = np.mean(kern_reg_mat)
#
#    sm_max = np.amax(kern_sm_mat)
#    sm_avg = np.mean(kern_sm_mat)
#
#    trans_max = np.amax(trans_mat)
#    trans_avg = np.mean(trans_mat)
#
#    #
#    # return feature metrics
#    #
#    feature_list = [appDur, cpuTime, gpuTime, 
#            threads_max, threads_avg, 
#            reg_max, reg_avg, 
#            sm_max, sm_avg,
#            trans_max, trans_avg]
#
#    ##gapSum = 0.
#    ##
#    ##transT = 0.
#    ##transCalls = 0.
#
#    ##kernsT = 0.
#    ##kernelCalls = 0.
#    ##KernelTime_list = [] # to compute min/max/mean/std
#
#    ##for i, app in enumerate(app_trace):
#    ##    if app[0] in ['h2d', 'd2h', 'd2d', 'memset']:
#    ##        transT = transT + (app[2] - app[1])
#    ##        transCalls = transCalls + 1.
#
#    ##    if app[0] == "kern":
#    ##        myKernT = app[2] - app[1] 
#    ##        kernsT = kernsT + myKernT 
#    ##        kernelCalls = kernelCalls + 1.
#    ##        KernelTime_list.append(myKernT)
#    ##            
#    ##    if i>0:
#    ##        gapSum = gapSum + (app_trace[i][1] - app_trace[i-1][2])
#    ##
#    ##kerns_time_ratio = kernsT / gpuDur   # sum(kernels_runtime) / app_runtime
#    ##trans_time_ratio = transT / gpuDur   # sum(transfer_runtime) / app_runtime
#    ##
#    ##avgKernsTime = kernsT / kernelCalls
#    ##avgTransTime = transT / transCalls
#    ##
#    ### avg interval between two consecutive apis
#    ##avgApiGap = gapSum / float(len(app_trace) - 1)
#    ##
#    #### add the rest
#    ###feat_array[startpos]     = startT
#    ###feat_array[startpos + 1] = endT
#    ###feat_array[startpos + 2] = appDur
#    ###feat_array[startpos + 3] = kernelCalls
#    ###feat_array[startpos + 4] = transCalls
#    ###feat_array[startpos + 5] = kerns_time_ratio
#    ###feat_array[startpos + 6] = trans_time_ratio
#    ###feat_array[startpos + 7] = avgKernsTime
#    ###feat_array[startpos + 8] = avgTransTime
#    ###feat_array[startpos + 9] = avgApiGap
#
#    ##apiNum = len(app_trace)
#
#    ##KernelTime_array = np.array(KernelTime_list)
#
#    ##MinKernT = np.amin(KernelTime_array)
#    ##MaxKernT = np.amax(KernelTime_array)
#    ##StdKernT = np.std(KernelTime_array)
#    ##MeanKernT = np.mean(KernelTime_array)
#    ##MedianKernT = np.median(KernelTime_array)
#
#
#
#    return feature_list 
