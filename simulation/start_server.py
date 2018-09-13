#!/usr/bin/env python
'''
GPU Workload Scheduler
'''
#
# Ref:
# [1] https://gist.github.com/micktwomey/606178
#

#import math
import Queue
#import operator
#import copy # deepcopy

import numpy as np
import ctypes

from random import randint



#=============================================================================#
# libs 
#=============================================================================#
import os,sys
import socket # ipv4
from subprocess import check_call, STDOUT, CalledProcessError # run/check process 
import time # time the job


# multiprocessing
import multiprocessing as mp
from multiprocessing import Pool, Value, Lock, Manager

# logging
import logging
logging.basicConfig(level=logging.DEBUG)

# read app info
sys.path.append('../prepare')
from app_info import *

# utility func 
sys.path.append('../pycode')
from magicUtil import genRandSeq 


#=============================================================================#
# arguments
#=============================================================================#
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('-g', dest='gpus', default=1, help='gpus to use')
parser.add_argument('-c', dest='clientMode', default=1, help='running with clients (default) or dedicated mode')

#parser.add_argument('-s', dest='scheme', default='rr', help='rr/ll/sim/perf/dinn/simp/perf1/perf2/rrPerf/rrSim/llSim/llSim1/llPerf/lldelay')
#parser.add_argument('-j', dest='jobs', default=0, help='jobs to simulate')
args = parser.parse_args()




#=================================#
# dict used for similarity scheme
#=================================#
#app2cmd = None
#app2dir = None
#app2metric = None
#app2trace = None

# parameters
#JobsPerGPU = 6
#JobsPerGPU = 32
#LARGE_NUM = 1e9

DEVNULL = open(os.devnull, 'wb', 0)  # no std out
magus_debug = False



def getruntime(appTraceList):
    """
    Return the difference between 1st api start and last api end.
    """
    return appTraceList[-1][2] - appTraceList[0][1]


def update_trace_offset(tracelist, offset):
    """
    Adjust the starting time (add offset) to each api call in the traceList.
    """
    for eachApi in tracelist:
        eachApi[1] += offset
        eachApi[2] += offset

def update_trace_api(tracelist, api_index, offset):
    """
    Adjust the starting time (starting from api_index) in the traceList.
    """
    for pid in xrange(len(tracelist)):
        if pid >= api_index:
            tracelist[pid][1] += offset
            tracelist[pid][2] += offset


def adjust_prevTraceTable_api(traceTable, apiID, newStart, oldStart):
    offset = newStart - oldStart
    for api_id, apiCall in enumerate(traceTable):
        # for each api that start before oldStart, remain the same
        # that start after oldStart, add an offset
        myStart, myEnd = apiCall[1], apiCall[2]
        if myStart >= oldStart:
            # add offset
            traceTable[api_id][1] += offset
            traceTable[api_id][2] += offset

#-----------------------------------------------------------------------------#
# 
#-----------------------------------------------------------------------------#
def getCombo(GpuDinnFeats, current_dinnfeats):
    GpuDinnFeats_dd = dict(GpuDinnFeats)

    X= None
    count = 0
    # app1 (running app) + app2 (waiting app)
    for key, value in GpuDinnFeats_dd.iteritems():
        #print value
        combo = np.append(value, current_dinnfeats)
        #print combo.shape
        if count == 0:
            X= combo
        else:
            X= np.vstack((X, combo))
        count = count + 1

    #print "X_input shape"
    #print X_input.shape

    #
    # test X_input on deep learning model
    #

    #print X_input

    #test_results = dpModel.test(X_input, ckpt_model='./dinn/models/dinn_final.ckpt')
    #print test_results


    return X


#
#
#
def model_contention(prevTraceList, newapi, copyEngineNum=2):
    """
    For the newapi, look for contention duing apiStart and apiEnd.
    Default configuration assumes the copy engine number is 2.
    """
    curType, curStart, curEnd = newapi[0], newapi[1], newapi[2]

    #print "\n(Current Api)"
    #print curType, curStart, curEnd

    contentionCount = 0
    adjCurrent, adjTraceTab = False, False

    # iterate all the apps in the traceTable
    for apiID, apiCall in enumerate(prevTraceList):
        preType, preStart, preEnd = apiCall[0], apiCall[1], apiCall[2]

        if (curStart < preEnd <= curEnd) or (curStart <= preStart < curEnd) or (curStart > preStart and curEnd < preEnd):
            if preType == curType:
                contentionCount = contentionCount + 1
                if preStart <= curStart:  # delay current api till the end of prevEnd
                    # print "adjust new api"
                    adjCurrent = True
                    newStart = preEnd
                    oldStart = curStart
                else:  # move the app in traceTable after current api
                    # print "adjust app in traceTable"
                    adjTraceTab = True
                    newStart = curEnd
                    oldStart = preStart
                # find out whether current api has any contention with previous application's api calls 
                return contentionCount, adjCurrent, adjTraceTab, newStart, oldStart, apiID


            if ((preType == 'h2d' and curType == 'd2h') or (preType == 'd2h' and curType == 'h2d')) and (copyEngineNum == 1):
                contentionCount = contentionCount + 1
                # Duplicate previous operations
                if preStart <= curStart:  # delay current api till the end of prevEnd
                    #print "adjust new api"
                    adjCurrent = True
                    newStart = prevEnd
                    oldStart = curStart
                else:  # move the app in traceTable after current api
                    #print "adjust app in traceTable"
                    adjTraceTab = True
                    newStart = curEnd
                    oldStart = preStart
                return contentionCount, adjCurrent, adjTraceTab, newStart, oldStart, apiID

    return contentionCount, adjCurrent, adjTraceTab, None, None, None



def predict_perf(prev_trace_org, current_trace_org):
    """
    Predict performance impact between two application traces
    """
    prev_trace = copy.deepcopy(prev_trace_org)
    current_trace = copy.deepcopy(current_trace_org)

    AvgSlowDown = 0

    #===============#
    # record the orginal runtime 
    #===============#
    orgTime = []
    prev_rt = getruntime(prev_trace)
    orgTime.append(prev_rt)
    ##print "\n=> prev app runtime : %f" % prev_rt

    current_rt = getruntime(current_trace)
    ##print "=> current app runtime : %f" % current_rt
    orgTime.append(current_rt)

    #===============#
    # figure out when to start the coming workload
    #===============#
    # get the ending time of 1st api (for prev app) : [apitype, start, end, .... ]
    prevapp_type  = prev_trace[0][0]
    prevapp_start = prev_trace[0][1]
    prevapp_end   = prev_trace[0][2]

    newapp_type = current_trace[0][0]

    simulate_startPos = None
    extra_delay_for_newapp = 0.

    if prevapp_type == newapp_type:
        # when there is contention, start after prev ends
        simulate_startPos = prevapp_end
        # [Note] count in the starting delay
        extra_delay_for_newapp = prevapp_end - prevapp_start
    else:
        # if different, assume they start at the same time
        simulate_startPos = prevapp_start

    newapp_start = current_trace[0][1] # update new app api starting point

    prev_cur_diff = simulate_startPos - newapp_start  # the amount to adjust the starting point

    newapp_trace = copy.deepcopy(current_trace)

    # sync newapp timing with traceTable
    update_trace_offset(newapp_trace, prev_cur_diff)

    #===============#
    # analyze the contention for each API 
    #===============#
    for i in xrange(len(newapp_trace)):
        api = newapp_trace[i]
        CheckContention = True

        while CheckContention:
            #
            # check contention for current api call
            #
            contentionCount, adjCurrent, adjTraceTab, newStart, oldStart, apiID = model_contention(prev_trace, api)

            if contentionCount == 0:
                CheckContention = False  # move to the next api
            else:
                # there are contention for current api
                #print contentionCount, adjCurrent, adjTraceTab, newStart, appID, apiID

                if adjCurrent:
                    #print "=>adjust current api"
                    #print "before updating api"
                    #print newapp_trace

                    api_offset = newStart - api[1]
                    update_trace_api(newapp_trace, i, api_offset)  # update new app trace list

                    #print "after updating api"
                    #print newapp_trace

                if adjTraceTab:
                    adjust_prevTraceTable_api(prev_trace, apiID, newStart, oldStart)

    #=====================================================#
    # measure slowdown ratio for each application
    #=====================================================#
    newTime = []
    myRuntime = getruntime(prev_trace)
    ##print "\n=> prev app runtime (after adjustment) : %f" % myRuntime 
    newTime.append(myRuntime)

    # add adjusted timing for new app + with extra starting delay
    newTime.append(getruntime(newapp_trace) + extra_delay_for_newapp) 
    ##print "\n=> current app runtime (after adjustment) : %f" % getruntime(newapp_trace)
    
    #=====================================================#
    # measure slowdown ratio for each application
    #=====================================================#
    slowdown_ratio = []
    for i, newT in enumerate(newTime):
        sdr = float(newT) / orgTime[i] - 1.   # compute slowdown ratio
        slowdown_ratio.append(sdr)

    AvgSlowDown = sum(slowdown_ratio) / float(len(newTime))

    return AvgSlowDown


def find_row_for_currentJob(GpuMetricStat_array, jobID):
    """
    Stat array is 32 x 2 where 1st col is status and 2nd col is the jobID
    """
    (rows, cols) = GpuMetricStat_array.shape

    target_row = 0
    FOUND_ROW = False
    for i in xrange(rows):
        # find the active job
        if GpuMetricStat_array[i,1] == jobID  and GpuMetricStat_array[i,0] == 1:
            target_row = i
            FOUND_ROW = True
            break

    if not FOUND_ROW:
        print "\n[ERROR] : No record for jobID in GpuMetricStat_dd! Something is wrong."
        sys.exit(1)
            
    return target_row




def check_availrow_metricarray(stat_array):
    """
    stat_array is 32 x 2 where 2 columns are status and jobID
    """
    avail_row = None 
    [rows, cols] = stat_array.shape
    for i in xrange(rows): # look for the 1st avail (0) row, and return the row
        if stat_array[i,0] == 0:
            avail_row = i
            break
    if avail_row is None:
        logger.info("[*** Warning ***] all 32 slots are busy")
        avial_row = 0
    return avail_row


def check_key(app2dir, app2cmd, app2metric):
    sameApps = True

    # read the keys for each dict
    key_dir = set(app2dir.keys())
    key_cmd = set(app2cmd.keys())
    key_metric = set(app2metric.keys())

    #
    # compare dir with cmd
    #
    if key_dir != key_cmd:
        print "[Error] keys in app2dir and app2cmd are different!"
        if len(key_dir) > len(key_cmd):
            print "[Error] Missing keys for key_cmd"

        if len(key_dir) < len(key_cmd):
            print "[Error] Missing keys for key_dir"

        belong2cmd = key_cmd - key_dir
        belong2dir = key_dir - key_cmd

        onlyincmd = list(belong2cmd)
        onlyindir = list(belong2dir)

        if onlyincmd:  # not empty
            print("[Error] Missing keys: {}".format(onlyincmd))

        if onlyindir:  # not empty
            print("[Error] Missing keys: {}".format(onlyindir))

    else:
        # print "Keys in app2dir and app2cmd match!"
        logging.info("Checking ... ")

    #
    # compare dir with metric
    #
    if key_dir != key_metric:
        print "[Error] keys in app2dir and app2metric are different!"
        if len(key_metric) > len(key_dir):
            print "[Error] Missing keys for key_dir"

        if len(key_metric) < len(key_dir):
            print "[Error] Missing keys for key_metric"

        belong2metric = key_metric - key_dir
        belong2dir = key_dir - key_metric

        onlyinmetric = list(belong2metric)
        onlyindir = list(belong2dir)

        if onlyinmetric:  # not empty
            print("[Error] Missing keys: {}".format(onlyinmetric))

        if onlyindir:  # not empty
            print("[Error] Missing keys: {}".format(onlyindir))

    else:
        # print "Keys in app2dir and app2metric match!"
        logging.info("Checking ... ")

    return sameApps


#
# for each process, go to the target dir
#
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


# def run_remote(app_dir, app_cmd, devid=0):
##    rcuda_select_dev = "RCUDA_DEVICE_" + str(devid) + "=mcx1.coe.neu.edu:" + str(devid)
##    cmd_str = rcuda_select_dev + " " + str(app_cmd)
##
##    startT = time.time()
# with cd(app_dir):
# try:
##            check_call(cmd_str, stdout=DEVNULL, stderr=STDOUT, shell=True)
# except CalledProcessError as e:
##            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
##    endT = time.time()
##
# return [startT, endT]


def run_remote(app_dir, app_cmd, devid=0):
    #rcuda_select_dev = "RCUDA_DEVICE_" + str(devid) + "=mcx1.coe.neu.edu:" + str(devid)
    cmd_str = app_cmd + " " + str(devid)

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


def run_job(app_dir, devid=0):
    cmd_str = "./run.sh " + str(devid)

    startT = time.time()
    with cd(app_dir):
        try:
            check_call(cmd_str, stdout=DEVNULL, stderr=STDOUT, shell=True)
        except CalledProcessError as e:
            raise RuntimeError(
                "command '{}' return with error (code {}): {} ({})".format(
                    e.cmd, e.returncode, e.output, app_dir))

    endT = time.time()

    return [startT, endT]

#------------------------------------------------------------------------------
# Server 
#------------------------------------------------------------------------------
class Server(object):
    def __init__(self, hostname, port, gpuNum=1, clientMode=0):
        self.logger = logging.getLogger("server")
        self.hostname = hostname
        self.port = port
        self.gpuNum = gpuNum                      # Note:  gpus in cluster
        self.clientMode = clientMode          # simulate clients
        self.lock = Lock()
        self.manager = Manager()

    #def find_least_loaded_node(self, GpuJobs_dd, topK=1):
    #    if topK == 1:
    #        ll_dev_list = None 
    #        ll_dev_jobs = None 

    #    if topK >= 2:
    #        ll_dev_list = []
    #        ll_dev_jobs = []

    #    with self.lock:
    #        # sort the dd in ascending order
    #        sorted_stat = sorted(
    #            GpuJobs_dd.items(),
    #            key=operator.itemgetter(1))
    #        # print sorted_stat

    #        if topK > len(sorted_stat):
    #            print "Error! topK is too large."
    #            sys.exit(1)
    #        
    #        if topK == 1:
    #            ll_dev_list = int(sorted_stat[0][0])
    #            ll_dev_jobs = int(sorted_stat[0][1])

    #        if topK >= 2:
    #            for i in xrange(topK):
    #                ll_dev_list.append(int(sorted_stat[i][0]))
    #                ll_dev_jobs.append(int(sorted_stat[i][1]))

    #    return ll_dev_list, ll_dev_jobs 









    #-------------------------------------------------------------------------#
    # 
    #-------------------------------------------------------------------------#
    def scheduler(self, appName, jobID, 
            GpuJobs_dd, 
            GpuMetric_dd, GpuMetricStat_dd,
            GpuTraces_dd, GpuDinnFeats_dd, gpu_to_run, dinn_num,
            scheme='rr'):
        """
        Decide whitch gpu to allocate the job.
        """
        self.logger.debug("(Monitoring)")
        target_dev = 0
        global gpu_options 
        global TFconfig
        #global gpu_to_run

        #--------------------------------
        # check current GPU Node Status
        #--------------------------------
        with self.lock:
            print("\nGpuID\tActiveJobs (job %d)" % jobID)
            for key, value in dict(GpuJobs_dd).iteritems():
                print("{}\t{}".format(key, value))

        if scheme == 'rr':  # round-robin
            target_dev = jobID % self.gpuNum
            with self.lock: GpuJobs_dd[target_dev] = GpuJobs_dd[target_dev] + 1

        elif scheme == 'll':  # least load
            target_dev, _ = self.find_least_loaded_node(GpuJobs_dd)
            with self.lock: GpuJobs_dd[target_dev] = GpuJobs_dd[target_dev] + 1

        elif scheme == 'lldelay':  # least load with delay
            lldev, lldev_jobs = self.find_least_loaded_node(GpuJobs_dd)

            if lldev_jobs < 2:
                target_dev = lldev
                with self.lock: GpuJobs_dd[target_dev] = GpuJobs_dd[target_dev] + 1
            elif lldev_jobs >= 2:
                # wait for 
                while True:
                    #time.sleep(0.5)
                    time.sleep(1)
                    cur_lldev, cur_jobs = self.find_least_loaded_node(GpuJobs_dd)
                    if cur_jobs <=1:
                        target_dev = cur_lldev
                        with self.lock: GpuJobs_dd[target_dev] = GpuJobs_dd[target_dev] + 1
                        break














        elif scheme == 'rrPerf':
            _, lldev_jobs = self.find_least_loaded_node(GpuJobs_dd)
            current_app_trace = app2trace[appName]

            if lldev_jobs == 0:
                target_dev = jobID % self.gpuNum
            else:
                # apply perf model

                AvgSlowDown_list = []
                for gid in xrange(self.gpuNum): 
                    AvgSlowDown = predict_perf(GpuTraces_dd[gid], current_app_trace)
                    AvgSlowDown_list.append(AvgSlowDown)
                #========#
                # look for the smallest slowdown
                #========#
                min_slowdown = LARGE_NUM
                for devid, slowdown_ratio in enumerate(AvgSlowDown_list):
                    if slowdown_ratio < min_slowdown:
                        min_slowdown = slowdown_ratio
                        target_dev = devid

            with self.lock: GpuJobs_dd[target_dev] = GpuJobs_dd[target_dev] + 1

            #=========#
            # update trace on that node 
            #=========#
            with self.lock:
                GpuTraces_dd[target_dev] = current_app_trace 

        #--------------------------------
        # check current GPU Node Status
        #--------------------------------
        elif scheme == 'llSim':
            lldev, lldev_jobs = self.find_least_loaded_node(GpuJobs_dd)
            appMetric = app2metric[appName].as_matrix()

            if lldev_jobs == 0:
                target_dev = lldev 
                with self.lock:
                    GpuJobs_dd[target_dev] = GpuJobs_dd[target_dev] + 1
                    # if there is no jobs on current device, use the 1st row
                    avail_row = 0
                    GpuMetric_array = GpuMetric_dd[target_dev]
                    GpuMetric_array[avail_row,:] = appMetric 
                    GpuMetric_dd[target_dev] = GpuMetric_array 
                    GpuMetricStat_array = GpuMetricStat_dd[target_dev]
                    GpuMetricStat_array[avail_row, : ] = np.array([1, jobID])
                    GpuMetricStat_dd[target_dev] = GpuMetricStat_array 
            else:
                # apply sim 
                with self.lock:
                    min_dist = LARGE_NUM # a quite large number
                    for i in xrange(self.gpuNum): 
                        # max column metric for each gpu in the numpy array
                        currentGpuMetric = np.amax(GpuMetric_dd[i], axis=0)
                        # euclidian dist (currentGpuMetric, appMetric)
                        dist = np.linalg.norm(currentGpuMetric - appMetric)
                        #print "euclidian dist: %f  ( GPU %d )" % (dist, i)

                        #
                        # select the least dist 
                        if dist < min_dist:
                            min_dist =  dist
                            target_dev = i

                    GpuJobs_dd[target_dev] = GpuJobs_dd[target_dev] + 1
                    # =====================
                    # find the row to write
                    # =====================
                    avail_row = check_availrow_metricarray(GpuMetric_dd[target_dev])
                    #========================
                    # add metric to the GpuMetric
                    #========================
                    GpuMetric_array = GpuMetric_dd[target_dev]
                    GpuMetric_array[avail_row,:] = appMetric 
                    GpuMetric_dd[target_dev] = GpuMetric_array 
                    #========================
                    # update stat in GpuMetricStat (32 x 2, stat + jobID)
                    #========================
                    GpuMetricStat_array = GpuMetricStat_dd[target_dev]
                    GpuMetricStat_array[avail_row, : ] = np.array([1, jobID])
                    GpuMetricStat_dd[target_dev] = GpuMetricStat_array 

        #========================
        # 
        #========================
        elif scheme == 'llSim1':
            lldev, lldev_jobs = self.find_least_loaded_node(GpuJobs_dd, topK=2)
            appMetric = app2metric[appName].as_matrix()

            diff = lldev_jobs[1] - lldev_jobs[0]

            if lldev_jobs[0] == 0: 
                target_dev = lldev[0]
                with self.lock:
                    GpuJobs_dd[target_dev] = GpuJobs_dd[target_dev] + 1
                    # if there is no jobs on current device, use the 1st row
                    avail_row = 0
                    GpuMetric_array = GpuMetric_dd[target_dev]
                    GpuMetric_array[avail_row,:] = appMetric 
                    GpuMetric_dd[target_dev] = GpuMetric_array 
                    GpuMetricStat_array = GpuMetricStat_dd[target_dev]
                    GpuMetricStat_array[avail_row, : ] = np.array([1, jobID])
                    GpuMetricStat_dd[target_dev] = GpuMetricStat_array 

            elif lldev_jobs[0] > 0 and diff <=1:
                target_dev = lldev[0]
                with self.lock:
                    GpuJobs_dd[target_dev] = GpuJobs_dd[target_dev] + 1
                    avail_row = check_availrow_metricarray(GpuMetric_dd[target_dev])
                    GpuMetric_array = GpuMetric_dd[target_dev]
                    GpuMetric_array[avail_row,:] = appMetric 
                    GpuMetric_dd[target_dev] = GpuMetric_array 
                    GpuMetricStat_array = GpuMetricStat_dd[target_dev]
                    GpuMetricStat_array[avail_row, : ] = np.array([1, jobID])
                    GpuMetricStat_dd[target_dev] = GpuMetricStat_array 


            else:
                # apply sim 
                with self.lock:
                    min_dist = LARGE_NUM # a quite large number
                    for i in xrange(self.gpuNum): 
                        # max column metric for each gpu in the numpy array
                        currentGpuMetric = np.amax(GpuMetric_dd[i], axis=0)
                        # euclidian dist (currentGpuMetric, appMetric)
                        dist = np.linalg.norm(currentGpuMetric - appMetric)
                        #print "euclidian dist: %f  ( GPU %d )" % (dist, i)

                        #
                        # select the least dist 
                        if dist < min_dist:
                            min_dist =  dist
                            target_dev = i

                    GpuJobs_dd[target_dev] = GpuJobs_dd[target_dev] + 1
                    # =====================
                    # find the row to write
                    # =====================
                    avail_row = check_availrow_metricarray(GpuMetric_dd[target_dev])
                    #========================
                    # add metric to the GpuMetric
                    #========================
                    GpuMetric_array = GpuMetric_dd[target_dev]
                    GpuMetric_array[avail_row,:] = appMetric 
                    GpuMetric_dd[target_dev] = GpuMetric_array 
                    GpuMetricStat_array = GpuMetricStat_dd[target_dev]
                    GpuMetricStat_array[avail_row, : ] = np.array([1, jobID])
                    GpuMetricStat_dd[target_dev] = GpuMetricStat_array 

        elif scheme == 'rrSim':
            lldev, lldev_jobs = self.find_least_loaded_node(GpuJobs_dd)
            appMetric = app2metric[appName].as_matrix()

            if lldev_jobs == 0:
                target_dev = jobID % self.gpuNum

                target_dev_jobs = dict(GpuJobs_dd)[target_dev]
                if target_dev_jobs == 0:
                    with self.lock:
                        GpuJobs_dd[target_dev] = GpuJobs_dd[target_dev] + 1
                        # if there is no jobs on current device, use the 1st row
                        avail_row = 0
                        GpuMetric_array = GpuMetric_dd[target_dev]
                        GpuMetric_array[avail_row,:] = appMetric 
                        GpuMetric_dd[target_dev] = GpuMetric_array 
                        GpuMetricStat_array = GpuMetricStat_dd[target_dev]
                        GpuMetricStat_array[avail_row, : ] = np.array([1, jobID])
                        GpuMetricStat_dd[target_dev] = GpuMetricStat_array 
                else:
                    # apply sim 
                    with self.lock:
                        min_dist = LARGE_NUM # a quite large number
                        for i in xrange(self.gpuNum): 
                            # max column metric for each gpu in the numpy array
                            currentGpuMetric = np.amax(GpuMetric_dd[i], axis=0)
                            # euclidian dist (currentGpuMetric, appMetric)
                            dist = np.linalg.norm(currentGpuMetric - appMetric)
                            #print "euclidian dist: %f  ( GPU %d )" % (dist, i)

                            #
                            # select the least dist 
                            if dist < min_dist:
                                min_dist =  dist
                                target_dev = i

                        GpuJobs_dd[target_dev] = GpuJobs_dd[target_dev] + 1
                        # =====================
                        # find the row to write
                        # =====================
                        avail_row = check_availrow_metricarray(GpuMetric_dd[target_dev])
                        #========================
                        # add metric to the GpuMetric
                        #========================
                        GpuMetric_array = GpuMetric_dd[target_dev]
                        GpuMetric_array[avail_row,:] = appMetric 
                        GpuMetric_dd[target_dev] = GpuMetric_array 
                        #========================
                        # update stat in GpuMetricStat (32 x 2, stat + jobID)
                        #========================
                        GpuMetricStat_array = GpuMetricStat_dd[target_dev]
                        GpuMetricStat_array[avail_row, : ] = np.array([1, jobID])
                        GpuMetricStat_dd[target_dev] = GpuMetricStat_array 


            else:
                # apply sim 
                with self.lock:
                    min_dist = LARGE_NUM # a quite large number
                    for i in xrange(self.gpuNum): 
                        # max column metric for each gpu in the numpy array
                        currentGpuMetric = np.amax(GpuMetric_dd[i], axis=0)
                        # euclidian dist (currentGpuMetric, appMetric)
                        dist = np.linalg.norm(currentGpuMetric - appMetric)
                        #print "euclidian dist: %f  ( GPU %d )" % (dist, i)

                        #
                        # select the least dist 
                        if dist < min_dist:
                            min_dist =  dist
                            target_dev = i

                    GpuJobs_dd[target_dev] = GpuJobs_dd[target_dev] + 1
                    # =====================
                    # find the row to write
                    # =====================
                    avail_row = check_availrow_metricarray(GpuMetric_dd[target_dev])
                    #========================
                    # add metric to the GpuMetric
                    #========================
                    GpuMetric_array = GpuMetric_dd[target_dev]
                    GpuMetric_array[avail_row,:] = appMetric 
                    GpuMetric_dd[target_dev] = GpuMetric_array 
                    #========================
                    # update stat in GpuMetricStat (32 x 2, stat + jobID)
                    #========================
                    GpuMetricStat_array = GpuMetricStat_dd[target_dev]
                    GpuMetricStat_array[avail_row, : ] = np.array([1, jobID])
                    GpuMetricStat_dd[target_dev] = GpuMetricStat_array 

        #---------------------------------------------------------------------#
        # Similarity 
        #---------------------------------------------------------------------#
        elif scheme == 'sim': 
            # print app2metric[appName]
            appMetric = app2metric[appName].as_matrix()
            #print appMetric 
            #print appMetric.size

            #-------------------------#
            # check gpu node metrics
            # 1) use 'll' to find the vacant node
            # 2) Given all nodes are busy, select node with the least euclidean
            # distance
            #-------------------------#
            #current_dev, current_jobs = self.find_least_loaded_node(GpuJobs_dd)
            current_dev, current_jobs = self.find_least_loaded_node(GpuJobs_dd)

            if current_jobs == 0:
                target_dev = current_dev
                
                #-------------------------#
                # add job metrics to the GpuMetric
                # update GpuMetricStat
                #-------------------------#
                with self.lock:
                    GpuJobs_dd[target_dev] = GpuJobs_dd[target_dev] + 1

                    # if there is no jobs on current device, use the 1st row
                    avail_row = 0

                    #------------------#
                    # add metric to the GpuMetric
                    #------------------#
                    GpuMetric_array = GpuMetric_dd[target_dev]
                    #print type(GpuMetric_array)
                    #print "\norg:"
                    #print GpuMetric_array[avail_row,:]
                    GpuMetric_array[avail_row,:] = appMetric 
                    #print "\nafter:"
                    #print GpuMetric_array[avail_row,:]
                    GpuMetric_dd[target_dev] = GpuMetric_array 

                    #print "\n\nUpdated Metric : "
                    #print GpuMetric_dd[target_dev]

                    #------------------#
                    # update stat in GpuMetricStat (32 x 2, stat + jobID)
                    #------------------#
                    GpuMetricStat_array = GpuMetricStat_dd[target_dev]
                    GpuMetricStat_array[avail_row, : ] = np.array([1, jobID])
                    GpuMetricStat_dd[target_dev] = GpuMetricStat_array 

                    #print "\n\nUpdated Metric Stat : "
                    #print GpuMetricStat_dd[target_dev]
                    
                    #avail_row = check_availrow_metricarray(stat_array)

            elif current_jobs > 0:
                #
                # select the least similar GPU node to run 
                #


                #
                # what is each GPU's (max) metric ?  
                #
                with self.lock:
                    print "\nCheck GpuMetric_dd\n"
                    #for key, value in GpuMetric_dd.iteritems():
                    #    print key

                    min_dist = LARGE_NUM # a quite large number

                    for i in xrange(self.gpuNum): 
                        #
                        # max column metric for each gpu in the numpy array
                        currentGpuMetric = np.amax(GpuMetric_dd[i], axis=0)

                        # 
                        # euclidian dist (currentGpuMetric, appMetric)
                        dist = np.linalg.norm(currentGpuMetric - appMetric)
                        print "euclidian dist: %f  ( GPU %d )" % (dist, i)

                        #
                        # select the least dist 
                        if dist < min_dist:
                            min_dist =  dist
                            target_dev = i

                    #
                    # after selection,  1) add current appMetric to GpuMetric
                    #                   2) update GpuMetricStat

                    GpuJobs_dd[target_dev] = GpuJobs_dd[target_dev] + 1
                    # =====================
                    # find the row to write
                    # =====================
                    avail_row = check_availrow_metricarray(GpuMetric_dd[target_dev])


                    #========================
                    # add metric to the GpuMetric
                    #========================
                    GpuMetric_array = GpuMetric_dd[target_dev]
                    GpuMetric_array[avail_row,:] = appMetric 
                    GpuMetric_dd[target_dev] = GpuMetric_array 

                    #========================
                    # update stat in GpuMetricStat (32 x 2, stat + jobID)
                    #========================
                    GpuMetricStat_array = GpuMetricStat_dd[target_dev]
                    GpuMetricStat_array[avail_row, : ] = np.array([1, jobID])
                    GpuMetricStat_dd[target_dev] = GpuMetricStat_array 

                #
                # log
                #
                self.logger.debug("\n[Similarity] select GPU %r\n", target_dev)
                    
            else:
                self.logger.debug(
                    "[Error!] gpu node job is negative! Existing...")
                sys.exit(1)

        #---------------------------------------------------------------------#
        # Performance Model
        #---------------------------------------------------------------------#
        elif scheme == 'perf':  # performance model 
            #print "running perfModel"
            current_app_trace = app2trace[appName]
            #print len(current_app_trace)

            #-------------------------#
            # 1) use 'll' to find the vacant node
            # 2) Given all nodes are busy, select node with the least performance impact 
            #-------------------------#
            current_dev, current_jobs = self.find_least_loaded_node(GpuJobs_dd)

            #-----------#
            # use vacant gpu when there is no worloads 
            #-----------#
            if current_jobs == 0:
                target_dev = current_dev
                
                #-------------------------#
                # add job trace to the GpuTraces
                #-------------------------#
                with self.lock:
                    GpuJobs_dd[target_dev] = GpuJobs_dd[target_dev] + 1
                    GpuTraces_dd[target_dev] = current_app_trace 

            #-----------#
            # When there has been active jobs running on current device,
            # use performance 
            #-----------#
            elif current_jobs > 0:

                AvgSlowDown_list = []
                for gid in xrange(self.gpuNum): 
                    AvgSlowDown = predict_perf(GpuTraces_dd[gid], current_app_trace)
                    #print AvgSlowDown
                    AvgSlowDown_list.append(AvgSlowDown)

                #========#
                # look for the smallest slowdown
                #========#
                min_slowdown = LARGE_NUM
                for devid, slowdown_ratio in enumerate(AvgSlowDown_list):
                    if slowdown_ratio < min_slowdown:
                        min_slowdown = slowdown_ratio
                        target_dev = devid

                #=========#
                # update trace on that node 
                #=========#
                with self.lock:
                    GpuJobs_dd[target_dev] = GpuJobs_dd[target_dev] + 1
                    GpuTraces_dd[target_dev] = current_app_trace 



            else:
                self.logger.debug(
                    "[Error!] gpu job is negative! Existing...")
                sys.exit(1)

        #---------------------------------------------------------------------#
        # Performance Model: v1
        #---------------------------------------------------------------------#
        elif scheme == 'perf1':  # performance model 
            current_app_trace = app2trace[appName]
            twodev, twodev_jobs = self.find_least_loaded_node(GpuJobs_dd, topK=2)

            firstdev_jobs = twodev_jobs[0]
            secnddev_jobs = twodev_jobs[1]

            #-----------#
            # use vacant gpu when there is no worloads 
            #-----------#
            if firstdev_jobs == 0:
                target_dev = twodev[0] 
                #-------------------------#
                # add job trace to the GpuTraces
                #-------------------------#
                with self.lock:
                    GpuJobs_dd[target_dev] = GpuJobs_dd[target_dev] + 1
                    GpuTraces_dd[target_dev] = current_app_trace 

            #-----------#
            # When there has been active jobs running on current device,
            # use performance 
            #-----------#
            elif firstdev_jobs > 0:

                if firstdev_jobs == secnddev_jobs:
                    AvgSlowDown_list = []
                    for gid in xrange(self.gpuNum): 
                        AvgSlowDown = predict_perf(GpuTraces_dd[gid], current_app_trace)
                        #print AvgSlowDown
                        AvgSlowDown_list.append(AvgSlowDown)

                    #========#
                    # look for the smallest slowdown
                    #========#
                    min_slowdown = LARGE_NUM
                    for devid, slowdown_ratio in enumerate(AvgSlowDown_list):
                        if slowdown_ratio < min_slowdown:
                            min_slowdown = slowdown_ratio
                            target_dev = devid

                else: # select the min job device
                    if firstdev_jobs < secnddev_jobs:
                        target_dev = twodev[0]
                    else:
                        target_dev = twodev[1]

                #=========#
                # update trace on that node 
                #=========#
                with self.lock:
                    GpuJobs_dd[target_dev] = GpuJobs_dd[target_dev] + 1
                    GpuTraces_dd[target_dev] = current_app_trace 



            else:
                self.logger.debug(
                    "[Error!] gpu job is negative! Existing...")
                sys.exit(1)

        #---------------------------------------------------------------------#
        # Performance Model: v2
        #---------------------------------------------------------------------#
        elif scheme == 'perf2':  # performance model 
            current_app_trace = app2trace[appName]
            twodev, twodev_jobs = self.find_least_loaded_node(GpuJobs_dd, topK=2)
            firstdev_jobs = twodev_jobs[0]
            secnddev_jobs = twodev_jobs[1]
            if firstdev_jobs == 0:
                target_dev = twodev[0] 
                with self.lock:
                    GpuJobs_dd[target_dev] = GpuJobs_dd[target_dev] + 1
                    GpuTraces_dd[target_dev] = current_app_trace 
            elif firstdev_jobs > 0:
                if abs(firstdev_jobs - secnddev_jobs) <= 1: # job number difference <=1
                    AvgSlowDown_list = []
                    for gid in xrange(self.gpuNum): 
                        AvgSlowDown = predict_perf(GpuTraces_dd[gid], current_app_trace)
                        #print AvgSlowDown
                        AvgSlowDown_list.append(AvgSlowDown)

                    #========#
                    # look for the smallest slowdown
                    #========#
                    min_slowdown = LARGE_NUM
                    for devid, slowdown_ratio in enumerate(AvgSlowDown_list):
                        if slowdown_ratio < min_slowdown:
                            min_slowdown = slowdown_ratio
                            target_dev = devid

                else: # select the min job device
                    if firstdev_jobs < secnddev_jobs:
                        target_dev = twodev[0]
                    else:
                        target_dev = twodev[1]

                #=========#
                # update trace on that node 
                #=========#
                with self.lock:
                    GpuJobs_dd[target_dev] = GpuJobs_dd[target_dev] + 1
                    GpuTraces_dd[target_dev] = current_app_trace 



            else:
                self.logger.debug(
                    "[Error!] gpu job is negative! Existing...")
                sys.exit(1)

        #---------------------------------------------------------------------#
        # llPerf 
        #---------------------------------------------------------------------#
        elif scheme == 'llPerf':  # performance model 
            current_app_trace = app2trace[appName]
            twodev, twodev_jobs = self.find_least_loaded_node(GpuJobs_dd, topK=2)

            diff = twodev_jobs[1] - twodev_jobs[0]

            if twodev_jobs[0] == 0:
                target_dev = twodev[0] 
                with self.lock:
                    GpuJobs_dd[target_dev] = GpuJobs_dd[target_dev] + 1
                    GpuTraces_dd[target_dev] = current_app_trace 

            elif diff >= 2:
                target_dev = twodev[0] 
                with self.lock:
                    GpuJobs_dd[target_dev] = GpuJobs_dd[target_dev] + 1
                    GpuTraces_dd[target_dev] = current_app_trace 

            else:
                AvgSlowDown_list = []
                for gid in xrange(self.gpuNum):
                    AvgSlowDown = predict_perf(GpuTraces_dd[gid], current_app_trace)
                    #print AvgSlowDown
                    AvgSlowDown_list.append(AvgSlowDown)

                #========#
                # look for the smallest slowdown
                #========#
                min_slowdown = LARGE_NUM
                for devid, slowdown_ratio in enumerate(AvgSlowDown_list):
                    if slowdown_ratio < min_slowdown:
                        min_slowdown = slowdown_ratio
                        target_dev = devid

                #=========#
                # update trace on that node 
                #=========#
                with self.lock:
                    GpuJobs_dd[target_dev] = GpuJobs_dd[target_dev] + 1
                    GpuTraces_dd[target_dev] = current_app_trace 



        #---------------------------------------------------------------------#
        # SIMP 
        #---------------------------------------------------------------------#
        elif scheme == 'simp':  
            appMetric = app2metric[appName].as_matrix() # get metric
            current_app_trace = app2trace[appName] # get trace

            lldev_list, lldev_jobs = self.find_least_loaded_node(GpuJobs_dd, topK=2)
            
            first_ll_node_job = lldev_jobs[0]
            #secnd_ll_node_job = lldev_jobs[1]

            # check the 1st ll node
            if first_ll_node_job == 0:
                target_dev = lldev_list[0] # use the 1st ll device 
                with self.lock:
                    GpuJobs_dd[target_dev] = GpuJobs_dd[target_dev] + 1
                    avail_row = 0
                    # add metric to the GpuMetric
                    GpuMetric_array = GpuMetric_dd[target_dev]
                    GpuMetric_array[avail_row,:] = appMetric 
                    GpuMetric_dd[target_dev] = GpuMetric_array 
                    GpuMetricStat_array = GpuMetricStat_dd[target_dev]
                    GpuMetricStat_array[avail_row, : ] = np.array([1, jobID])
                    GpuMetricStat_dd[target_dev] = GpuMetricStat_array 
                    # add trace to the gpu
                    GpuTraces_dd[target_dev] = current_app_trace 

            elif first_ll_node_job == 1: # use perfmodel 
                AvgSlowDown_list = []
                for gid in xrange(self.gpuNum): 
                    AvgSlowDown = predict_perf(GpuTraces_dd[gid], current_app_trace)
                    #print AvgSlowDown
                    AvgSlowDown_list.append(AvgSlowDown)
                # look for the smallest slowdown
                min_slowdown = LARGE_NUM
                for devid, slowdown_ratio in enumerate(AvgSlowDown_list):
                    if slowdown_ratio < min_slowdown:
                        min_slowdown = slowdown_ratio
                        target_dev = devid

                with self.lock:
                    GpuJobs_dd[target_dev] = GpuJobs_dd[target_dev] + 1
                    # update trace on that node 
                    GpuTraces_dd[target_dev] = current_app_trace 
                    # find the row to write
                    avail_row = check_availrow_metricarray(GpuMetric_dd[target_dev])
                    GpuMetric_array = GpuMetric_dd[target_dev]
                    GpuMetric_array[avail_row,:] = appMetric 
                    GpuMetric_dd[target_dev] = GpuMetric_array 
                    GpuMetricStat_array = GpuMetricStat_dd[target_dev]
                    GpuMetricStat_array[avail_row, : ] = np.array([1, jobID])
                    GpuMetricStat_dd[target_dev] = GpuMetricStat_array 

            elif first_ll_node_job >=2 : # use similarity
                with self.lock:
                    min_dist = LARGE_NUM # a quite large number
                    for i in xrange(self.gpuNum): 
                        # max column metric for each gpu in the numpy array
                        currentGpuMetric = np.amax(GpuMetric_dd[i], axis=0)
                        # euclidian dist 
                        dist = np.linalg.norm(currentGpuMetric - appMetric)
                        #print "euclidian dist: %f  ( GPU %d )" % (dist, i)
                        if dist < min_dist:
                            min_dist =  dist
                            target_dev = i
                    # update trace on the device
                    GpuJobs_dd[target_dev] = GpuJobs_dd[target_dev] + 1
                    GpuTraces_dd[target_dev] = current_app_trace 

                # find the row to write
                avail_row = check_availrow_metricarray(GpuMetric_dd[target_dev])
                GpuMetric_array = GpuMetric_dd[target_dev]
                GpuMetric_array[avail_row,:] = appMetric 
                GpuMetric_dd[target_dev] = GpuMetric_array 
                GpuMetricStat_array = GpuMetricStat_dd[target_dev]
                GpuMetricStat_array[avail_row, : ] = np.array([1, jobID])
                GpuMetricStat_dd[target_dev] = GpuMetricStat_array 

            else:
                self.logger.debug(
                    "[Error!] gpu job is negative! Existing... (simp)")
                sys.exit(1)




        #---------------------------------------------------------------------#
        # DINN Model : 
        # 1) filter out good candidate to co-run, 
        # 2) then use perfmodel to select the best
        #---------------------------------------------------------------------#
        elif scheme == 'dinn':  # deep interference performance model 
            current_dinnfeats = app2dinnfeats[appName]
            #print current_dinnfeats
            current_trace = app2trace[appName]

            #-----------------------------------------------------------------#
            # 1) use 'll' to find the vacant node
            # 2) Given all nodes are busy, select node with least perf impact 
            #-----------------------------------------------------------------#
            current_dev, current_jobs = self.find_least_loaded_node(GpuJobs_dd)

            if current_jobs == 0:
                target_dev = current_dev
                with self.lock:
                    GpuJobs_dd[target_dev] = GpuJobs_dd[target_dev] + 1
                    GpuTraces_dd[target_dev]    = current_trace 
                    GpuDinnFeats_dd[target_dev] = current_dinnfeats 

            elif current_jobs > 0: # when there are active jobs running
                #print ">>> running DINN"


                #DINN_2_PERF = False
                DINN_2_LL = False

                with self.lock:
                    if dinn_num.value == 0: # check the limit
                        #DINN_2_PERF = True
                        DINN_2_LL = True

                if DINN_2_LL:
                    target_dev = current_dev
                    with self.lock:
                        GpuJobs_dd[target_dev] = GpuJobs_dd[target_dev] + 1
                    GpuTraces_dd[target_dev] = current_trace 
                    GpuDinnFeats_dd[target_dev] = current_dinnfeats 

                ##if DINN_2_PERF:
                ##    print ">>> fall back to perf model"
                ##    AvgSlowDown_list = []
                ##    for gid in xrange(self.gpuNum): 
                ##        AvgSlowDown = predict_perf(GpuTraces_dd[gid], current_trace)
                ##        AvgSlowDown_list.append(AvgSlowDown)
                ##    #========#
                ##    # look for the smallest slowdown
                ##    #========#
                ##    min_slowdown = LARGE_NUM
                ##    for devid, slowdown_ratio in enumerate(AvgSlowDown_list):
                ##        if slowdown_ratio < min_slowdown:
                ##            min_slowdown = slowdown_ratio
                ##            target_dev = devid
                ##    #=========#
                ##    # update trace on that node 
                ##    #=========#
                ##    with self.lock:
                ##        GpuTraces_dd[target_dev]    = current_trace 
                ##        GpuDinnFeats_dd[target_dev] = current_dinnfeats 

                #==========#
                # USE DINN
                #==========#
                else:
                    print ">>> running DINN"
                    pred_array = None
                    # update asap
                    with self.lock:
                        dinn_num.value = dinn_num.value - 1

                    ### select the good candidates among all the gpus using dpModel 
                    ##with self.lock:
                    ##    ### randomly pick 7-11 gpu
                    ##    ##tf_dev = randint(7,11) 
                    ##    ##os.environ['CUDA_VISIBLE_DEVICES'] = str(tf_dev) 

                    ##    ### from 7 to 11
                    ##    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_to_run.value) 
                    ##    gpu_to_run.value = gpu_to_run.value + 1
                    ##    if gpu_to_run.value == 12:
                    ##        gpu_to_run.value = 8  # start from gpu 8

                    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
                    sess = tf.Session(config=TFconfig)
                    dpModel = dinn(sess) # init a dinn class
                    X_input = getCombo(GpuDinnFeats_dd, current_dinnfeats)
                    #print X_input.shape
                    # predict using the deep learning model
                    test_results = dpModel.test(X_input, ckpt_model='./dinn/models/dinn_final.ckpt')
                    pred_array = test_results[0] # NOTE: test_results is a list
                    sess.close()
                    reset_graph()

                    # double check
                    if pred_array is None:
                        print(">>> Error! pred_array is None!")

                    good_list = []
                    for i in xrange(self.gpuNum):
                        #print pred_array[i,0], pred_array[i,1]
                        [bad,good] = pred_array[i,:]
                        if good > bad:
                            good_list.append(i)

                    if len(good_list) == 1: # when there is only one candidate
                        target_dev = int(good_list[0])
                    else: # either there are 2+ options or 0 options
                        #print ">>> run perf model"
                        AvgSlowDown_list = []
                        for gid in good_list:
                            AvgSlowDown = predict_perf(GpuTraces_dd[gid], current_trace)
                            AvgSlowDown_list.append(AvgSlowDown)
                        #print AvgSlowDown_list

                        # look for the smallest slowdown
                        min_slowdown = LARGE_NUM
                        for idx, slowdown_ratio in enumerate(AvgSlowDown_list):
                            if slowdown_ratio < min_slowdown:
                                min_slowdown = slowdown_ratio
                                target_dev = int(good_list[idx])

                    # update trace on that node 
                    with self.lock:
                        GpuTraces_dd[target_dev] = current_trace 
                        GpuDinnFeats_dd[target_dev] = current_dinnfeats 
                        dinn_num.value = dinn_num.value + 1 # update the dinn_num

            else: # in case the job number is < 0
                self.logger.debug(
                    "[Error!] gpu job is negative! Existing...")
                sys.exit(1)
        else: # when the scheme is not defined
            self.logger.debug("Unknown scheduling scheme!")
            sys.exit(1)

        return target_dev


    #-------------------------------------------------------------------------#
    # 
    #-------------------------------------------------------------------------#
    def PrintGpuJobTable(self, GpuJobTable, total_jobs):
        #--------------------------------------------------------------
        # gpu job table : 5 columns
        #    jobid      gpu     status      starT       endT
        #--------------------------------------------------------------
        print("JobID\tGPU\tStatus\tStart\t\t\tEnd\t\t\tDuration")
        for row in xrange(total_jobs):
            print("{}\t{}\t{}\t{}\t\t{}\t\t{}".format(GpuJobTable[row, 0],
                                                      GpuJobTable[row, 1],
                                                      GpuJobTable[row, 2],
                                                      GpuJobTable[row, 3],
                                                      GpuJobTable[row, 4],
                                                      GpuJobTable[row, 4] - GpuJobTable[row, 3]))
    #-------------------------------------------------------------------------#
    # Dispatch without clients 
    #-------------------------------------------------------------------------#
    def dispatch_wo_client(self, jobID, appName, appDir, AppStat, GpuStat, gpu=0):
        """dispatch directly to the default gpu 0"""
        self.logger.debug("Job=%r, appName=%r, gpu=%r\n", jobID, appName, gpu)

        # assign to gpu

        #-----------------------------------------
        # update app stat table
        #
        # (columns)   
        # jobid      gpu     status      starT       endT
        #-----------------------------------------
        AppStat[jobID, 0] = jobID
        AppStat[jobID, 1] = gpu 
        AppStat[jobID, 2] = 0

        #--------------------------------------------------------------
        # work on the job
        #--------------------------------------------------------------
        [startT, endT] = run_job(app_dir=appDir, devid=gpu)
        self.logger.debug("(Job {}) {} to {} = {:.3f} seconds".format(jobID, startT, endT, endT - startT))

        #--------------------------------------------------------------
        # The job is done!
        #--------------------------------------------------------------



    #-------------------------------------------------------------------------#
    # Run incoming workload
    #-------------------------------------------------------------------------#
    def handleWorkload(self, connection, address, jobID,
                       GpuJobTable, GpuJobs_dd, GpuMetric_dd, GpuMetricStat_dd,
                       GpuTraces_dd, GpuDinnFeats_dd, gpu_to_run, dinn_num):
        '''
        schedule workloads on the gpu
        '''
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger("process-%r" % (address,))

        try:
            #logger.debug("Connected %r at %r", connection, address)
            logger.debug("Connected")
            while True:
                #--------------------------------------------------------------
                # 1) receive data
                #--------------------------------------------------------------
                data = connection.recv(1024)
                if data == "":
                    logger.debug("Socket closed remotely")
                    break

                if data == "end_simulation":
                    RunServer = False

                logger.debug("Received data %r", data)

                appName = data

                #------------------------------------#
                # 2) get the app_dir, app_cmd
                #------------------------------------#
                app_dir = app2dir[appName]
                app_cmd = app2cmd[appName]

                #--------------------------------------------------------------
                # 3) scheduler : with different schemes
                #--------------------------------------------------------------
                target_gpu = self.scheduler(appName, jobID, 
                        GpuJobs_dd, GpuMetric_dd, GpuMetricStat_dd,
                        GpuTraces_dd, GpuDinnFeats_dd, gpu_to_run, dinn_num,
                        scheme=args.scheme)

                self.logger.debug("TargetGPU-%r (job %r)", target_gpu, jobID)

                #-----------------------------------------
                # 4) update gpu job table
                # (columns)   jobid      gpu     status      starT       endT
                #-----------------------------------------
                # Assign job to the target GPU
                GpuJobTable[jobID, 0] = jobID
                GpuJobTable[jobID, 1] = target_gpu
                GpuJobTable[jobID, 2] = 0

                #--------------------------------------------------------------
                # 5) add job to Gpu Node
                #--------------------------------------------------------------
                #with self.lock:
                #    GpuJobs_dd[target_gpu] = GpuJobs_dd[target_gpu] + 1


                #--------------------------------------------------------------
                # 6) work on the job
                #--------------------------------------------------------------
                [startT, endT] = run_remote(app_dir=app_dir, app_cmd=app_cmd, devid=target_gpu)
                #print("{} to {} = {:.3f} seconds".format(startT, endT, endT - startT))
                self.logger.debug(
                    "(Job {}) {} to {} = {:.3f} seconds".format(
                        jobID, startT, endT, endT - startT))

                #--------------------------------------------------------------
                # The job is done!
                #--------------------------------------------------------------

                #--------------------------------------------------------------
                # 7) delete the job, update Gpu Node information
                #--------------------------------------------------------------
                with self.lock:
                    GpuJobs_dd[target_gpu] = GpuJobs_dd[target_gpu] - 1 

                    if args.scheme in ["sim", "simp", "rrSim", "llSim", "llSim1"]:
                        #========================
                        # Find the corresponding row for the current jobID 
                        #========================
                        GpuMetricStat_array = GpuMetricStat_dd[target_gpu]
                        #
                        # find the right row to update
                        myrow = find_row_for_currentJob(GpuMetricStat_array, jobID)
                        #
                        # reset the metric stat
                        GpuMetricStat_array[myrow, : ] = np.array([0, -1])
                        GpuMetricStat_dd[target_gpu] = GpuMetricStat_array 
                        #
                        # del metric in the GpuMetric / reset to zeros
                        GpuMetric_array = GpuMetric_dd[target_gpu]
                        GpuMetric_array[myrow,:] = np.zeros((1,26))
                        GpuMetric_dd[target_gpu] = GpuMetric_array 

                    #==================
                    # Note: for perf model, we use the latest trace for the gpu
                    # when there is no job running, the new trace will be added
                    # There is no need to del it. 
                    # 1) If currently there is only job, the new app will be added
                    # 2) If there was some jobs running, only the new job is added 
                    #==================


                #--------------------------------------------------------------
                # 8) update gpu job table
                #
                # 5 columns:
                #    jobid      gpu     status      starT       endT
                #--------------------------------------------------------------
                # mark the job is done, and update the timing info
                GpuJobTable[jobID, 2] = 1  # done
                GpuJobTable[jobID, 3] = startT
                GpuJobTable[jobID, 4] = endT

                #self.logger.debug("%r ", GpuJobTable[:5,:])

                # job_q.put(int(data))

                # for elem in list(job_q.queue):
                #    print elem

                #print("current queue size : {}".format(job_q.qsize()))
                #
                # 1) as we receive data, we put into the queue
                # (don't work on that asap)
                #job_param = None
                # if not job_q.empty():
                #    job_param = job_q.get()
                #    print("work on {}".format(job_param))

                #print("after dequeue size : {}".format(job_q.qsize()))

                # 2) start a new process
                #newjob = mp.Process(target=foo, args=(int(data),))
                # newjob.start()

                # run_job()

                #result = pool.apply_async(foo, (2000 * int(data), ))
                #[startT, endT] = result.get()

                # connection.sendall(data) # send feedback
                #logger.debug("Sent data")
        except BaseException:
            logger.exception("Problem handling request")
        finally:
            logger.debug("Closing socket")
            connection.close()

    #--------------------------------------------------------------------------
    # Simulation without clients 
    #--------------------------------------------------------------------------
    def simu_dedicate(self, AppStat, GpuStat):
        """Run simulation without clients"""

        #
        # input: app, app2dir_dd in app_info.py
        #
        if len(app) <> len(app2dir_dd):
            print "Error: app number wrong, check ../prepare/app_info.py!"
            sys.exit(1)

        app_num = len(app)

        # step 1: read all the apps
        print("Run simulation without clients.\n" \
            + "total apps={}\n".format(app_num))

        # step 2: three random sequences
        app_s1 = genRandSeq(app, seed=31415926) # pi
        app_s2 = genRandSeq(app, seed=161803398875) # golden ratio
        app_s3 = genRandSeq(app, seed=299792458) # speed of light

        # step 3: run s1 in fcfs
        self.logger.info("FCFS")

        total_jobs = app_num
        jobID = -1

        while True:
            jobID = jobID + 1

            #-----------------------------------------
            # schedule the workload to the target GPU
            #-----------------------------------------
            appName = app_s1[jobID] # get job from s1
            appDir  = app2dir_dd[appName]

            #print appName, appDir

            process = mp.Process(target=self.dispatch_wo_client,
                                 args=(jobID, appName, appDir, AppStat, GpuStat))

            #process.daemon = True 
            process.daemon = False
            process.start()
            
            process.join()  # make sure the last process ends

            break

            ##------------------------------------------------------------------
            ## Check the timing trace for all the GPU jobs
            ##------------------------------------------------------------------
            #if jobID == total_jobs - 1:  # jobID starts from 0
            #    process.join()  # make sure the last process ends

            #    self.logger.debug("\n\nWait 1 seconds before ending.\n\n")
            #    time.sleep(1)

            #    self.logger.debug("\n\nWaiting for jobs to end. Grab a coffee if you like.\n\n")
            #    AllFinish = False
            #    while not AllFinish:
            #        with self.lock:
            #            GpuJobs_dict = dict(GpuJobs_dd)
            #            activeJobNum = 0
            #            for key, value in GpuJobs_dict.iteritems():
            #                activeJobNum = activeJobNum + value
            #            if activeJobNum ==0:
            #                AllFinish = True

            #    self.logger.debug("\n\nEnd Simulation\n\n")

            #    if maxJobs < total_jobs:
            #        self.logger.debug(
            #            "\n\nError! The total_jobs exceeds the limit!\n\n")

            #    self.PrintGpuJobTable(GpuJobTable, total_jobs)


    #--------------------------------------------------------------------------
    # server start
    #--------------------------------------------------------------------------
    def start(self):
        print("GPUs={} RunningClients={}\n".format(self.gpuNum, self.clientMode))

        #----------------------------------------------------------------------
        # 1) application status table : 5 columns
        #----------------------------------------------------------------------
        #
        #    jobid      gpu     status      starT       endT
        #       0       0           1       1           2
        #       1       1           1       1.3         2.4
        #       2       0           0       -           -
        #       ...
        #----------------------------------------------------------------------
        maxJobs = 10000
        rows, cols = maxJobs, 5  # note: init with a large prefixed table
        d_arr = mp.Array(ctypes.c_double, rows * cols)
        arr = np.frombuffer(d_arr.get_obj())
        AppStat = arr.reshape((rows, cols))

        #----------------------------------------------------------------------
        # 2) gpu node status: 1 columns
        #----------------------------------------------------------------------
        #
        #    GPU_Node(rows)     ActiveJobs
        #       0               0
        #       1               0
        #       2               0
        #       ...
        #----------------------------------------------------------------------
        GpuStat = self.manager.dict()
        for i in xrange(self.gpuNum):
            GpuStat[i] = 0



        #
        # running
        #

        if self.clientMode == 0:
            self.simu_dedicate(AppStat, GpuStat)


        ## read app2metric_dd, app2dir_dd, app2cmd_dd
        #global app2metric
        #global app2trace
        #global app2dinnfeats

        ##dpModel = None

        #if args.scheme == "rr":
        #    self.logger.info("Round-Robin Scheduling")

        #if args.scheme in ["ll", "lldelay"]:
        #    self.logger.info("Least loaded Scheduling")

        #if args.scheme in ["sim", "rrSim", "llSim", "llSim1"]:
        #    self.logger.info("Scheduling based on Similarity")
        #    app2metric = np.load('./similarity/app2metric_dd.npy').item()
        #    if check_key(app2dir, app2cmd, app2metric):
        #        self.logger.info("Looks good!")

        #if args.scheme in ["perf", "perf1", "perf2", "rrPerf", "llPerf"]:
        #    self.logger.info("Scheduling using Performance Model")
        #    app2trace = np.load('./perfmodel/app2trace_dd.npy').item()
        #    self.logger.info("Total GPU applications = %r",len(app2trace))
        #    #print app2trace['cudasdk_MCEstimatePiP']

        #if args.scheme == "simp":
        #    self.logger.info("Scheduling based on SIMP (Similarity +  Performance Model)")
        #    app2metric = np.load('./similarity/app2metric_dd.npy').item()
        #    app2trace = np.load('./perfmodel/app2trace_dd.npy').item()



        #if args.scheme == "dinn":
        #    #### Warning: use 0.2 gpu memory. Overuse will impact the coming workloads. 
        #    ##reset_graph()
        #    ##gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        #    ##sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        #    ##dpModel = dinn(sess)  # init a dinn class
        #    self.logger.info("Scheduling using Deep Interference Neural Net Model (DINN) with Performance Model")
        #    self.logger.info("Loading dinn model features ...")
        #    app2dinnfeats = np.load('./dinn/app2dinnFeats_dd.npy').item()
        #    self.logger.info("Loading performance model features ...")
        #    app2trace = np.load('./perfmodel/app2trace_dd.npy').item()
        #    self.logger.info("Total GPU applications = %r",len(app2dinnfeats))


        #self.logger.debug("listening")
        #self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ## resue socket address
        #self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #self.socket.bind((self.hostname, self.port))
        #self.socket.listen(1)



        ##----------------------------------------------------------------------
        ## 3) gpu node metrics
        ##
        ## for each gpu, allocate 32 (max jobs per gpu) x 26 (metrics for earch
        ## jobs)
        ##----------------------------------------------------------------------
        #GpuMetric_dd = self.manager.dict()

        ##GpuMetric_dd[0] = 'hello'

        #for i in xrange(self.gpuNum):
        #    GpuMetric_dd[i] = np.zeros((JobsPerGPU, 26))

        ###print GpuMetric_dd[0]
        ###GpuMetric_dd[0] = np.ones(26) 
        ###print "\n updated"
        ###print GpuMetric_dd[0]

        ##----------------------------------------------------------------------
        ## 4) gpu node metrics status
        ##
        ## for each gpu, allocate 32 (max jobs per gpu) x 2 ( status + jobID )
        ##----------------------------------------------------------------------
        #GpuMetricStat_dd = self.manager.dict()

        #for i in xrange(self.gpuNum):
        #    np_array = np.zeros((JobsPerGPU, 2))
        #    np_array[:,1] = -1  # the 2nd col for jobID is init with -1
        #    GpuMetricStat_dd[i] = np_array

        ####----------------------------------------------------------------------
        #### 5) gpu node trace status
        ####
        #### // for each gpu, allocate 32 (rows / jobs) x 3 ( status + jobID + start_time)
        #### for each gpu, allocate 1 job : 3 columns = status + jobID 
        #### since we only use the latest trace to model the performance impact
        ####----------------------------------------------------------------------
        ###GpuTracesStat_dd = self.manager.dict()

        ###for i in xrange(self.gpuNum):
        ###    #gpu_trace_list = []
        ###    #for j in xrange(32):
        ###    #    gpu_trace_list.append([0,0,0])

        ###    #print len(gpu_trace_list)
        ###    #print len(gpu_trace_list[0])
        ###    #GpuTracesStat_dd[i] = gpu_trace_list 

        ##----------------------------------------------------------------------
        ## 6) gpu node traces 
        ##
        ## // for each gpu, allocate 32 (max jobs per gpu) x [] 
        ## // jobs)
        ##----------------------------------------------------------------------
        #GpuTraces_dd = self.manager.dict()
        #for i in xrange(self.gpuNum):
        #    GpuTraces_dd[i] = [] 

        ##----------------------------------------------------------------------
        ## 7) gpu node dinn feats 
        ##----------------------------------------------------------------------
        #GpuDinnFeats_dd = self.manager.dict()
        #for i in xrange(self.gpuNum):
        #    GpuDinnFeats_dd[i] = [] 

        ##----------------------------------------------------------------------
        ## 8) node to run tf 
        ##----------------------------------------------------------------------
        ##gpu_to_run = Value('i', 6) 
        #gpu_to_run = Value('i', 8) 

        ##----------------------------------------------------------------------
        ## 9) processes to run dinn 
        ##   NOTE: max 4 dinns
        ##----------------------------------------------------------------------
        #dinn_num = Value('i', 1) 

        ###print gpu_to_run.value
        ###print type(gpu_to_run.value)

        ## print len(GpuMetric_dd)
        ## print GpuMetric_dd[0].shape

        ## for key, value in dict(GpuJobs_dd).iteritems():
        ## print key, value

        ##self.logger.debug("%r ", type(gpuTable))
        ##self.logger.debug("%r ", gpuTable.dtype)
        ##self.logger.debug("%r ", gpuTable[:])

        #total_jobs = int(args.jobs) # Note: Flag to terminate simulation

        #jobID = -1

        ##----------------------------------------------------------------------
        ## keep listening to the clients
        ##----------------------------------------------------------------------
        #while True:
        #    #target_gpu = 0

        #    conn, address = self.socket.accept()

        #    jobID = jobID + 1
        #    #self.logger.debug("Got connection : %r at %r ( job %r )", conn, address, jobID)
        #    self.logger.debug("Got connection : %r ( job %r )", address, jobID)

        #    #-----------------------------------------
        #    # schedule the workload to the target GPU
        #    #-----------------------------------------
        #    process = mp.Process(target=self.handleWorkload,
        #                         args=(conn, address, jobID, 
        #                             GpuJobTable, GpuJobs_dd,
        #                             GpuMetric_dd, GpuMetricStat_dd,
        #                             GpuTraces_dd, GpuDinnFeats_dd,gpu_to_run, dinn_num))

        #    process.daemon = False
        #    process.start()
        #    self.logger.debug("Started process %r", process)

        #    #------------------------------------------------------------------
        #    # Check the timing trace for all the GPU jobs
        #    #------------------------------------------------------------------
        #    if jobID == total_jobs - 1:  # jobID starts from 0
        #        process.join()  # make sure the last process ends

        #        self.logger.debug("\n\nWait 1 seconds before ending.\n\n")
        #        time.sleep(1)

        #        self.logger.debug("\n\nWaiting for jobs to end. Grab a coffee if you like.\n\n")
        #        AllFinish = False
        #        while not AllFinish:
        #            with self.lock:
        #                GpuJobs_dict = dict(GpuJobs_dd)
        #                activeJobNum = 0
        #                for key, value in GpuJobs_dict.iteritems():
        #                    activeJobNum = activeJobNum + value
        #                if activeJobNum ==0:
        #                    AllFinish = True

        #        self.logger.debug("\n\nEnd Simulation\n\n")

        #        if maxJobs < total_jobs:
        #            self.logger.debug(
        #                "\n\nError! The total_jobs exceeds the limit!\n\n")

        #        self.PrintGpuJobTable(GpuJobTable, total_jobs)



if __name__ == "__main__":
    print "\n#-------------------------------------------------------------------------#" \
        + "\n#  Machine-learning based Interference-aware Scheduler for GPU workloads  #" \
        + "\n#               Copyright (c) 2018 Leiming Yu <ylm@ece.neu.edu>           #" \
        + "\n#                                                                         #" \
        + "\n#         Department of ECE, Northeastern University, Boston, MA, USA     #" \
        + "\n#-------------------------------------------------------------------------#" \
        + "\n"

    if int(args.gpus) < 1 :
        logging.info("GPUs >= 1 (-g gpuNum). (Existing!)")
        sys.exit(1)

    if int(args.clientMode) not in xrange(0, 2) :
        logging.info("Running clients or not (-c 0/1). (Existing!)")
        sys.exit(1)

    gpuNum = int(args.gpus) if int(args.gpus) > 1 else 1 
    clientMode = 1 if int(args.clientMode) == 1 else 0 

    Magic = Server("0.0.0.0", 9000, gpuNum, clientMode)

    try:
        logging.info("Listening")
        Magic.start()
    except BaseException:
        logging.exception("Unexpected exception")
    finally:
        logging.info("Shutting down")
        for process in mp.active_children():
            logging.info("Shutting down process %r", process)
            process.terminate()
            process.join()
    logging.info("All done")
