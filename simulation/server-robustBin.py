#!/usr/bin/env python

import os,sys
import operator, copy, random, time, ctypes
import numpy as np

import multiprocessing as mp
from multiprocessing import Process, Lock, Manager, Value, Pool

from sklearn.externals import joblib  # to save/load model to disk

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("[server]")

# read app info
sys.path.append('../prepare')
from app_info import * 

# read common api
sys.path.append('.')
from magic_common import * 

lock = Lock()
manager = Manager()

#-----------------------------------------------------------------------------#
# arguments 
#-----------------------------------------------------------------------------#
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('-c', dest='maxCoRun', default=1,        help='max collocated jobs per gpu')
parser.add_argument('-s', dest='seed',     default=31415926, help='random seed for shuffling the app order')
parser.add_argument('-f', dest='ofile',    default=None,     help='output file to save the app timing trace')
args = parser.parse_args()

#-----------------------------------------------------------------------------#
#
#-----------------------------------------------------------------------------#
def predict_appclass(app2metric, bestmodel):
    app2class_dd = {}
    for cur_app, metric in app2metric.iteritems():
        metric = metric.values.reshape(-1, metric.shape[0]) # col to row
        #print metric.shape
        app_class = bestmodel.predict(metric) # 1 as robust , 0 as sensitive
        # print cur_app, app_class[0]
        app2class_dd[cur_app] = app_class[0]
        #break
    return app2class_dd


#-----------------------------------------------------------------------------#
# Run incoming workload
#-----------------------------------------------------------------------------#
def run_work(jobID, AppStat, appDir):

    #
    #    jobid      gpu     status      starT       endT
    #

    AppStat[jobID, 0] = jobID 
    # avoid tagging gpu since we simulate with 1 gpu
    AppStat[jobID, 2] = 0 

    # run the application 
    [startT, endT] = run_remote(app_dir=appDir, devid=0)

    logger.debug("jodID:{0:5d} \t start: {1:.3f}\t end: {2:.3f}\t duration: {3:.3f}".format(jobID, startT, endT, endT - startT))


    #=========================#
    # update gpu job table
    #
    # 5 columns:
    #    jobid      gpu     starT       endT
    #=========================#
    # mark the job is done, and update the timing info
    AppStat[jobID, 2] = 1   # done
    AppStat[jobID, 3] = startT 
    AppStat[jobID, 4] = endT 


#=============================================================================#
# main program
#=============================================================================#
def main():

    MAXCORUN = int(args.maxCoRun)    # max jobs per gpu
    RANDSEED = int(args.seed)
    gpuNum = 1

    logger.debug("MaxCoRun={}\trandseed={}\tsaveFile={}".format(MAXCORUN, RANDSEED, args.ofile))

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

    id2name = {}

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
    #GpuStat = manager.dict()
    #for i in xrange(gpuNum):
    #    GpuStat[i] = 0


    #--------------------------------------------------------------------------
    # input: app, app2dir_dd in app_info.py
    #--------------------------------------------------------------------------
    if len(app) <> len(app2dir_dd):
        print "Error: app number wrong, check ../prepare/app_info.py!"
        sys.exit(1)

    # three random sequences
    app_s1 = genRandSeq(app, seed=RANDSEED) # pi

    apps_num = len(app)
    logger.debug("Total GPU Applications = {}.".format(apps_num))

    #--------------------------------------------------------------------------
    # 3) load  [1] app2metric and [2] bestmodel for the selected feature set  
    #--------------------------------------------------------------------------
    #app2metric = np.load('../prepare/app2metric_featAll.npy').item()  # featAll
    #bestmodel = joblib.load('../01_classification/featall_bestmodel.pkl') # load model, predict app class 

    #app2metric = np.load('../prepare/app2metric_feat9.npy').item()     # feat9
    #bestmodel = joblib.load('../01_classification/feat9_bestmodel.pkl') # load model, predict app class 

    #app2metric = np.load('../prepare/app2metric_feat12.npy').item()     # feat12
    #bestmodel = joblib.load('../01_classification/feat12_bestmodel.pkl') # load model, predict app class 

    #app2metric = np.load('../prepare/app2metric_feat14.npy').item()     # feat14
    #bestmodel = joblib.load('../01_classification/feat14_bestmodel.pkl') # load model, predict app class 

    #app2metric = np.load('../prepare/app2metric_feat18.npy').item()     # feat18
    #bestmodel = joblib.load('../01_classification/feat18_bestmodel.pkl') # load model, predict app class 

    #app2metric = np.load('../prepare/app2metric_feat26.npy').item()     # feat26
    #bestmodel = joblib.load('../01_classification/feat26_bestmodel.pkl') # load model, predict app class 

    #app2metric = np.load('../prepare/app2metric_feat42.npy').item()     # feat42
    #bestmodel = joblib.load('../01_classification/feat42_bestmodel.pkl') # load model, predict app class 

    #app2metric = np.load('../prepare/app2metric_feat64.npy').item()     # feat64
    #bestmodel = joblib.load('../01_classification/feat64_bestmodel.pkl') # load model, predict app class 

    app2metric = np.load('../prepare/app2metric_featMystic.npy').item()     # featMystic
    bestmodel = joblib.load('../01_classification/featMystic_bestmodel.pkl') # load model, predict app class 

    app2class_dd = predict_appclass(app2metric, bestmodel) 




    #--------------------------------------------------------------------------
    # according to the input app sequence, arrange them by putting robust
    # app in front of the sensitive app
    #--------------------------------------------------------------------------
    robust_list, sensitive_list = [],  []
    for i in app_s1:
        if app2class_dd[i] == 1:
            robust_list.append(i)
        else:
            sensitive_list.append(i)

    print robust_list
    print len(robust_list)

    #---------
    # read the bin size
    #---------
    app_binsize_dd = np.load('../01_classification/app_binsize_dd.npy').item()

    # for k,v in app_binsize_dd.iteritems(): print k, v
    robust_bin_dd = {}
    for ap in robust_list:
         robust_bin_dd[ap] = app_binsize_dd[ap]

    sensitive_bin_dd = {}
    for ap in sensitive_list:
         sensitive_bin_dd[ap] = app_binsize_dd[ap]


    robust_bin_sorted = sorted(robust_bin_dd.items(), key=operator.itemgetter(1), reverse=True)
    sensitive_bin_sorted = sorted(sensitive_bin_dd.items(), key=operator.itemgetter(1), reverse=True) # big to small

    #print robust_bin_sorted

    new_seq = []
    for (ap, _) in robust_bin_sorted: # add robust first
        new_seq.append(ap)
    for (ap, _) in sensitive_bin_sorted: # add sensitive 
        new_seq.append(ap)

    #print new_seq


    #return

    #print sensitive_list
    #print len(sensitive_list)

    #new_seq = copy.deepcopy(robust_list)
    #new_seq.extend(sensitive_list)

    #print new_seq



    #--------------------------------------------------------------------------
    #
    #--------------------------------------------------------------------------
    #appQueList = app_s1 
    appQueList = new_seq 

    workers = [] # for mp processes


    #==================================#
    # run the apps in the queue 
    #==================================#
    activeJobs = 0
    jobID = -1

    current_jobid_list = [] # keep track of current application 

    for i in xrange(apps_num):
        Dispatch = True if activeJobs < MAXCORUN else False 
        #print("iter {} dispatch={}".format(i, Dispatch))

        if Dispatch:
            activeJobs += 1
            jobID += 1
            current_jobid_list.append(jobID)

            appName = appQueList[i] 
            id2name[jobID] = appName
            process = Process(target=run_work, args=(jobID, AppStat, app2dir_dd[appName]))

            process.daemon = False
            #logger.debug("Start %r", process)
            workers.append(process)
            process.start()

        else:
            # spin
            while True:
                break_loop = False

                current_running_jobs = 0
                jobs2del = []

                for jid in current_jobid_list:
                    if AppStat[jid, 2] == 1: # check the status, if one is done
                        jobs2del.append(jid)
                        break_loop = True

                if break_loop:
                    activeJobs -= 1

                    # update
                    if jobs2del:
                        for id2del in jobs2del:
                            del_idx = current_jobid_list.index(id2del)
                            del current_jobid_list[del_idx]
                    break

            #------------------------------------
            # after spinning, schedule the work
            #------------------------------------

            #print("iter {}: activeJobs = {}".format(i, activeJobs))
            activeJobs += 1
            jobID += 1
            current_jobid_list.append(jobID)
            #print("iter {}: activeJobs = {}".format(i, activeJobs))

            appName = appQueList[i] 
            id2name[jobID] = appName
            process = Process(target=run_work, args=(jobID, AppStat, app2dir_dd[appName]))

            process.daemon = False
            workers.append(process)
            process.start()


    #=========================================================================#
    # end of running all the jobs
    #=========================================================================#
    for p in workers:
        p.join()

    total_jobs = jobID + 1
    if total_jobs <> apps_num:
        logger.debug("[Warning] job number doesn't match.")

    
    # print out / save trace table 
    if args.ofile:
        PrintGpuJobTable(AppStat, total_jobs, id2name, saveFile=args.ofile)
    else:
        PrintGpuJobTable(AppStat, total_jobs, id2name)


if __name__ == "__main__":
    main()
