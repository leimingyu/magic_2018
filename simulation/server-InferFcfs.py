#!/usr/bin/env python

import os,sys
import operator, copy, random, time, ctypes
import numpy as np

import multiprocessing as mp
from multiprocessing import Process, Lock, Manager, Value, Pool

from sklearn.externals import joblib  # to save/load model to disk
#import pickle 

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
#
#-----------------------------------------------------------------------------#
def FindJob_InferFCFS(wait_queue, app_class, active_job_list, app2class_dd):

    # if active_job_list is empty, select robust then fcfs
    # if no robust in the active_job_list, select robust then fcfs

    target_app = None
    target_app_idx = None

    robust_jobs = sum([app2class_dd[j] for j in active_job_list])
    if (not active_job_list) or robust_jobs == 0:
        has_robust = False

        idx = -1 
        for ap, cls in zip(wait_queue, app_class):
            idx += 1
            if cls == 1: # robust 
                has_robust = True
                target_app = ap
                target_app_idx = idx
                break

        # if there is no robust app, pick the 1st one in the queue
        if not has_robust:
            target_app = wait_queue[0]
            target_app_idx = 0

        return target_app, target_app_idx

    # if robust in the active_job_list, select fcfs
    if robust_jobs >= 1:
        target_app = wait_queue[0]
        target_app_idx = 0
        return target_app, target_app_idx


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
# arguments 
#-----------------------------------------------------------------------------#
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('-c', dest='maxCoRun', default=1,        help='max collocated jobs per gpu')
parser.add_argument('-s', dest='seed',     default=31415926, help='random seed for shuffling the app order')
parser.add_argument('-f', dest='ofile',    default=None,     help='output file to save the app timing trace')
args = parser.parse_args()


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


    #--------------------------------------------------------------------------
    # 2) input: app, app2dir_dd in app_info.py
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
    app2metric = np.load('../prepare/app2metric_featAll.npy').item()  # featAll
    bestmodel = joblib.load('../01_classification/featall_bestmodel.pkl') # load model, predict app class 
    #bestmodel = pickle.load(open('../01_classification/featall_bestmodel.pkl', 'rb'))

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

    #app2metric = np.load('../prepare/app2metric_featMystic.npy').item()     # featMystic
    #bestmodel = joblib.load('../01_classification/featMystic_bestmodel.pkl') # load model, predict app class 

    app2class_dd = predict_appclass(app2metric, bestmodel) 
    app_s1_class = [app2class_dd[k] for k in app_s1] # list of class for each app_s1

    logger.debug("app2metric = {}.".format(len(app2metric)))

    #--------------------------------------------------------------------------
    # 4) prepare 
    #--------------------------------------------------------------------------
    appQueList = copy.deepcopy(app_s1)
    wait_queue = copy.deepcopy(app_s1)
    app_class  = copy.deepcopy(app_s1_class)

    workers = [] # for mp processes
    active_job_list = []  # keep track of current active job name

    name2jobid = {}  # use name to find out the jobID
    jobid2name = {}

    activeJobs = 0
    jobID = -1

    #--------------------------------------------------------------------------
    # 5) Start dispatching 
    #--------------------------------------------------------------------------

    jobID += 1
    activeJobs += 1
    # find 1st robust app to run, if not, use the 1st app in the queue
    appName, app_idx = FindJob_InferFCFS(wait_queue, app_class, active_job_list, app2class_dd)
    active_job_list.append(appName)       # add app to the active job list
    del wait_queue[app_idx], app_class[app_idx] # remove the app from the waiting queue 
    name2jobid[appName] = jobID # update job info
    jobid2name[jobID] = appName

    process = Process(target=run_work, args=(jobID, AppStat, app2dir_dd[appName]))
    process.daemon = False
    workers.append(process)
    process.start()

    #--------------------------------------------------------------------------
    # 6) continue 
    #--------------------------------------------------------------------------
    for i in xrange(1, apps_num):
        Dispatch = True if activeJobs < MAXCORUN else False

        #----------------------------------------------------------------------
        # keep dispatching 
        #----------------------------------------------------------------------
        if Dispatch:
            # find the next job for corunning
            appName, app_idx = FindJob_InferFCFS(wait_queue, app_class, active_job_list, app2class_dd)

            if appName is None:
                logger.debug("[Error] appName is None!")
                #sys.exit(1)
            else:
                #
                # run the selected app
                #
                activeJobs += 1
                jobID += 1

                active_job_list.append(appName) # add app to the active job list
                del wait_queue[app_idx], app_class[app_idx] # remove the app from the waiting queue 
                name2jobid[appName] = jobID # update name to jobID
                jobid2name[jobID] = appName 

                process = Process(target=run_work, args=(jobID, AppStat, app2dir_dd[appName]))
                process.daemon = False
                workers.append(process)
                process.start()

        #----------------------------------------------------------------------
        # when reaching the limit
        #----------------------------------------------------------------------
        else:
            # the active jobs reach limit, wait
            # spin
            while True:
                break_loop = False

                current_running_jobs = 0
                jobs2del = []

                for jobname in active_job_list:
                    jid = name2jobid[jobname]
                    if AppStat[jid, 2] == 1: # check the status, if one is done
                        jobs2del.append(jid)  # add the jobID
                        break_loop = True

                if break_loop:
                    # remove job which has ended 
                    for job_id in jobs2del:
                        activeJobs -= 1
                        appname = jobid2name[job_id]
                        del_idx = active_job_list.index(appname)
                        del active_job_list[del_idx]

                    break # stop spinning, exit while loop
                
            #
            # after spinning, schedule the work
            #

            # for the last application, go directly schedule it
            if i == (apps_num - 1):
                nextapp = wait_queue[0]
                nextapp_idx = 0
            else:
                nextapp, nextapp_idx = FindJob_InferFCFS(wait_queue, app_class, active_job_list, app2class_dd)

            activeJobs += 1
            jobID += 1

            active_job_list.append(nextapp)
            del wait_queue[nextapp_idx], app_class[nextapp_idx] # remove the app from the waiting queue 
            name2jobid[nextapp] = jobID # update name to jobID
            jobid2name[jobID] = nextapp 

            process = Process(target=run_work, args=(jobID, AppStat, app2dir_dd[nextapp]))
            process.daemon = False
            workers.append(process)
            process.start()
            
    #--------------------------------------------------------------------------
    # end of for loop  : finish dispatching all the jobs
    #--------------------------------------------------------------------------
    for p in workers:
        p.join()

    total_jobs = jobID + 1
    if total_jobs <> apps_num:
        logger.debug("[Warning] job number doesn't match.")

    # print out / save trace table 
    if args.ofile:
        PrintGpuJobTable(AppStat, total_jobs, jobid2name, saveFile=args.ofile)
    else:
        PrintGpuJobTable(AppStat, total_jobs, jobid2name)


if __name__ == "__main__":
    main()
