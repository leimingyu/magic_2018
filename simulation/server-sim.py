#!/usr/bin/env python

import os,sys
import operator, copy, random, time, ctypes
import numpy as np

import multiprocessing as mp
from multiprocessing import Process, Lock, Manager, Value, Pool

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
parser.add_argument('-c', dest='maxCoRun', default=2,        help='max collocated jobs per gpu')
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


#-----------------------------------------------------------------------------#
# GPU Job Table 
#-----------------------------------------------------------------------------#
def FindNextJob(active_job_list, app2app_dist, waiting_list, app2metric):
    if len(active_job_list) == 0:
        return waiting_list[0]

    leastsim_app = None

    #--------------------------------------------------#
    # 
    #--------------------------------------------------#
    if len(active_job_list) == 1:
        job_name = active_job_list[0]
        #--------------------------# 
        # run similarity analysis
        #--------------------------# 
        dist_dd = app2app_dist[job_name] # get the distance dict
        dist_sorted = sorted(dist_dd.items(), key=operator.itemgetter(1))

        # the sorted in non-decreasing order, use reversed()
        for appname_and_dist in reversed(dist_sorted):
            sel_appname = appname_and_dist[0]
            if sel_appname in waiting_list: # find 1st app in the list, and exit
                leastsim_app = sel_appname
                break

    #--------------------------------------------------#
    # 
    #--------------------------------------------------#
    if len(active_job_list) >= 2 :
        activeJobs = len(active_job_list)
        # obtain the metrics in the active job list, apply max() method
        active_mets = app2metric[active_job_list[0]].values

        for i in xrange(1, activeJobs):
            curr_mat = app2metric[active_job_list[i]].values
            active_mets = np.vstack((active_mets, curr_mat))
        active_mets = np.amax(active_mets, axis=0) # max value in each col

        #
        # now, I need to compute the dist to each app in the waiting list
        #
        cur_dist = {}
        for app, app_metric in app2metric.iteritems():
            if app in waiting_list:
                curr_mat = app2metric[app].values
                cur_dist[app] = np.linalg.norm(curr_mat - active_mets) # add to the dist dict

        #
        # sort the cur_dist, find the largest dist app
        #
        dist_sorted = sorted(cur_dist.items(), key=operator.itemgetter(1))
        leastsim_app = dist_sorted[-1][0] # largest dist, (appName, appDist)

    return leastsim_app


#=============================================================================#
# main program
#=============================================================================#
def main():

    MAXCORUN = int(args.maxCoRun)    # max jobs per gpu
    RANDSEED = int(args.seed)
    gpuNum = 1

    logger.debug("MaxCoRun={}\trandseed={}\tsaveFile={}".format(MAXCORUN, RANDSEED, args.ofile))

    #--------------------------------------------------------------------------
    # 1) application status table : 5 columns
    #--------------------------------------------------------------------------
    #
    #    jobid      gpu     status      starT       endT
    #       0       0           1       1           2
    #       1       1           1       1.3         2.4
    #       2       0           0       -           -
    #       ...
    #--------------------------------------------------------------------------
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
    # 3) app2metric dd 
    #--------------------------------------------------------------------------
    #app2metric = np.load('../prepare/app2metric_featAll.npy').item()  # featAll
    #app2metric = np.load('../prepare/app2metric_feat9.npy').item()     # feat9
    #app2metric = np.load('../prepare/app2metric_feat12.npy').item()     # feat12
    #app2metric = np.load('../prepare/app2metric_feat14.npy').item()     # feat14
    #app2metric = np.load('../prepare/app2metric_feat18.npy').item()     # feat18
    #app2metric = np.load('../prepare/app2metric_feat26.npy').item()     # feat26
    #app2metric = np.load('../prepare/app2metric_feat42.npy').item()     # feat42
    #app2metric = np.load('../prepare/app2metric_feat64.npy').item()     # feat64
    app2metric = np.load('../prepare/app2metric_featMystic.npy').item()     # featMystic

    logger.debug("app2metric = {}.".format(len(app2metric)))

    #
    # check the appName in app2metric  = appName in app_s1 
    #
    if apps_num <> len(app2metric):
        print "The length of input app list and app2metric does not match!\n"
        sys.exit(1)

    app2metric_key = app2metric.keys()
    app2metric_key_set = set(app2metric_key)
    applist_set = set(app)

    # to make sure for each coming app, there is always a metric vector for it
    if applist_set == app2metric_key_set:
        logger.debug("Perfect! Keep going!")
    else:
        print "Oops! The app names does not match!\n"
        sys.exit(1)


    #--------------------------------------------------------------------------
    # 4) compute pairwise dist
    #--------------------------------------------------------------------------
    # NOTE: this is useful for maxJobRun = 2
    logger.debug("Compute Euclidean dist between apps.")

    app2app_dist = {}
    for app1, metric1 in app2metric.iteritems():
        curApp_dist = {}
        m1 = metric1.values
        for app2, metric2 in app2metric.iteritems():
            if app1 <> app2:
                m2 = metric2.values
                curApp_dist[app2] = np.linalg.norm(m1 - m2) 

        app2app_dist[app1] = curApp_dist

    logger.debug("Finish computing distance.")


    #--------------------------------------------------------------------------
    # 5) Prepare dispatching 
    #--------------------------------------------------------------------------
    appQueList = copy.deepcopy(app_s1)
    wait_queue = copy.deepcopy(app_s1)

    ##name2indx_dd = {}
    ##indx2name_dd = {}
    ##for i, app_name in enumerate(appQueList):
    ##    name2indx_dd[app_name] = i
    ##    indx2name_dd[i] = app_name 

    ###print appQueList[:3]
    ###print indx2name_dd[0], indx2name_dd[1], indx2name_dd[2]

    workers = [] # for mp processes

    active_job_list = []  # keep track of current active job name

    name2jobid = {}  # use name to find out the jobID
    jobid2name = {}

    activeJobs = 0
    jobID = -1

    #--------------------------------------------------------------------------
    # 6) Start dispatching 
    #--------------------------------------------------------------------------

    # dipatch 1st app in the queue
    jobID += 1
    activeJobs += 1

    appName= wait_queue[0]
    active_job_list.append(appName)       # add app to the active job list
    app_idx = wait_queue.index(appName)   # remove the app from the waiting queue 
    del wait_queue[app_idx]
    name2jobid[appName] = jobID # update job info
    jobid2name[jobID] = appName
    id2name[jobID] = appName

    process = Process(target=run_work, args=(jobID, AppStat, app2dir_dd[appName]))
    process.daemon = False
    workers.append(process)
    process.start()

    
    # continue dispatching other jobs in the queue
    for i in xrange(1, apps_num):
        Dispatch = True if activeJobs < MAXCORUN else False

        #----------------------------------------------------------------------
        # keep dispatching 
        #----------------------------------------------------------------------
        if Dispatch:
            # find the least similar job for corunning
            leastsim_app = FindNextJob(active_job_list, app2app_dist, wait_queue, app2metric)

            if leastsim_app is None:
                logger.debug("[Error] leastsim_app is None!")
                #sys.exit(1)
            else:
                #
                # run the selected app
                #
                activeJobs += 1
                jobID += 1

                active_job_list.append(leastsim_app) # add app to the active job list
                leastsim_idx = wait_queue.index(leastsim_app) # del app from list
                del wait_queue[leastsim_idx]
                name2jobid[leastsim_app] = jobID # update name to jobID
                jobid2name[jobID] = leastsim_app
                id2name[jobID] = leastsim_app 

                process = Process(target=run_work, args=(jobID, AppStat, app2dir_dd[leastsim_app]))
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
                leastsim_app = wait_queue[0]
            else:
                leastsim_app = FindNextJob(active_job_list, app2app_dist, wait_queue, app2metric)

            activeJobs += 1
            jobID += 1

            active_job_list.append(leastsim_app)
            leastsim_idx = wait_queue.index(leastsim_app) # del app from list
            del wait_queue[leastsim_idx]
            name2jobid[leastsim_app] = jobID # update name to jobID
            jobid2name[jobID] = leastsim_app
            id2name[jobID] = leastsim_app 

            process = Process(target=run_work, args=(jobID, AppStat, app2dir_dd[leastsim_app]))
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
        PrintGpuJobTable(AppStat, total_jobs, id2name, saveFile=args.ofile)
    else:
        PrintGpuJobTable(AppStat, total_jobs, id2name)



if __name__ == "__main__":
    main()
