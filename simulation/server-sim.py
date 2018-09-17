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

    logger.debug("jodID:{} \t start: {}\t end: {}\t duration: {}".format(jobID, 
        startT, endT, endT - startT))


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
def FindNextJob(active_job_list, app2app_dist, waiting_list):
    if len(active_job_list) == 0:
        return waiting_list[0]


    if len(active_job_list) == 1:
        job_name = active_job_list[0]

        #--------------------------# 
        # run similarity analysis
        #--------------------------# 
        dist_dd = app2app_dist[job_name] # get the distance dict
        dist_sorted = sorted(dist_dd.items(), key=operator.itemgetter(1))

        leastsim_app = None
        # the sorted in non-decreasing order, use reversed()
        for appname_and_dist in reversed(dist_sorted):
            sel_appname = appname_and_dist[0]
            if sel_appname in waiting_list: # find 1st app in the list, and exit
                leastsim_app = sel_appname
                break
        return leastsim_app


    if len(active_job_list) > 1:
        print "FindNextJob(): TODO multiple activejobs! "
        sys.exit(1)



#=============================================================================#
# main program
#=============================================================================#
def main():

    MAXCORUN = 2    # max jobs per gpu
    gpuNum = 1

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

    #--------------------------------------------------------------------------
    # 2) input: app, app2dir_dd in app_info.py
    #--------------------------------------------------------------------------
    if len(app) <> len(app2dir_dd):
        print "Error: app number wrong, check ../prepare/app_info.py!"
        sys.exit(1)

    # three random sequences
    app_s1 = genRandSeq(app, seed=31415926) # pi
    #app_s2 = genRandSeq(app, seed=161803398875) # golden ratio
    #app_s3 = genRandSeq(app, seed=299792458) # speed of light

    apps_num = len(app)
    logger.debug("Total GPU Applications = {}.".format(apps_num))

    #--------------------------------------------------------------------------
    # 3) app2metric dd 
    #--------------------------------------------------------------------------
    app2metric = np.load('../prepare/app2metric_featAll.npy').item()  # featAll
    logger.debug("app2metric = {}.".format(len(app2metric)))

    #--------------------------------------------------------------------------
    # 4) compute pairwise dist 
    #--------------------------------------------------------------------------
    logger.debug("Compute Euclidean dist between apps.")

    app2app_dist = {}
    for app1, metric1 in app2metric.iteritems():
        curApp_dist = {}
        m1 = metric1.as_matrix()
        for app2, metric2 in app2metric.iteritems():
            if app1 <> app2:
                m2 = metric2.as_matrix()
                curApp_dist[app2] = np.linalg.norm(m1 - m2) 

        app2app_dist[app1] = curApp_dist

    logger.debug("Finish computing distance.")

    #--------------------------------------------------------------------------
    # 5) Prepare dispatching 
    #--------------------------------------------------------------------------
    appQueList = copy.deepcopy(app_s1)
    wait_queue = copy.deepcopy(app_s1)

    name2indx_dd = {}
    indx2name_dd = {}
    for i, app_name in enumerate(appQueList):
        name2indx_dd[app_name] = i
        indx2name_dd[i] = app_name 

    #print appQueList[:3]
    #print indx2name_dd[0], indx2name_dd[1], indx2name_dd[2]

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

    process = Process(target=run_work, args=(jobID, AppStat, app2dir_dd[appName]))
    process.daemon = False
    workers.append(process)
    process.start()

    
    # continue dispatching other jobs in the queue
    for i in xrange(1, apps_num):
        Dispatch = True if activeJobs < MAXCORUN else False

        if Dispatch:
            # find the least similar job for corunning
            leastsim_app = FindNextJob(active_job_list, app2app_dist, wait_queue)

            if leastsim_app is None:
                logger.debug("[Warning] leastsim_app is None!")
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
                jobid2name[jobID] =leastsim_app 

                process = Process(target=run_work, args=(jobID, AppStat, app2dir_dd[leastsim_app]))
                process.daemon = False
                workers.append(process)
                process.start()

        else:
            # the active jobs reach limit, wait
            while True:
                # spin
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
                leastsim_app = FindNextJob(active_job_list, app2app_dist, wait_queue)

            activeJobs += 1
            jobID += 1

            active_job_list.append(leastsim_app)
            leastsim_idx = wait_queue.index(leastsim_app) # del app from list
            del wait_queue[leastsim_idx]
            name2jobid[leastsim_app] = jobID # update name to jobID
            jobid2name[jobID] = leastsim_app 

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
    PrintGpuJobTable(AppStat, total_jobs)

    if total_jobs <> apps_num:
        logger.debug("[Warning] job number doesn't match.")




if __name__ == "__main__":
    main()
