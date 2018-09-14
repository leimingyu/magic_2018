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




#=============================================================================#
# main program
#=============================================================================#
def main():

    MAXCORUN = 2    # max jobs per gpu
    gpuNum = 1

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
    GpuStat = manager.dict()
    for i in xrange(gpuNum):
        GpuStat[i] = 0


    #--------------------------------------------------------------------------
    # input: app, app2dir_dd in app_info.py
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
    #
    #--------------------------------------------------------------------------


    appQueList = app_s1 

    workers = [] # for mp processes


    #==================================#
    # run the apps in the queue 
    #==================================#
    activeJobs = 0
    jobID = -1

    current_jobid_list = [] # keep track of current application 

    for i in xrange(apps_num):
        Dispatch = False 

        if activeJobs < MAXCORUN:  # NOTE: assuming only one gpu is used
            Dispatch = True

        #print("iter {} dispatch={}".format(i, Dispatch))

        if Dispatch:
            activeJobs += 1
            jobID += 1
            current_jobid_list.append(jobID)

            appName = appQueList[i] 
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
    PrintGpuJobTable(AppStat, total_jobs)

    if total_jobs <> apps_num:
        logger.debug("[Warning] job number doesn't match.")


if __name__ == "__main__":
    main()
