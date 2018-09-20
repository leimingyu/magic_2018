#!/usr/bin/env python

import os
import glob
import csv
import numpy as np


print  "Reading csv files to obtain app runtime."

extension = 'csv'
os.chdir('./') # check current dir
targetFiles = [i for i in glob.glob('*.{}'.format(extension))]
#print(targetFiles)


appRuntime = {}
for curfile in targetFiles:
    #print curfile
    with open(curfile) as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            appName, appRT = row[1], row[-1]
            if appName not in appRuntime:
                appRuntime[appName] = appRT  # update fresh
            else:
                if appRT < appRuntime[appName]: # exist, then cmp and select the smaller one
                    appRuntime[appName] = appRT


# save the dedicated runtime to npy
np.save('app_dedicated_rt.npy', appRuntime)

print  "Done! Check app_dedicated_rt.npy"
## load
# runtime_dd = np.load('app_dedicated_rt.npy').item()
# print type(runtime_dd)
# for k,v in runtime_dd.iteritems():
#     print k,v
