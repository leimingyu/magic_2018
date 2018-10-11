#!/usr/bin/env python

import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#
# heatmap color: https://edwards.sdsu.edu/research/python-dataviz-seaborn-heatmap-palettes/
#

def main(args):
    if len(args) <> 4:
        print("{} <gpu_num> <csv_file> <output_heatmap_name>".format(args[0]))
        sys.exit(1)

    gpu_num = int(args[1])
    input_csv = args[2]
    output_file_name = args[3]

    df_trace = pd.read_csv(input_csv,  engine='python')

    sampling_records = int(df_trace.shape[0] / float(gpu_num))

    util_array = np.zeros((gpu_num, sampling_records))

    for index, row in df_trace.iterrows():
        gpu_util = float(row[1][:-2]) * 0.01 # process the utilization
        col_id =  int(index) / gpu_num 
        row_id =  int(index) % gpu_num
        util_array[row_id, col_id] = gpu_util

    #
    # plot heatmap using seaborn
    #

    #fig, ax = plt.subplots()
    #sns.set()
    #ax = sns.heatmap(util_array)
    #fig.savefig(output_file_name)

    sns.set()
    ax = sns.heatmap(util_array)
    fig = ax.get_figure()
    fig.savefig(output_file_name)


if __name__ == "__main__":
    main(sys.argv)


