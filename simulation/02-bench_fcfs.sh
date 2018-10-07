#!/bin/bash

seedopt=(31415926 161803398875 299792458) # pi/golden ratio/speed of light

scheme="fcfs"


for (( jobs=2; jobs<=16; jobs++ ))
do
	#echo $jobs

	#----------------------------------------------
	# for each maxjobs per gpu, run different seeds
	#----------------------------------------------
	s_idx=0
	for seed in ${seedopt[@]}; do
		s_idx=$((s_idx+1))

		outFile=$scheme"_run"$jobs"_s"$s_idx".csv"

		#echo $jobs
		#echo $seed
		echo $outFile

		./server-fcfs.py -c $jobs -s $seed -f $outFile 

		#break
	done
	#break
done

#for seed in ${seedopt[@]}; do
#	s_idx=$((s_idx+1))
#	echo $s_idx
#	echo $seed
#	#outFile=$scheme"_run6_s$s_idx.csv"
#	outFile=$scheme"_run6_s"$s_idx".csv"
#	echo $outFile
#
#	#./server-fcfs.py -c 6 -s $seed -f "fcfs_run6_s1_test1.csv" 
#	break
#done

#./server-fcfs.py -c 1                  # default: 31415926
#./server-fcfs.py -c 1  -s 161803398875 # golden ratio
#./server-fcfs.py -c 1  -s 299792458    # speed of light
