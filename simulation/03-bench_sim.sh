#!/bin/bash

seedopt=(31415926 161803398875 299792458) # pi/golden ratio/speed of light

#scheme="sim_featAll"
#scheme="sim_feat9"
#scheme="sim_feat12"
#scheme="sim_feat14"
#scheme="sim_feat18"
#scheme="sim_feat26"
#scheme="sim_feat42"
#scheme="sim_feat64"
scheme="sim_featMystic"

#./server-sim.py -c 6  -s 161803398875 -f "test.csv" # golden ratio

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

		./server-sim.py -c $jobs -s $seed -f $outFile 

		#break
	done
	#break
done
