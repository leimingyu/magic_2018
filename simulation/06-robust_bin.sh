#!/bin/bash

seedopt=(31415926 161803398875 299792458) # pi/golden ratio/speed of light


#scheme="robustBin_featAll"
#scheme="robustBin_feat9"
scheme="robustBin_featMystic"



#scheme="InterFCFS_featAll"
#scheme="InterFCFS_feat9"
#scheme="InterFCFS_feat12"
#scheme="InterFCFS_feat14"
#scheme="InterFCFS_feat18"
#scheme="InterFCFS_feat26"
#scheme="InterFCFS_feat42"
#scheme="InterFCFS_feat64"
#scheme="InterFCFS_featMystic"


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

		./server-robustBin.py -c $jobs -s $seed -f $outFile 

		#break
	done
	#break
done
