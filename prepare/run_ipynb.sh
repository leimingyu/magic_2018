#!/bin/bash
notebook_file=$1
#echo $notebook_file

file_name=`echo "$notebook_file" | cut -d'.' -f1`
#echo $file_name

py_file=$file_name".py"
#echo $py_file

# generate py file from ipynb
jupyter nbconvert --to python $notebook_file 

# run python file
ipython $py_file
