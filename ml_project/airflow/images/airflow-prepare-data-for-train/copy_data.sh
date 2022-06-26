#!/bin/sh
input_dir=$1
output_dir=$2

mkdir -p ${output_dir}

cp ${input_dir}/*.csv ${output_dir}/train_data.csv
