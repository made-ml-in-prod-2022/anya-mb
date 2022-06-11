#!/bin/sh
output_dir=$1

mkdir -p ${output_dir}

wget https://raw.githubusercontent.com/anya-mb/public_data/main/heart_cleveland_upload.csv -O ${output_dir}/heart_cleveland_upload.csv
