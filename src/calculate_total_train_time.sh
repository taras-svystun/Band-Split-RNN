#!/bin/bash

total_seconds=0

for file in $(find logs/bandsplitrnn/ -name 'train.log')
do
    start_time=$(head -n 1 $file | awk -F'[][]' '{print $2}')
    end_time=$(tail -n 1 $file | awk -F'[][]' '{print $2}')
    
    start_seconds=$(date --date="$start_time" +%s)
    end_seconds=$(date --date="$end_time" +%s)
    
    diff_seconds=$((end_seconds-start_seconds))
    total_seconds=$((total_seconds+diff_seconds))
done

total_hours=$(echo "scale=2; $total_seconds / 3600" | bc)
echo "Total time in hours: $total_hours"
