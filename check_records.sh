#!/bin/bash
while IFS='' read -r line || [[ -n "$line"  ]]; do
    if [ -e ~/mimic3-benchmarks/data/root/train/$line ]
    then
        echo "$line train"
    elif [ -e ~/mimic3-benchmarks/data/root/test/$line ]
    then
        echo "$line test"
    else
        echo "$line absent"
    fi
done < "$1"
