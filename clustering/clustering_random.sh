#!/bin/bash

mkdir result
mkdir visualization

export PYTHONPATH=.

echo 'Start Clustering'
TYPES=( "add" "cat" )
WEIGHTS=( "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" )
#ITERATION=( "0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24" "25" "26" "27" "28" "29" )
ITERATION=( "0" "1" "2" "3" "4" )
CLASSNUM=( "23" "34" )
CONVTYPES=( "se" "re" )

for n in ${CLASSNUM[@]}; do
    for i in ${ITERATION[@]}; do
        echo "seq2seq_${n}_${i} clustering start"
        python clustering.py -d seq2seq_${n}_${i} -k ${CLASSNUM}
        for CONVTYPE in ${CONVTYPES[@]}; do
            echo "conv_${CONVTYPE}_${n}_${i} clustering start"
            python clustering.py -d conv_${CONVTYPE}_${n}_${i} -k ${CLASSNUM}
        done
    done
done

echo 'Done...'
