#!/bin/bash

mkdir result
mkdir visualization

export PYTHONPATH=.

echo 'Start Clustering'
TYPES=( "add" "cat" )
WEIGHTS=( "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" )
ITERATION=( "0" "1" "2" "3" "4" )
CLASSNUM=( "23" "34" )
CONVTYPES=( "se" "re" )

for n in ${CLASSNUM[@]}; do
    echo "rico_${n} clustering start"
    python clustering.py -d rico_${n} -k ${CLASSNUM}
    for i in ${ITERATION[@]}; do
        echo "seq2seq_${n}_${i} clustering start"
        python clustering.py -d seq2seq_${n}_${i} -k ${CLASSNUM}
        for CONVTYPE in ${CONVTYPES[@]}; do
            echo "conv_${CONVTYPE}_${n}_${i} clustering start"
            python clustering.py -d conv_${CONVTYPE}_${n}_${i} -k ${CLASSNUM}
        done
        for TYPE in ${TYPES[@]}; do
            echo "rico_seq2seq_${n}_${i}_${TYPE} clustering start"
            python clustering.py -d rico_${n}_seq2seq_${n}_${i} -t ${TYPE} -k ${CLASSNUM}
        done
        for TYPE in ${TYPES[@]}; do
            echo "conv_re_conv_se_${n}_${i}_${TYPE} clustering start"
            python clustering.py -d conv_re_${n}_${i}_conv_se_${n}_${i} -t ${TYPE} -k ${CLASSNUM}
        done
        for CONVTYPE in ${CONVTYPES[@]}; do
            for TYPE in ${TYPES[@]}; do
                echo "rico_conv_${CONVTYPE}_${n}_${i}_${TYPE} clustering start"
                python clustering.py -d rico_${n}_conv_${CONVTYPE}_${n}_${i} -t ${TYPE} -k ${CLASSNUM}
            done
            for TYPE in ${TYPES[@]}; do
                echo "seq2seq_conv_${CONVTYPE}_${n}_${i}_${TYPE} clustering start"
                python clustering.py -d seq2seq_${n}_${i}_conv_${CONVTYPE}_${n}_${i} -t ${TYPE} -k ${CLASSNUM}
            done
        done
        for TYPE in ${TYPES[@]}; do
            for WEIGHT in ${WEIGHTS[@]}; do
                echo "rico_seq2seq_${n}_${i}_${TYPE}_${WEIGHT} clustering start"
                python clustering.py -d rico_${n}_seq2seq_${n}_${i} -t ${TYPE} -w ${WEIGHT} -k ${CLASSNUM}
            done
        done
        for TYPE in ${TYPES[@]}; do
            for WEIGHT in ${WEIGHTS[@]}; do
                echo "conv_re_conv_se_${n}_${i}_${TYPE}_${WEIGHT} clustering start"
                python clustering.py -d conv_re_${n}_${i}_conv_se_${n}_${i} -t ${TYPE} -w ${WEIGHT} -k ${CLASSNUM}
            done
        done
        for CONVTYPE in ${CONVTYPES[@]}; do
            for TYPE in ${TYPES[@]}; do
                for WEIGHT in ${WEIGHTS[@]}; do
                    echo "rico_conv_${CONVTYPE}_${n}_${i}_${TYPE}_${WEIGHT} clustering start"
                    python clustering.py -d rico_${n}_conv_${CONVTYPE}_${n}_${i} -t ${TYPE} -w ${WEIGHT} -k ${CLASSNUM}
                done
            done
            for TYPE in ${TYPES[@]}; do
                for WEIGHT in ${WEIGHTS[@]}; do
                    echo "seq2seq_conv_${CONVTYPE}_${n}_${i}_${TYPE}_${WEIGHT} clustering start"
                    python clustering.py -d seq2seq_${n}_${i}_conv_${CONVTYPE}_${n}_${i} -t ${TYPE} -w ${WEIGHT} -k ${CLASSNUM}
                done
            done
        done
    done
done

echo 'Done...'
