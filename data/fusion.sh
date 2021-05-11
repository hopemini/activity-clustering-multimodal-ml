#!/bin/bash

export PYTHONPATH=.

echo 'RICO test data extraction...'
python test_data_extraction.py -d rico

echo 'Data fusion...'
TYPES=( "add" "cat" )
WEIGHTS=( "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" )
ITERATION=( "0" "1" "2" "3" "4" )
CLASSNUM=( "23" "34" )
CONVTYPES=( "se" "re" )

for i in ${ITERATION[@]}; do
    for n in ${CLASSNUM[@]}; do
        for TYPE in ${TYPES[@]}; do
            python data_fusion.py -d1 rico_${n} -d2 seq2seq_${n}_${i} -t ${TYPE}
            echo "rico_seq2seq_${n}_${i}_${TYPE} done.."
        done
        for TYPE in ${TYPES[@]}; do
            python data_fusion.py -d1 conv_re_${n}_${i} -d2 conv_se_${n}_${i} -t ${TYPE}
            echo "conv_re_conv_se_${n}_${i}_${TYPE} done.."
        done
        for CONVTYPE in ${CONVTYPES[@]}; do
            for TYPE in ${TYPES[@]}; do
                python data_fusion.py -d1 rico_${n} -d2 conv_${CONVTYPE}_${n}_${i} -t ${TYPE}
                echo "rico_conv_${CONVTYPE}_${n}_${i}_${TYPE} done.."
            done
            for TYPE in ${TYPES[@]}; do
                python data_fusion.py -d1 seq2seq_${n}_${i} -d2 conv_${CONVTYPE}_${n}_${i} -t ${TYPE}
                echo "seq2seq_conv_${CONVTYPE}_${n}_${i}_${TYPE} done.."
            done
        done

        for TYPE in ${TYPES[@]}; do
            for WEIGHT in ${WEIGHTS[@]}; do
                python data_fusion.py -d1 rico_${n} -d2 seq2seq_${n}_${i} -t ${TYPE} -w ${WEIGHT}
                echo "rico_seq2seq_${n}_${i}_${TYPE}_${WEIGHT} done.."
            done
        done
        for TYPE in ${TYPES[@]}; do
            for WEIGHT in ${WEIGHTS[@]}; do
                python data_fusion.py -d1 conv_re_${n}_${i} -d2 conv_se_${n}_${i} -t ${TYPE} -w ${WEIGHT}
                echo "conv_re_conv_se_${n}_${i}_${TYPE}_${WEIGHT} done.."
            done
        done
        for CONVTYPE in ${CONVTYPES[@]}; do
            for TYPE in ${TYPES[@]}; do
                for WEIGHT in ${WEIGHTS[@]}; do
                    python data_fusion.py -d1 rico_${n} -d2 conv_${CONVTYPE}_${n}_${i} -t ${TYPE} -w ${WEIGHT}
                    echo "rico_conv_${CONVTYPE}_${n}_${i}_${TYPE}_${WEIGHT} done.."
                done
            done
            for TYPE in ${TYPES[@]}; do
                for WEIGHT in ${WEIGHTS[@]}; do
                    python data_fusion.py -d1 seq2seq_${n}_${i} -d2 conv_${CONVTYPE}_${n}_${i} -t ${TYPE} -w ${WEIGHT}
                    echo "seq2seq_conv_${CONVTYPE}_${n}_${i}_${TYPE}_${WEIGHT} done.."
                done
            done
        done
    done
done

echo 'Done...'
