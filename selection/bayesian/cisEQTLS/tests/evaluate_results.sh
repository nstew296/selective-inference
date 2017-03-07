#!/bin/bash

DIR=/scratch/PI/sabatti/controlled_access_data/cisEQTLS_sim_exp2
SEED=1

START=$1
END=$2
MODE=$3

source /home/jjzhu/source_code/cis_eqtl_pipeline/.env/bin/activate 

cd /home/jjzhu/source_code/cis_eqtl_pipeline/selective-inference/selection/bayesian/cisEQTLS/tests
CMD="python sim_hier_high_dim.py evaluate -d $DIR -s $SEED -b $START -e $END -t $MODE" 
echo $CMD

OUT=preliminary_results/${START}-${END}-${MODE}.out
ERR=preliminary_results/${START}-${END}-${MODE}.err

$CMD > $OUT 2> $ERR

