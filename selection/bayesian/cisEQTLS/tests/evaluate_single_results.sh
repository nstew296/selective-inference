#!/bin/bash

DIR=/scratch/PI/sabatti/controlled_access_data/cisEQTLS_sim_exp2
SEED=1
MODE=single

source /home/jjzhu/source_code/cis_eqtl_pipeline/.env/bin/activate 

# for GENE in {0..500}
# for GENE in {3001..3500}
for GENE in {4001..4500}
do
    CMD="python sim_hier_high_dim.py evaluate -d $DIR -s $SEED -i $GENE -t $MODE"
    $CMD
done

