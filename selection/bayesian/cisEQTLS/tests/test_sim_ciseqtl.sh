#!/bin/bash

DIR=/scratch/PI/sabatti/test
NGENES=100
YSEED=0

source /home/jjzhu/source_code/cis_eqtl_pipeline/.env/bin/activate 

# python sim_ciseqtl.py generateX -o $DIR -g $NGENES -k 10
# python sim_ciseqtl.py generateY -o $DIR -g $NGENES -k 10 -s $YSEED

# python sim_ciseqtl.py runSimes -o $DIR -g $NGENES -k 10 -s $YSEED
python sim_ciseqtl.py evalSimes -o $DIR -s $YSEED
# 
# python sim_sel_inf_pipeline.py -d $DIR -s 0 -i 0 -g $NGENES -t single
# python sim_sel_inf_pipeline.py -d $DIR -s 0 -i 1 -g $NGENES -t single
# python sim_sel_inf_pipeline.py -d $DIR -s 0 -i 6 -g $NGENES -t single
# python sim_sel_inf_pipeline.py -d $DIR -s 0 -i 9 -g $NGENES -t single
# DIR=/scratch/PI/sabatti/controlled_access_data/cisEQTLS_sim_exp2
# for YSEED in {1..10}
# do  
#     python sim_ciseqtl.py evalSimes -o $DIR -s $YSEED
# done
