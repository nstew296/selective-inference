#!/bin/bash

DIR=/scratch/PI/sabatti/test
NGENES=100
YSEED=0

source /home/jjzhu/source_code/cis_eqtl_pipeline/.env/bin/activate 

cd /home/jjzhu/source_code/cis_eqtl_pipeline/selective-inference/selection/bayesian/cisEQTLS/tests

# python sim_ciseqtl.py generateX -o $DIR -g $NGENES -k 10
# python sim_ciseqtl.py generateY -o $DIR -g $NGENES -k 10 -s $YSEED

# python sim_ciseqtl.py runSimes -o $DIR -g $NGENES -k 10 -s $YSEED
# python sim_ciseqtl.py evalSimes -o $DIR -s $YSEED

# python sim_sel_inf_pipeline.py -d $DIR -s 0 -i 0 -g $NGENES -t single
# python sim_sel_inf_pipeline.py -d $DIR -s 0 -i 1 -g $NGENES -t single
# python sim_sel_inf_pipeline.py -d $DIR -s 0 -i 6 -g $NGENES -t single
# python sim_sel_inf_pipeline.py -d $DIR -s 0 -i 9 -g $NGENES -t single
DIR=/scratch/PI/sabatti/controlled_access_data/cisEQTLS_sim_exp2
YSEED=1

BEG=0
END=5000
python sim_ciseqtl.py evalSimes -o $DIR -s $YSEED -b $BEG -e $END > preliminary_results/summary_${BEG}-${END}.txt

BEG=0
END=1250
python sim_ciseqtl.py evalSimes -o $DIR -s $YSEED -b $BEG -e $END > preliminary_results/summary_${BEG}-${END}.txt

BEG=1250
END=2500
python sim_ciseqtl.py evalSimes -o $DIR -s $YSEED -b $BEG -e $END > preliminary_results/summary_${BEG}-${END}.txt

BEG=2500
END=3750
python sim_ciseqtl.py evalSimes -o $DIR -s $YSEED -b $BEG -e $END > preliminary_results/summary_${BEG}-${END}.txt

BEG=3750
END=5000
python sim_ciseqtl.py evalSimes -o $DIR -s $YSEED -b $BEG -e $END > preliminary_results/summary_${BEG}-${END}.txt

