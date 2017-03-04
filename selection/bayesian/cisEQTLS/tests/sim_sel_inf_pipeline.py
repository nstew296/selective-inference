from __future__ import print_function
import argparse
import os
import sys
import numpy as np

# local import functions
from sim_ciseqtl import read_X, read_y, read_simes, read_signals, mkdir_p
from hierarchical_high_dimentional import hierarchical_inference

if __name__=="__main__":
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--indir', required=True)
    parser.add_argument('-s', '--seed', required=True)
    parser.add_argument('-i', '--geneid', required=True)
    parser.add_argument('-g', '--ngenes', required=True)
    parser.add_argument('-t', '--seltype', required=True)
    args = parser.parse_args()

    s = int(args.seed)
    i = int(args.geneid)
    sel_type = str(args.seltype)

    trial_dir = os.path.join(args.indir,"trial_"+str(s))
    res_dir = os.path.join(trial_dir,"infs_"+sel_type)
    mkdir_p(res_dir)

    X_f = os.path.join(args.indir,"X_data","X_"+str(i)+".txt")
    y_f = os.path.join(trial_dir,"y_data", "y_"+str(i)+".txt")
    sime_fname = os.path.join(trial_dir,"simes_result.txt")

    sel_res = read_simes(sime_fname)
    sel = sel_res[0][i]

    if not sel: 
        sys.stderr.write("Family "+str(i)+" was not selected by Simes\n")

    else:
        sys.stderr.write("Family "+str(i)+" was selected by Simes\n")

        X = read_X(X_f)
        y = read_y(y_f)
        index = sel_res[1][i]
        idx_order = sel_res[2][i]
        sign = sel_res[3][i]
        simes_level = sel_res[4]
        rej = sel_res[5][i]

        n_sel = np.sum(np.array(sel_res[0]))
        n_genes = int(args.ngenes)
        n_samples, n_variables = X.shape

        pgenes = 1.0 * n_sel / n_genes 
        sys.stderr.write("Read X, y and eGene selection result data\n")
        sys.stderr.write("Number of samples: "+str(n_samples)+"\n")
        sys.stderr.write("Number of variables: "+str(n_variables)+"\n")
        sys.stderr.write("Proportion of selected genes: "+str(pgenes)+"\n")
        sys.stderr.write("Uniform Sime's simes_level: "+str(simes_level)+"\n")
        sys.stderr.write("Other params: "+str(index)+", "+str(idx_order)+", "+str(sign)+"\n")
        sys.stderr.write("Writing results to: "+res_dir+"\n")
        sys.stderr.write("Selection mode: "+sel_type+"\n")
        
        result_file = os.path.join(res_dir,"sel_out_"+str(i)+".txt")
        hierarchical_inference(result_file, X, y, index, simes_level, pgenes, J=rej, t_0=idx_order, T_sign=sign, selection_method=sel_type)
