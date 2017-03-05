from __future__ import print_function
import argparse
import os
import sys
import numpy as np

# local import functions
from sim_ciseqtl import read_X, read_y, read_simes, read_signals, mkdir_p
from hierarchical_high_dimentional import hierarchical_inference

def evaluate_hierarchical_results(result, X, s, snr):
    """
    evaluate the output for hierarhical with ground truth information
    """
    # Inputs:
    #   result: np matrix of results (rows: genes, 
    #           cols: [lasso_sel, ***, unadj_l, unadj_r, adj_l, adj_r, **, **]) 
    #   X: np matrix of the data input
    #   s: integer of value of the signal (the first s variables are signals)
    #   snr: float of value of the snr 
    # Outputs:
    #   None

    FDR = 0.
    power = 0.

    n, p = X.shape
    true_beta = np.zeros(p)
    true_beta[:s] = snr

    discoveries = np.array(result[:, 1], np.bool)

    if true_beta[0] > 0:

        true_discoveries = discoveries[:s].sum()

    else:
        true_discoveries = 0

    false_discoveries = discoveries[s:].sum()
    FDR += false_discoveries / max(float(discoveries.sum()), 1.)
    power += true_discoveries / float(s)

    active_ind = np.array(result[:, 0], np.bool)
    nactive = active_ind.sum()

    projection_active = X[:, active_ind].dot(np.linalg.inv(X[:, active_ind].T.dot(X[:, active_ind])))
    true_val = projection_active.T.dot(X.dot(true_beta))

    coverage_ad = np.zeros(true_val.shape[0])
    coverage_unad = np.zeros(true_val.shape[0])

    adjusted_intervals = np.zeros((2,nactive))
    adjusted_intervals[0,:] = (result[:, 2])[active_ind]
    adjusted_intervals[1,:] = (result[:, 3])[active_ind]

    unadjusted_intervals = np.zeros((2, nactive))
    unadjusted_intervals[0, :] = (result[:, 4])[active_ind]
    unadjusted_intervals[1, :] = (result[:, 5])[active_ind]

    for l in range(nactive):
        if (adjusted_intervals[0, l] <= true_val[l]) and (true_val[l] <= adjusted_intervals[1, l]):
            coverage_ad[l] += 1
        if (unadjusted_intervals[0, l] <= true_val[l]) and (true_val[l] <= unadjusted_intervals[1, l]):
            coverage_unad[l] += 1

    adjusted_coverage = float(coverage_ad.sum() / nactive)
    unadjusted_coverage = float(coverage_unad.sum() / nactive)

    return adjusted_coverage, unadjusted_coverage, FDR, power


def do_evaluation(args):
    # sys.stderr.write("Evaluating lasso selection and inference results\n")
    s = int(args.seed)
    i = int(args.geneid)
    sel_type = str(args.seltype)

    trial_dir = os.path.join(args.indir,"trial_"+str(s))

    X_fname = os.path.join(args.indir,"X_data","X_"+str(i)+".txt")
    result_fname = os.path.join(trial_dir, "infs_"+sel_type, "sel_out_"+str(i)+".txt")
    sig_fname = os.path.join(trial_dir,"signals.txt")
    sime_fname = os.path.join(trial_dir,"simes_result.txt")
    sel_res = read_simes(sime_fname)
    sel = sel_res[0][i]

    if not sel: 
        sys.stderr.write("Family "+str(i)+" was not selected by Simes\n")
        return

    if not os.path.isfile(result_fname):
        sys.stderr.write("Family "+str(i)+" was selected by Simes, but second stage selection was not complete\n")
        sys.exit(1)

    try:
        true_sig, snr, _ = read_signals(sig_fname)
        num_true_sig = true_sig[i]
    except OSError:
        sys.stderr.write('Cannot open: '+sig_fname+"\n")
    try:
        X = read_X(X_fname)
    except OSError:
        sys.stderr.write('Cannot open: '+X_fname+"\n")
    try: 
        result = np.loadtxt(result_fname) 
    except OSError:
        sys.stderr.write('Cannot open: '+result_fname+"\n")

    # adjusted_coverage, unadjusted_coverage, FDR, power = evaluate_hierarchical_results(result, X, s, snr)
    print(evaluate_hierarchical_results(result, X, num_true_sig, snr))

def do_inference(args):
    sys.stderr.write("Running lasso selection and inference\n")
    np.random.seed(0)
    s = int(args.seed)
    i = int(args.geneid)
    sel_type = str(args.seltype)
    n_genes = int(args.ngenes)

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
        hierarchical_inference(result_file, X, y, index, simes_level, pgenes, 
                J=rej, t_0=idx_order, T_sign=sign, selection_method=sel_type)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="run hierarchical inference or evaluate its simulation results")
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('inference', help='') 
    command_parser.add_argument('-d', '--indir', required=True)
    command_parser.add_argument('-s', '--seed', required=True)
    command_parser.add_argument('-i', '--geneid', required=True)
    command_parser.add_argument('-g', '--ngenes', required=True)
    command_parser.add_argument('-t', '--seltype', required=True)
    command_parser.set_defaults(func=do_inference)

    command_parser = subparsers.add_parser('evaluate', help='') 
    command_parser.add_argument('-d', '--indir', required=True)
    command_parser.add_argument('-s', '--seed', required=True)
    command_parser.add_argument('-i', '--geneid', required=True)
    command_parser.add_argument('-t', '--seltype', required=True)
    command_parser.set_defaults(func=do_evaluation)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)


