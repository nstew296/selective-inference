from __future__ import print_function
import sys
import os
import errno    
import itertools
import multiprocessing as mp
import argparse
import numpy as np
import random
from selection.bayesian.cisEQTLS.Simes_selection import simes_selection
from selection.tests.instance import gaussian_instance
from selection.bayesian.cisEQTLS.initial_sol_wocv import selection, instance


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def imerge(a, b, mode="concat"):
    if mode == "append":
        for i, j in itertools.izip(a,b):
            yield i + (j,)
    elif mode == "concat":  
        for i, j in itertools.izip(a,b):
            yield i + j
    else:
        sys.stderr.write("Mode not recognized\n")
        sys.exit(1)

def write_X(f, X): 
    np.savetxt(f, X, delimiter='\t')

def read_X(f):
    X = np.loadtxt(f, delimiter='\t')
    return X

def write_y(f, y):
    np.savetxt(f, y, delimiter='\t')

def read_y(f):
    y = np.loadtxt(f, delimiter='\t')
    return y

def write_simes(fname, threshold, sel_simes):
    # a numpy matrix with the following fields
    # fields:
    # 0: select or not
    # 1: index of the most significant variable
    # 2: the first index of the rejected (sorted) variable 
    # 3: sign of the T statistic
    # 4: (bonforoni) threshold
    # 5: rejection index

    if sel_simes:
        i_0, rej, t_0, sign = sel_simes
        rej = ",".join(map(str,rej))
        sign = int(sign)
        results = "\t".join(map(str,[1, i_0, t_0, sign, threshold, rej]))
    else:
        results = "\t".join(map(str,(0, 0, 0, 0, threshold, 0)))
    
    with open(fname, 'w') as f:
        f.write(results)
    # np.savetxt(f, result, delimiter='\t')

def merge_simes(fname , file_paths):
    with open(fname, 'w') as f:
        for fn in file_paths:
            with open(fn, 'r') as g:
                f.write(g.readline().strip()+"\n") 

def read_simes(fname, full=False):
    
    if full:
        # read sumary file
        with open(fname, "r") as f:
            results = [line.strip().split('\t') for line in f]
            sel = [int(result[0]) for result in results]
            idx_sigs = [int(result[1]) for result in results]
            idx_orders = [int(result[2]) for result in results]
            signs = [int(result[3]) for result in results]
            threshold = float(results[0][4])
            rej_idx = [map(int,map(float,result[5].split(','))) for result in results]
    else:
        with open(fname, "r") as f:
            result = f.readline().strip().split('\t')
            sel = int(result[0])
            idx_sigs = int(result[1]) 
            idx_orders = int(result[2])
            signs = int(result[3])
            threshold = float(result[4])
            rej_idx = map(int,map(float,result[5].split(','))) 

    return sel, idx_sigs, idx_orders, signs, threshold, rej_idx
    

def read_signals(fname):
    with open(fname, "r") as f:
        results = [line.strip().split('\t') for line in f]
    true_sig = [int(result[0]) for result in results]
    snr = results[1][0] 
    sigma = results[2][0] 

    return true_sig, snr, sigma

def run_simes_selection(fltuple):
    """ 
    Run simes selection and store results 
    """
    s_f, X_f, y_f, alpha = fltuple 
    # check if the simes result already exists
    if os.path.isfile(s_f):
        return 

    X = read_X(X_f)
    y = read_y(y_f)
    sel_simes = simes_selection(X, y, alpha=alpha, randomizer='gaussian')
    write_simes(s_f, alpha, sel_simes)

def signal_setting(setting, tot_s, snr=5.0, sigma=1.0, outfile=None):
    if setting == 0:
        svals = itertools.chain(itertools.repeat((1, snr, sigma), tot_s/4),
                                itertools.repeat((5, snr, sigma), tot_s/4), 
                                itertools.repeat((10, snr, sigma), tot_s/4), 
                                itertools.repeat((0, snr, sigma), tot_s-3*(tot_s/4)))
        svals, svals_return = itertools.tee(svals)
    else:
        sys.stderr.write("Setting not recognized\n")
        sys.exit(1)

    if outfile:
        with open(outfile, 'w') as f:
            for signal in svals:
                f.write("\t".join(map(str,signal))+"\n")

    return(svals_return)

def generate_fixed_design(fltuple):
    """ 
    Generate data and save to file 
    """
    # Inputs:
    # f -- output file to store X
    # n -- number of samples 
    # p -- number of covariates (snps per gene)
    # s -- the random seed for the gene 
    # Outputs: 
    # None 

    f, n, p, s = fltuple

    np.random.seed(s)

    rho = 0.0
    scale=True
    center=True 
    X = (np.sqrt(1-rho) * np.random.standard_normal((n,p)) + 
        np.sqrt(rho) * np.random.standard_normal(n)[:,None])
    if center:
        X -= X.mean(0)[None,:]
    if scale:
        X /= (X.std(0)[None,:] * np.sqrt(n))

    write_X(f, X)


def generate_response(fltuple):
    """ 
    Generate data and save to file 
    """
    # Inputs:
    # y_f :  output file storing y
    # X_f : input file storing X
    # seed : random seed number
    # s : int  True sparsity
    # snr : float Size of each coefficient
    # sigma: float Noise level
    # Outputs: 
    # None 

    y_f, X_f, seed, s, snr, sigma = fltuple

    np.random.seed(seed)
    random_signs=False
    df=np.inf

    X = read_X(X_f)
    n, p = X.shape

    beta = np.zeros(p) 
    beta[:s] = snr 
    if random_signs:
        beta[:s] *= (2 * np.random.binomial(1, 0.5, size=(s,)) - 1.)
    active = np.zeros(p, np.bool)
    active[:s] = True

    # noise model
    def _noise(n, df=np.inf):
        if df == np.inf:
            return np.random.standard_normal(n)
        else:
            sd_t = np.std(tdist.rvs(df,size=50000))
            return tdist.rvs(df, size=n) / sd_t

    Y = (X.dot(beta) + _noise(n, df)) * sigma
    write_y(y_f, Y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('job_type', help='evalSimes, runSimes, generateX or generateY')
    parser.add_argument('-o', '--outdir', required=True)
    parser.add_argument('-n', '--nsamples', default=350)
    parser.add_argument('-p', '--nsnps', default=7000)
    parser.add_argument('-g', '--ngenes', default=5000)
    parser.add_argument('-k', '--nproc', default=1)
    parser.add_argument('-s', '--seed', default=0)
    parser.add_argument('-t', '--setting', default=0)

    args = parser.parse_args()
    
    g = int(args.ngenes)
    p = int(args.nsnps)
    n = int(args.nsamples)
    s = int(args.seed) # or trial ID
    
    if args.job_type =="evalSimes":
        trial_dir = os.path.join(args.outdir,"trial_"+str(s))
        sime_fname = os.path.join(trial_dir,"simes_result.txt")
        sig_fname = os.path.join(trial_dir, 'signals.txt')

        selection = np.array(read_simes(sime_fname,full=True)[0])
        true_sigs = np.array(read_signals(sig_fname)[0]) > 0 

        tp = np.sum(selection * true_sigs)
        tn = np.sum(np.logical_not(selection) * np.logical_not(true_sigs))
        fn = np.sum((true_sigs - selection) > 0 )
        fp = np.sum((true_sigs - selection) < 0 )
        
        print("True positives:  "+str(tp))
        print("True negative:   "+str(tn))
        print("False positives: "+str(fp))
        print("False negatives: "+str(fn))

    if args.job_type =="runSimes":
        # run simes procedure on the matrices
        random.seed(s)
        trial_dir = os.path.join(args.outdir,"trial_"+str(s))
        y_dir = os.path.join(trial_dir,"y_data")
        x_dir = os.path.join(args.outdir,"X_data")
        y_fnames = [os.path.join(y_dir,"y_"+str(i)+".txt") for i in xrange(g)]
        x_fnames = [os.path.join(x_dir,"X_"+str(i)+".txt") for i in xrange(g)]
        simes_level = itertools.repeat(0.1 / g, g)

        tmp_dir = os.path.join(trial_dir,"tmp")
        mkdir_p(tmp_dir)
        s_fnames = [os.path.join(tmp_dir,"s_"+str(i)+".txt") for i in xrange(g)]
    
        fltuple = imerge(itertools.izip(s_fnames, x_fnames,y_fnames), simes_level, mode="append")
        pool=mp.Pool(processes=int(args.nproc))
        pool.map(run_simes_selection, fltuple)
        
        sime_fname = os.path.join(trial_dir,"simes_result.txt")
        merge_simes(sime_fname, s_fnames)


    if args.job_type == "generateX":
        # store random designs in X_data, one per gene
        sys.stderr.write("Generating X data\n")
        x_dir = os.path.join(args.outdir,"X_data")
        mkdir_p(x_dir)
        sys.stderr.write("Writing to: "+x_dir+"\n")
        x_fnames = [os.path.join(x_dir,"X_"+str(i)+".txt") for i in xrange(g)]

        fltuple = imerge(itertools.product(x_fnames, [n], [p]), xrange(g), mode="append")  
        pool=mp.Pool(processes=int(args.nproc))
        pool.map(generate_fixed_design, fltuple)


    if args.job_type == "generateY":
        # generate responses for each signal
        trial_dir = os.path.join(args.outdir,"trial_"+str(s))
        mkdir_p(trial_dir)
        sys.stderr.write("Generating signals in "+trial_dir+"\n")
        sig_fname = os.path.join(trial_dir, 'signals.txt')
        svals= signal_setting(int(args.setting), g, outfile=sig_fname)

        y_dir = os.path.join(trial_dir,"y_data")
        mkdir_p(y_dir)
        sys.stderr.write("Writing y data to: "+y_dir+"\n")

        y_fnames = [os.path.join(y_dir,"y_"+str(i)+".txt") for i in xrange(g)]
        x_dir = os.path.join(args.outdir,"X_data")
        x_fnames = [os.path.join(x_dir,"X_"+str(i)+".txt") for i in xrange(g)]
        seed_ns = xrange(s, g+s) # seed for additive noise
        
        meta_info = imerge(itertools.izip(y_fnames,x_fnames), seed_ns, mode="append")
        
        fltuple = imerge(meta_info, svals)
        pool=mp.Pool(processes=int(args.nproc))
        pool.map(generate_response, fltuple)
