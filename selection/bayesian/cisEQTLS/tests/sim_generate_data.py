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
        

def signal_setting(setting, tot_s, snr=5.0, sigma=1.0, outdir=None):
    if setting == 0:
        svals = itertools.chain(itertools.repeat((1, snr, sigma), tot_s/4),
                                itertools.repeat((5, snr, sigma), tot_s/4), 
                                itertools.repeat((10, snr, sigma), tot_s/4), 
                                itertools.repeat((0, snr, sigma), tot_s-3*(tot_s/4)))
        svals, svals_return = itertools.tee(svals)
    else:
        sys.stderr.write("Setting not recognized\n")
        sys.exit(1)

    if outdir:
        outfile = os.path.join(outdir,'signals.txt')
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

    np.savetxt(f, X, delimiter='\t')


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

    X = np.loadtxt(X_f, delimiter='\t')
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
    np.savetxt(y_f, Y, delimiter="\t")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('job_type', help='generateX or generateY')
    parser.add_argument('-o', '--outdir', required=True)
    parser.add_argument('-n', '--nsamples', default=350)
    parser.add_argument('-p', '--nsnps', default=5000)
    parser.add_argument('-g', '--ngenes', default=500)
    parser.add_argument('-k', '--nproc', default=1)
    parser.add_argument('-s', '--seed', default=0)
    parser.add_argument('-t', '--setting', default=0)

    args = parser.parse_args()
    
    g = int(args.ngenes)
    p = int(args.nsnps)
    n = int(args.nsamples)
    
    if args.job_type == "generateX":
        sys.stderr.write("Generating X data\n")
        # store random designs in X_data, one per gene
        x_dir = os.path.join(args.outdir,"X_data")
        mkdir_p(x_dir)
        sys.stderr.write("Writing to: "+x_dir+"\n")
        x_fnames = [os.path.join(x_dir,"X_"+str(i)+".txt") for i in xrange(g)]

        fltuple = imerge(itertools.product(x_fnames, [n], [p]), xrange(g), mode="append")  
        pool=mp.Pool(processes=int(args.nproc))
        pool.map(generate_fixed_design, fltuple)

    if args.job_type == "generateY":
        # generate responses for each signal
        s = int(args.seed)
            
        trial_dir = os.path.join(args.outdir,"trial_"+str(s))
        mkdir_p(trial_dir)
        sys.stderr.write("Generating signals in "+trial_dir+"\n")
        svals= signal_setting(int(args.setting), g, outdir=trial_dir)

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
         
