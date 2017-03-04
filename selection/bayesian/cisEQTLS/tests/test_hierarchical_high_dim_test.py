from __future__ import print_function
import sys
import time
import random
import glob
import numpy as np
from selection.tests.instance import gaussian_instance
from selection.bayesian.initial_soln import selection, instance
from selection.randomized.api import randomization
from selection.bayesian.cisEQTLS.Simes_selection import simes_selection
from scipy.stats import norm as normal
from selection.bayesian.cisEQTLS.Simes_selection import BH_q
from selection.bayesian.cisEQTLS.inference_2sels import selection_probability_genes_variants, \
    sel_prob_gradient_map_simes_lasso, selective_inf_simes_lasso
from selection.bayesian.cisEQTLS.inference_per_gene import selection_probability_variants, \
    sel_prob_gradient_map_lasso, selective_inf_lasso

from  hierarchical_high_dim_test import hierarchical_inference

#note that bh level is to decided upon how many we end up selecting:

if __name__ == "__main__":
    X, y, true_beta, nonzero, noise_variance = gaussian_instance(n=10, p=20, s=0, sigma=1, rho=0, snr=5.)
    result_file = "res_test_hierarchical_single.txt"
    hierarchical_inference(outputfile=result_file, simes_level=0.01, X=X, y=y, selection_method ="single")
    result_file = "res_test_hierarchical_double.txt"
    hierarchical_inference(outputfile=result_file, simes_level=0.01, X=X, y=y, selection_method ="double")
