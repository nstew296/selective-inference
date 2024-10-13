import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as ndist
import regreg.api as rr
from selectinf.randomized.randomization import randomization
from selectinf.randomized.multitask_lasso import multi_task_lasso


def common_format(ax):
    ax.grid(True, which='both', color='#f0f0f0')
    return ax


def set_boxplot_style(bp, color, linestyle):
    plt.setp(bp['boxes'], color=color, linestyle=linestyle, linewidth=3.5)
    plt.setp(bp['whiskers'], color=color, linestyle=linestyle, linewidth=3.5)
    plt.setp(bp['caps'], color=color, linewidth=2.5)
    plt.setp(bp['medians'], color=color, linewidth=2.5)


def ds_multi_task_tune(predictor_vars_selection, predictor_vars_inference, predictor_vars_validate,
                       response_selection, response_inference, response_validate, weight_list):
    """

    :param predictor_vars_selection: fMRL data for multi-task model selection
    :param predictor_vars_inference: fMRL data for inference
    :param predictor_vars_validate: fMRI data for tuning parameter selection
    :param response_selection: Dict of neurocognitive data for multi-task model selection
    :param response_inference: Dict of neurocognitive data for inference
    :param response_validate: Dict of neurocognitive data for tuning parameter selection
    :param weight_list: List of candidate tuning parameters
    :return: Two dictionaries whose keys are the candidate tuning parameters. The first dictionary gives the active
    signs for each candidate value and the second dictionary gives the predictive r's on the validation data
    for each candidate value.
    """

    ntask = len(response_inference)
    nfeatures = predictor_vars_inference.shape[1]
    sample_sizes_inference = predictor_vars_inference.shape[0]

    ridge_terms = np.zeros(ntask)
    randomizers = None

    # Estimate noise level (data is centered, saving 1 dof)
    noise_levels = []
    for j in range(ntask):
        noise_levels.append(np.sqrt(np.sum(np.asarray(response_inference[j] - predictor_vars_inference.dot(
            np.linalg.pinv(predictor_vars_inference).dot(response_inference[j]))) ** 2) /
                                    (sample_sizes_inference - nfeatures)))

    active_dict = {}
    pred_r_validate = {}

    # Fit MTL model (randomization/perturbation variable set to zero)
    for weight in weight_list:
        feature_weight = weight * np.ones(nfeatures)
        perturbations = np.zeros((nfeatures, ntask))
        loglikes = {j: rr.glm.gaussian(predictor_vars_selection, response_selection[j], coef=1., quadratic=None) for j
                    in range(ntask)}
        multi_lasso = multi_task_lasso(loglikes, np.asarray(feature_weight), ridge_terms, randomizers, nfeatures, ntask,
                                       perturbations)
        active_signs = multi_lasso.fit(perturbations=perturbations)

        # Compute predictive r on validation data
        predictive_r = []
        if (active_signs != 0).sum() > 0:
            for j in range(ntask):
                if (active_signs[:, j] != 0).sum() == 0:
                    predictive_r.append(None)
                else:
                    X = predictor_vars_inference
                    y = response_inference[j]
                    observed_target = np.linalg.pinv(X[:, (active_signs[:, j] != 0)]).dot(y)
                    predictive_r.append(
                        np.corrcoef(response_validate[j], predictor_vars_validate[:, (active_signs[:, j] != 0)].dot(
                            observed_target))[0, 1])
                    print(predictive_r)

        active_dict[weight] = active_signs
        pred_r_validate[weight] = predictive_r

    return active_dict, pred_r_validate


def ds_multi_task_selection_inference(predictor_vars_selection, predictor_vars_inference,
                                      predictor_vars_test, response_selection, response_inference,
                                      response_test, weight):
    """

    :param predictor_vars_selection: fMRI data for multi-task model selection
    :param predictor_vars_inference: fMRI data for inference
    :param predictor_vars_test: fMRI data for testing
    :param response_selection: Dict of neurocognitive data for multi-task model selection
    :param response_inference: Dict of neurocognitive data for inference
    :param response_test: Dict of neurocognitive data for testing
    :param weight: Tuning parameter
    :return: The coefficient estimates (final_estimates), the data splitting confidence intervals (final_intervals),
    the interval lengths (ds_interval_lengths), a dictionary of selected variables by task (all_variables_ds),
    a dictionary of significant variables by task (significant_variables), the average rMSE per task (final error),
    a list of predictive r's per task (predictive_r), and the coefficients of variation (final_coefs_var)
    """
    ntask = len(response_inference)
    ridge_terms = np.zeros(ntask)
    randomizers = None

    nfeatures = predictor_vars_inference.shape[1]
    sample_sizes_inference = predictor_vars_inference.shape[0]
    sample_sizes_test = predictor_vars_test.shape[0]

    # Estimate noise level (data is centered, saving 1 dof)
    noise_levels = []
    for j in range(ntask):
        noise_levels.append(np.sqrt(np.sum(np.asarray(response_inference[j] - predictor_vars_inference.dot(
            np.linalg.pinv(predictor_vars_inference).dot(response_inference[j]))) ** 2) / (
                                            sample_sizes_inference - nfeatures)))
    dispersions = [noise_levels[j] ** 2 for j in range(len(noise_levels))]

    # Fit MTL model (randomization/perturbation variable set to zero)
    feature_weight = weight * np.ones(nfeatures)
    perturbations = np.zeros((nfeatures, ntask))
    loglikes = {j: rr.glm.gaussian(predictor_vars_selection, response_selection[j], coef=1., quadratic=None) for j in
                range(ntask)}
    multi_lasso = multi_task_lasso(loglikes, np.asarray(feature_weight), ridge_terms, randomizers, nfeatures, ntask,
                                   perturbations)
    active_signs = multi_lasso.fit(perturbations=perturbations)

    estimate = []
    CV = []
    CIs = np.empty((0, 2))
    predictive_r = []

    # Conduct inference (estimate target and construct confidence intervals)
    if (active_signs != 0).sum() > 0:
        error = 0
        for j in range(ntask):
            X = predictor_vars_inference
            y = response_inference[j]
            observed_target = np.linalg.pinv(X[:, (active_signs[:, j] != 0)]).dot(y)
            estimate = np.concatenate([estimate, observed_target])

            Qfeat = np.linalg.inv(X[:, (active_signs[:, j] != 0)].T.dot(X[:, (active_signs[:, j] != 0)]))
            cov_target = Qfeat * dispersions[j]
            CV = np.concatenate([CV, np.sqrt(np.diag(cov_target)) / np.abs(observed_target)])

            alpha = 1. - 0.90
            quantile = ndist.ppf(1 - alpha / 2.)
            intervals = np.vstack([observed_target - quantile * np.sqrt(np.diag(cov_target)),
                                   observed_target + quantile * np.sqrt(np.diag(cov_target))]).T
            CIs = np.vstack([CIs, intervals])

            # Compute predictive r on hold-out data
            num_active = np.sum(active_signs[:, j] != 0)
            if num_active == 0:
                error += np.sqrt(np.sum(np.square(response_test[j])) / sample_sizes_test)
                predictive_r.append(None)
            else:
                error += np.sqrt(np.sum(np.square(
                    response_test[j] - predictor_vars_test[:, (active_signs[:, j] != 0)].dot(
                        observed_target))) / sample_sizes_test)
                predictive_r.append(
                    np.corrcoef(response_test[j], predictor_vars_test[:, (active_signs[:, j] != 0)].dot(
                        observed_target))[0, 1])

    else:
        error = 0
        for j in range(ntask):
            error += np.sqrt(np.linalg.norm(response_test[j], 2) ** 2 / sample_sizes_test)
        CIs = np.asarray([np.nan, np.nan])

    final_error = error / ntask
    final_estimates = estimate
    final_intervals = CIs
    final_coefs_var = CV

    # Evaluate significance (variables with confidence intervals that do not contain zero)
    significant = [final_intervals[j, 0] > 0 or final_intervals[j, 1] < 0 for j in range(np.shape(final_intervals)[0])]
    significant_variables = {}
    all_variables_ds = {}
    placeholder = 0
    for j in range(ntask):
        active_ = active_signs[:, j] != 0
        new_placeholder = np.sum(active_)
        # Retrieve predictors that are non-zero for given task. Note: np.nonzero returns tuple (non_zero_array,)
        all_variables_ds[j] = np.nonzero(active_)[0]
        significant_variables[j] = np.nonzero(active_)[0][significant[placeholder:placeholder + new_placeholder]]
        placeholder = placeholder + new_placeholder

    ds_interval_lengths = np.asarray(final_intervals[:, 1] - final_intervals[:, 0])

    return (final_estimates,
            final_intervals,
            ds_interval_lengths,
            all_variables_ds,
            significant_variables,
            final_error,
            predictive_r,
            final_coefs_var)


def rand_multi_task_tune(predictor_vars_train, predictor_vars_validate, response_train,
                         response_validate, weight_list, noise, rand_scale=0.7):
    """

    :param predictor_vars_train: fMRI data for multi-task selective inference
    :param predictor_vars_validate: fMRI data for tuning parameter selection
    :param response_train: Dict of neurocognitive data for multi-task selective inference
    :param response_validate: Dict of neurocognitive data for tuning parameter selection
    :param weight_list: A list of candidate tuning parameters
    :param noise: Pre-specified Gaussian perturbations
    :param rand_scale: Scales the noise level of the data to determine the s.d. of the perturbation term
    :return: Two dictionaries whose keys are the candidate tuning parameters. The first dictionary gives the active
    signs for each candidate value and the second dictionary gives the predictive r's on the validation data
    for each candidate value.
    """

    ntask = len(response_train)
    nfeatures = predictor_vars_train.shape[1]
    sample_size = predictor_vars_train.shape[0]

    # Noise estimation and randomization variable setup
    noise_levels = []
    for j in range(ntask):
        noise_levels.append(np.sqrt(np.sum(np.array(response_train[j] - predictor_vars_train.dot(
            np.linalg.pinv(predictor_vars_train).dot(response_train[j]))) ** 2) / (sample_size - nfeatures)))
    dispersions = [noise_levels[j] ** 2 for j in range(len(noise_levels))]

    # The MTL functions require the randomizer to be specified, but I will pass in the perturbation variable
    # rather than sampling from the randomizer. This allows the same noise data to be used across methods.
    ridge_terms = np.zeros(ntask)
    randomizer_scales = rand_scale * np.asarray([noise_levels[j] for j in range(ntask)])
    randomizers = {j: randomization.isotropic_gaussian((nfeatures,), randomizer_scales[j]) for j in range(ntask)}

    active_dict = {}
    pred_r_validate = {}

    # Fit MTL model and perform inference for given tuning parameter
    for weight in weight_list:
        feature_weight = weight * np.ones(nfeatures)
        perturbations = np.array(
            [randomizer_scales[j] * noise[j * nfeatures:(j + 1) * nfeatures] for j in range(ntask)]).T
        loglikes = {j: rr.glm.gaussian(predictor_vars_train, response_train[j], coef=1., quadratic=None)
                    for j in range(ntask)}
        multi_lasso = multi_task_lasso(
            loglikes, np.asarray(feature_weight), ridge_terms, randomizers, nfeatures, ntask, perturbations)
        active_signs = multi_lasso.fit(perturbations=perturbations)
        estimate, observed_info_mean, Z_scores, pvalues, intervals = multi_lasso.multitask_inference_hetero(
            dispersions=dispersions)

        active_dict[weight] = active_signs
        predictive_r = []
        # Calculate predictive r on validation data
        if (active_signs != 0).sum() > 0:
            idx = 0
            for j in range(ntask):
                idx_new = np.sum(active_signs[:, j] != 0)
                if idx_new == 0:
                    predictive_r.append(None)
                else:
                    predictive_r.append(
                        np.corrcoef(response_validate[j], predictor_vars_validate[:, (active_signs[:, j] != 0)].dot(
                            estimate[idx:idx + idx_new]))[0, 1])
                idx = idx + idx_new
                print(predictive_r)

        pred_r_validate[weight] = predictive_r

    return active_dict, pred_r_validate


def rand_multi_task_selection_inference(predictor_vars_train, predictor_vars_test, response_train,
                                        response_test, weight, noise, rand_scale=0.7):
    """

    :param predictor_vars_train: fMRI data for multi-task selective inference
    :param predictor_vars_test: fMRI data for testing
    :param response_train: Dict of neurocognitive data for multi-task selective inference
    :param response_test: Dict of neurocognitive data for testing
    :param weight: Tuning parameter
    :param noise: Pre-specified Gaussian perturbations
    :param rand_scale: Scales the noise level of the data to determine the s.d. of the perturbation term
    :return: The selective MLE (final_estimates), the post-selection confidence intervals (intervals),
    the interval lengths (selective_interval_lengths), a dictionary of selected variables by task (all_variables),
    a dictionary of significant variables by task (significant_variables), the average rMSE per task (final_avg_error),
    a list of predictive r's per task (predictive_r), and the coefficients of variation (coefs_var)
    """

    ntask = len(response_train)
    nfeatures = predictor_vars_train.shape[1]
    sample_size = predictor_vars_train.shape[0]
    sample_size_test = predictor_vars_test.shape[0]

    # Noise estimation and randomization variable setup
    noise_levels = []
    for j in range(ntask):
        noise_levels.append(np.sqrt(np.sum(np.array(response_train[j] - predictor_vars_train.dot(
            np.linalg.pinv(predictor_vars_train).dot(response_train[j]))) ** 2) / (sample_size - nfeatures)))
    dispersions = [noise_levels[j] ** 2 for j in range(len(noise_levels))]

    randomizer_scales = rand_scale * np.asarray([noise_levels[j] for j in range(ntask)])
    randomizers = {j: randomization.isotropic_gaussian((nfeatures,), randomizer_scales[j]) for j in range(ntask)}
    ridge_terms = np.zeros(ntask)

    # Fit MTL model and perform inference
    feature_weight = weight * np.ones(nfeatures)
    perturbations = np.array([randomizer_scales[j] * noise[j * nfeatures:(j + 1) * nfeatures] for j in range(ntask)]).T
    loglikes = {j: rr.glm.gaussian(predictor_vars_train, response_train[j], coef=1., quadratic=None)
                for j in range(ntask)}
    multi_lasso = multi_task_lasso(
        loglikes, np.asarray(feature_weight), ridge_terms, randomizers, nfeatures, ntask, perturbations)
    active_signs = multi_lasso.fit(perturbations=perturbations)
    estimate, observed_info_mean, Z_scores, pvalues, intervals = multi_lasso.multitask_inference_hetero(
        dispersions=dispersions)
    coefs_var = np.sqrt(np.diag(observed_info_mean)) / np.abs(estimate)

    # Calculate final testing error and predictive r on test set
    if (active_signs != 0).sum() > 0:
        final_error = 0
        predictive_r = []
        idx = 0
        for j in range(ntask):
            idx_new = np.sum(active_signs[:, j] != 0)
            if idx_new == 0:
                final_error += np.sqrt(np.sum(np.square(response_test[j])) / sample_size_test)
                predictive_r.append(None)
            else:
                final_error += np.sqrt(np.sum(
                    np.square((response_test[j] - predictor_vars_test[:, (active_signs[:, j] != 0)].dot(
                        estimate[idx:idx + idx_new])))) / sample_size_test)
                predictive_r.append(np.corrcoef(response_test[j], predictor_vars_test[:, (active_signs[:, j] != 0)].dot(
                    estimate[idx:idx + idx_new]))[0, 1])
            idx = idx + idx_new

    else:
        final_error = 0
        predictive_r = []
        for j in range(ntask):
            final_error += np.sqrt(np.linalg.norm(response_test[j], 2) ** 2 / sample_size_test)

    # Average final testing error by task
    final_avg_error = final_error / ntask

    # Identify intervals that do not cover zero
    all_variables = {}
    significant = [intervals[j, 0] > 0 or intervals[j, 1] < 0 for j in range(np.shape(intervals)[0])]
    significant_variables = {}
    placeholder = 0
    for j in range(ntask):
        # Identify variables (by task) corresponding to the significant intervals
        active_ = active_signs[:, j] != 0
        new_placeholder = np.sum(active_)
        # Retrieve predictors that are non-zero for given task. Note: np.nonzero returns tuple (non_zero_array,)
        all_variables[j] = np.nonzero(active_)[0]
        significant_variables[j] = np.nonzero(active_)[0][significant[placeholder:placeholder + new_placeholder]]
        placeholder = placeholder + new_placeholder

    selective_interval_lengths = np.asarray(intervals[:, 1] - intervals[:, 0])

    return (estimate,
            intervals,
            selective_interval_lengths,
            all_variables,
            significant_variables,
            final_avg_error,
            predictive_r,
            coefs_var)


def standardize_lambda_path_randomized_mtl(predictor_vars_train, response_train, noise, rand_scale=0.7,
                                           start_size=170, stop_size=15):
    """

    :param predictor_vars_train: fMRI data for multi-task selective inference
    :param predictor_vars_validate: fMRI data for tuning parameter selection
    :param response_train: Neurocognitive data for multi-task selective inference
    :param response_validate: Neurocognitive data for tuning parameter selection
    :param weight_list: A list of candidate tuning parameters
    :param noise: Pre-specified Gaussian perturbations
    :param rand_scale: Scales the noise level of the data to determine the s.d. of the perturbation term
    :return: Two dictionaries whose keys are the candidate tuning parameters. The first dictionary gives the active
    signs for each candidate value and the second dictionary gives the predictive r's on the validation data
    for each candidate value.
    """

    ntask = len(response_train)
    nfeatures = predictor_vars_train.shape[1]
    sample_size = predictor_vars_train.shape[0]

    # Noise estimation and randomization variable setup
    noise_levels = []
    for j in range(ntask):
        noise_levels.append(np.sqrt(np.sum(np.array(response_train[j] - predictor_vars_train.dot(
            np.linalg.pinv(predictor_vars_train).dot(response_train[j]))) ** 2) / (sample_size - nfeatures)))

    ridge_terms = np.zeros(ntask)
    randomizer_scales = rand_scale * np.asarray([noise_levels[j] for j in range(ntask)])
    randomizers = {j: randomization.isotropic_gaussian((nfeatures,), randomizer_scales[j]) for j in range(ntask)}

    perturbations = np.array(
        [randomizer_scales[j] * noise[j * nfeatures:(j + 1) * nfeatures] for j in range(ntask)]).T

    loglikes = {j: rr.glm.gaussian(predictor_vars_train, response_train[j], coef=1., quadratic=None)
                for j in range(ntask)}

    weight = 0.9
    while True:
        feature_weight = weight * np.ones(nfeatures)

        multi_lasso = multi_task_lasso(
            loglikes, np.asarray(feature_weight), ridge_terms, randomizers, nfeatures, ntask, perturbations)
        active_signs = multi_lasso.fit(perturbations=perturbations)

        avg_model_size = np.sum(active_signs != 0) / ntask
        if avg_model_size < start_size:
            min_weight = weight
            break

        weight = weight + .1

    weight = 3.0
    while True:
        feature_weight = weight * np.ones(nfeatures)

        multi_lasso = multi_task_lasso(
            loglikes, np.asarray(feature_weight), ridge_terms, randomizers, nfeatures, ntask, perturbations)
        active_signs = multi_lasso.fit(perturbations=perturbations)

        avg_model_size = np.sum(active_signs != 0) / ntask
        if avg_model_size <= stop_size:
            max_weight = weight
            break

        weight = weight + .1
    return np.linspace(min_weight, max_weight, 10)
