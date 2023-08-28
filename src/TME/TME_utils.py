"""
This code is based on Matlab code from the following paper:
@article{elsayed, 
title={Structure in neural population recordings: an expected byproduct of simpler phenomena?}, 
author={Gamaleldin Elsayed, John Cunningham}, 
journal={Nature Neuroscience}, 
volume={}, 
year={} }
Github: https://github.com/gamaleldin/TME
"""

from cmath import log
from typing import Tuple
import collections
import numpy as np
import sys
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb
from scipy.optimize import minimize


def fit_max_entropy(params: dict):
    """
    <TME>
    Copyright (C) 2017 Gamaleldin F. Elsayed and John P. Cunningham
        (see full notice in README)

    [maxEntropy] = fitMaxEntropy(params)

    This function solves for the Lagrangian multipliers of the maximum
    entropy distribution.

    Inputs:
        - params.
            - margCov: specified set of marginal covariance across tensor modes.
            - meanTensor: specified mean tensor.

    Outputs:
        - maxEntropy:
            .Lagrangians: are the eigenvalues of the largrangian
            multipliers of the optimization program
            .eigVectors: are the eigVectors of the largrangian
            multipliers of the optimization program.
            .objCost: is the final objective cost.
            .logObjperIter: is the transformed log objective cost at each
            iteration. Note, optimization is performed only on the
            log objective because the original objective can be
            problematic and slowly converges when the specified
            marginal covariance matrices are low rank. The optimal
            solution of the log objective and the original objective
            is the same and both the log objective and original objective
            values at the global optimum should be 0.
    """
    max_entropy = {}
    marg_cov = params["margCov"]
    # max entropy does not depend on the mean so just save it for the sampling func
    max_entropy["meanTensor"] = params["meanTensor"]
    # tensor size; ie the number of different modes of the tensor
    tensor_size = len(marg_cov)

    # eigenVectors of each of the specified marginal covariances
    eig_vectors = []
    # eigenValues of each of the specified marginal covariances
    eig_values = []
    # sum of each of the eigenValues of each of the specified marginal covariances
    tr_sigma = []
    dim = max_entropy["meanTensor"].shape  # tensor dimensions

    for i in range(tensor_size):
        if np.any(marg_cov[i]):
            # same as matlab
            tr = np.trace(marg_cov[i])
            continue

    # load all the inputs
    for i in range(tensor_size):
        # sigma the same
        sigma = marg_cov[i]
        if not np.any(sigma):
            sigma = np.eye(dim[i]) * (tr / dim[i])

        u, S, vh = np.linalg.svd(sigma)
        ix = np.arange(len(S))
        Q = u[:, ix]
        eig_vectors.append(Q)
        eig_values.append(S)
        tr_sigma.append(np.trace(sigma))

    max_entropy["eigVectors"] = np.array(eig_vectors)

    # the marginal covariances should all have the same trace (i.e. the sum of their eigenvalues should be equal)
    first_cond = tr_sigma - \
        np.mean(tr_sigma) >= -(np.sum(dim) * (np.spacing(1) ** 0.5))
    second_cond = tr_sigma - \
        np.mean(tr_sigma) <= (np.sum(dim) * (np.spacing(1) ** 0.5))
    if not np.all(np.logical_and(first_cond, second_cond)):
        print("The marginal covariances do not have the same trace.")

    # if the marginal covariances are low rank then the number of variables
    # that we solve for are less. If full rank the number of variables that we
    # solve for are equal to the sum of the tensor dimensions.

    fig_flg = False  # display summary figure flag
    # tensor size; ie the number of different dimensions of the tensor
    tensor_size = len(eig_values)
    dim = np.empty((tensor_size, 1))  # tensor dimensions
    tensor_ixs = np.arange(tensor_size)
    # if an eigenvalue is below this threshold it is considered 0.
    threshold = -10

    for x in tensor_ixs:
        dim[x] = len(eig_values[x])

    # prescale the eigenvalues for numerical stability
    pre_scale = np.sum(eig_values[0]) / np.mean(dim)
    log_eig_values = []  # log of the eigenvalues
    # true number of variables that we solve for, which is equal to the sum of the ranks of the marginal covariances
    opt_dim = np.empty((tensor_size, 1))

    for x in tensor_ixs:
        cur_log_eig_vals = np.log(eig_values[x] / pre_scale)
        # eigenvalues should be order apriori
        log_eig_values.append(
            cur_log_eig_vals[np.where(cur_log_eig_vals > threshold)[0]]
        )
        opt_dim[x] = len(log_eig_values[x])

    # instead of solving for the largrangians directly we optimize latent variables that is equal to the log of the lagrangians
    # initialization of the optimization variables
    log_l0 = []  # log of the lagrangians
    for x in tensor_ixs:
        n_x_set = tensor_ixs[np.where(tensor_ixs != x)[0]]
        log_l0.append(np.log(np.sum(opt_dim[n_x_set])) - log_eig_values[x])

    # optimization
    max_iter = [10000]  # maximum allowed iterations
    log_on = True
    # this function performs all the optimzation heavy lifting

    result = minimize(
        objective_max_entropy_tensor,
        np.hstack(log_l0),
        args=(log_eig_values, log_on),
        method="trust-constr",
    )
    log_l = result.x
    # calculates the optimal Largrangians from the latent by taking the exponential
    L = np.exp(log_l)

    lagrangians = []  # save the lagrangians to the output
    for x in tensor_ixs:
        L_dims = (np.arange(opt_dim[x]) + np.sum(opt_dim[0:x])).astype(int)
        inf_array = np.ones(((dim[x] - opt_dim[x]).astype(int)[0], 1)) * np.inf
        # add the lagrangians known solution (Infinity) of the zero marginal covariance eigenvalues (if low rank)
        lagrangians.append(
            np.squeeze(
                np.concatenate((np.expand_dims(L[L_dims], axis=-1), inf_array))
                / pre_scale
            )
        )

    # save and display summary
    obj_cost = objective_max_entropy_tensor(np.hstack(lagrangians), eig_values)
    log_obj_cost = objective_max_entropy_tensor(log_l, log_eig_values, log_on)
    max_entropy["Lagrangians"] = lagrangians
    # max_entropy["logObjperIter"] = log_obj_per_iter
    max_entropy["objCost"] = obj_cost

    # print("\n final cost value: \n")
    # print(
    #     "- gradient inconsistency with numerical gradient = %.5f \n",
    #     check_grad(
    #         "objective_max_entropy_tensor",
    #         np.random.randn(sum(opt_dim.astype(int))[0], 1),
    #         1e-5,
    #         log_eig_values,
    #         log_on,
    #         True,
    #     ),
    # )
    print("- final cost value = %1.5e \n", log_obj_cost)

    # print("\n cost: \n")
    # print(
    #     "- gradient inconsistency with numerical gradient = %.5f \n",
    #     check_grad(
    #         "objective_max_entropy_tensor",
    #         np.random.randn(sum(opt_dim.astype(int))[0], 1),
    #         1e-5,
    #         log_eig_values,
    #         False,
    #         True,
    #     ),
    # )
    print(" - cost at convergence = %.5f \n", obj_cost)

    if obj_cost > 1e-5:
        print("\n WARNING: algorithm did not converege, results may be inaccuarate \n")

    if fig_flg:
        # plot the eigenvalues and the lagrangians
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # this plot will no longer work
        ax.plot(log_obj_per_iter, "b-")
        ax.set_xlabel("iteration #")
        ax.set_ylabel("objective function value")
        ax.set_ylim([0, 1])
        # ax.plot(np.arange(tensor_size), log_eig_values, 'o', label='eigenvalues')
        # ax.plot(np.arange(tensor_size), log_l, 'o', label='lagrangians')
        plt.show()

    return max_entropy


def objective_max_entropy_tensor(
    L: np.ndarray, eig_values: list, log_on=False, return_grad=False
):
    """
    <TME>
    Copyright (C) 2017 Gamaleldin F. Elsayed and John P. Cunningham
        (see full notice in README)

    [f, gradf_L] = objectiveMaxEntropyTensor(L, eigValues)

    This function evaluates the objective function of the maximum entropy
    covariance eigenvalues problem

    Inputs:
        - L: is the vector of the stacked eigenvalues of the lagrangian
            matrices.
        - eigValues: cell with each element containing the vector of
            eigenvalues of the specified marginal covariance matrices.

    Outputs:
        - f: is the objective cost function evaluated at the input vector
            L.
        - gradf_L: is the gradient of the objective cost function evaluated
            at the input vector L with respect to L.
    Added in the log objective function here as well
    """

    # normalization value for the objective function and the gradient
    normalize_term = np.linalg.norm(np.hstack(eig_values)) ** 2
    # tensor size; i.e. the number of different dimensions of the tensor
    tensor_size = len(eig_values)
    dim = []  # tensor dimensions
    lagrangians = []
    tensor_ixs = np.arange(tensor_size)
    if log_on:
        log_ls = []

    for i in tensor_ixs:
        dim.append(len(eig_values[i]))
        L_inds = (np.sum(dim[0:i]) + np.arange(dim[i])).astype(int)
        if log_on:
            log_ls.append(L[L_inds])
            lagrangians.append(np.squeeze(np.exp(log_ls[i])))
        else:
            lagrangians.append(np.squeeze(L[L_inds]))

    # the building blocks of the gradient
    ls_tensor = diag_kron_sum(lagrangians)
    # kronecker sum of the lagrangian matrices eigenvalues
    ls_tensor = np.reshape(ls_tensor, dim, order="F")
    # elementwise inverse of the above
    inv_ls_tensor = 1.0 / ls_tensor
    # elementwise inverse square of the above
    inv_square_ls_tensor = (1.0 / ls_tensor) ** 2

    er = []
    # the cost decomposed to different tensor dimensions
    fx = []
    sums = []

    for i in tensor_ixs:
        n_x_set = tensor_ixs[np.where(tensor_ixs != i)[0]]
        # elementwise sums of invLsTensor

        if log_on:
            sums.append(np.log(sum_tensor(inv_ls_tensor, tuple(n_x_set))))
        else:
            sums.append(sum_tensor(inv_ls_tensor, tuple(n_x_set)))
        # error with respect to each marginal covariance eigenvalue
        er.append(np.reshape(eig_values[i],
                  (sums[i].shape), order="F") - sums[i])
        fx.append(np.reshape(er[i], (dim[i], 1), order="F") ** 2)

    f = np.sum(np.vstack(fx)) / normalize_term  # the objective value

    if not return_grad:
        return f

    else:
        # build the gradient from the blocks
        gradf_l = np.zeros((sum(dim), 1))  # gradient
        for i in tensor_ixs:
            n_x_set = tensor_ixs[np.where(tensor_ixs != i)[0]]

            if log_on:
                first_mat = np.reshape(
                    (2.0 * er[i] / sum_tensor(inv_ls_tensor, tuple(n_x_set)))
                    * (sum_tensor(inv_square_ls_tensor, tuple(n_x_set))),
                    (dim[i], 1),
                ).T
                grad_fx_lx = np.multiply(first_mat, lagrangians[i])
            else:
                grad_fx_lx = np.reshape(
                    (2.0 * er[i] *
                     sum_tensor(inv_square_ls_tensor, tuple(n_x_set))),
                    (dim[i], 1),
                ).T

            grad_fy_lx = np.zeros((dim[i], tensor_size - 1))
            z = 0
            for y in n_x_set:
                n_y_set = tensor_ixs[np.where(tensor_ixs != y)[0]]
                n_xy_set = tensor_ixs[
                    np.where((tensor_ixs != i) & (tensor_ixs != y))[0]
                ]

                if log_on:
                    grad_fy_lx[:, z] = np.multiply(
                        np.reshape(
                            sum_tensor(
                                (
                                    2.0
                                    * er[y]
                                    / sum_tensor(inv_ls_tensor, tuple(n_y_set))
                                )
                                * (sum_tensor(inv_square_ls_tensor, tuple(n_xy_set))),
                                y,
                            ),
                            (dim[i], 1),
                        ).T,
                        lagrangians[i],
                    )
                else:
                    grad_fy_lx[:, z] = np.reshape(
                        sum_tensor(
                            (
                                2.0
                                * er[y]
                                * sum_tensor(inv_square_ls_tensor, tuple(n_xy_set))
                            ),
                            y,
                        ),
                        (dim[i]),
                    )

                z += 1

            gradf_lx = np.expand_dims(
                np.sum(np.concatenate((grad_fx_lx.T, grad_fy_lx), axis=-1), axis=1),
                axis=1,
            )
            gradf_l[(np.arange(dim[i]) + np.sum(dim[0:i])).astype(int)] = gradf_lx

        gradf_l = gradf_l / normalize_term

        return f, gradf_l


def sample_TME(max_entropy: dict, num_surrogates: int):
    """
    <TME>
    Copyright (C) 2017 Gamaleldin F. Elsayed and John P. Cunningham
        (see full notice in README)

    [surrTensors] = sampleTME(maxEntropy, numSurrogates)

    This function generates tensor samples from the maximum entropy
    distribution with marginal mean and covariance constraints

    Inputs:
        - maxEntropy: tensor maximum entropy distribution parameters.
        - numSurrogates: number of surrogate tensors
    Outputs:
        - surrTensors: random surrogate tensors sampled from the maximum
        entropy distribution. Different tensor samples are along the last
        dimension of surrTensors.
    """
    if num_surrogates is not None:
        num_surrogates = 1

    # caculates the eigenvalues of maximum entropy covariance matrix from the lagrangians
    Lagrangians = max_entropy["Lagrangians"]
    # will assume max_entropy is a dict
    eig_vectors = max_entropy["eigVectors"]
    mean_tensor = max_entropy["meanTensor"]

    dim = []
    for i in range(len(eig_vectors)):
        dim.append(len(eig_vectors[i]))

    # tensor size; ie the number of different modes of the tensor
    tensor_size = len(dim)
    # D matches matlab code
    D = 1 / diag_kron_sum(Lagrangians)

    # sample from maximum entropy distribution
    # draw random samples from a normal distribution
    x = np.random.randn(np.prod(dim).astype(int), num_surrogates)
    # multiply by the inverse of the diagonal matrix of the lagrangians
    x = np.multiply(D**0.5, x)

    # load the eigenvectors of the covariance matrix
    # of the maximum entropy distribution
    Qs = []
    for i in range(tensor_size):
        Qs.append(eig_vectors[i])
    Qs = np.array(Qs)

    # efficiently multiply the samples by the eigenvectors of the covariance matrix of the maximum entropy distribution
    x = np.real(kron_mvprod(Qs, x))
    # add the mean tensor to the samples
    x = x + np.tile(mean_tensor.flatten(order="F"), (1, num_surrogates)).T

    if num_surrogates > 1:
        dim.append(num_surrogates)
    tensor_dims = np.array(dim).astype(int)
    surr_tensors = np.reshape(x, (tensor_dims), order="F")  # surrogate tensors

    return np.squeeze(surr_tensors)


def summarize_LDS(
    data_tensor: np.ndarray, model_dim: list, cross_val_flag: bool = False
):
    """
    No comment in Matlab code
    """
    T = data_tensor.shape[0]
    N = data_tensor.shape[1]
    C = data_tensor.shape[2]

    # (501, 5, 311)
    XN = np.reshape(np.swapaxes(data_tensor, 1, 2).T, (N, -1)).T
    # plt.plot(XN[0:10, 0])
    # plt.plot(XN[0:10, 1])
    # plt.show()

    # plt.plot(XN[:, 0])
    # plt.plot(XN[:, 1])
    # plt.show()

    # now do traditional PCA
    meanXN = np.mean(XN, axis=0)
    cur_pca = PCA()
    cur_pca.fit(XN)  # apply pca to the analyzed times
    maskT1Orig = np.ones((T, 1), dtype=bool)
    maskT1Orig[-1] = 0
    maskT1Orig = np.tile(maskT1Orig, (C, 1))
    maskT1Orig = np.squeeze(maskT1Orig)
    maskT2Orig = np.ones((T, 1), dtype=bool)
    maskT2Orig[0] = 0
    maskT2Orig = np.tile(maskT2Orig, (C, 1))
    maskT2Orig = np.squeeze(maskT2Orig)
    R2 = np.empty((len(model_dim), 1))

    if not cross_val_flag:
        for i in range(len(model_dim)):
            pc_vectors_i = cur_pca.components_[0: model_dim[i], :].T
            # meanXN, XN_red matches
            XN_red = np.dot(XN - meanXN, pc_vectors_i)
            # the masks just give us earlier and later times within each condition
            d_state = XN_red[maskT2Orig, :] - XN_red[maskT1Orig, :]
            # just for convenience, keep the earlier time in its own variable
            pre_state = XN_red[maskT1Orig, :]
            # M takes the state and provides a fit to dState
            M = np.linalg.lstsq(pre_state, d_state, rcond=None)[0].T
            # this is the error in the fit
            fit_error_M = d_state.T - np.dot(M, pre_state.T)
            # original data variance
            var_d_state = (d_state.flatten() ** 2).sum(axis=0)
            # how much is explained by the overall fit via M

            # looks like fit_error_M is wrong
            R2[i] = (
                var_d_state - (fit_error_M.flatten() ** 2).sum(axis=0)
            ) / var_d_state

    else:
        for i in range(len(model_dim)):
            fit_error_M_test = []
            d_state_test = []
            for c in range(C):
                mask_train = np.ones((1, C), dtype=bool)
                mask_train[c] = 0
                mask_train = np.reshape(np.tile(mask_train, (T, 1)), -1, 1)
                XN_train = XN[mask_train, :]
                pc_vectors_i = cur_pca.components_[:, 0: model_dim[i]]
                XN_red_train = np.dot(XN_train - meanXN, pc_vectors_i)

                mask_t1_train = np.ones((T, 1), dtype=bool)
                mask_t1_train[-1] = 0
                mask_t1_train = np.tile(mask_t1_train, (C - 1, 1))

                mask_t2_train = np.ones((T, 1), dtype=bool)
                mask_t2_train[0] = 0
                mask_t2_train = np.tile(mask_t2_train, (C - 1, 1))

                # the masks just give us earlier and later times within each condition
                d_state_train = (
                    XN_red_train[mask_t2_train, :] -
                    XN_red_train[mask_t1_train, :]
                )
                pre_state_train = XN_red_train[
                    mask_t1_train, :
                ]  # just for convenience, keep the earlier time in its own variable
                # M takes the state and provides a fit to dState
                M = np.linalg.lstsq(d_state, pre_state, rcond=None)[0]

                # Test
                mask_test = np.invert(mask_train)
                XN_test = XN[mask_test, :]
                XN_red_test = np.dot(XN_test - meanXN, pc_vectors_i)

                mask_t1_test = np.ones((T, 1), dtype=bool)
                mask_t1_test[-1] = 0

                mask_t2_test = np.ones((T, 1), dtype=bool)
                mask_t2_test[0] = 0

                d_state_test_c = (
                    XN_red_test[mask_t2_test, :] - XN_red_test[mask_t1_test, :]
                )
                pre_state_test_c = XN_red_test[
                    mask_t1_test, :
                ]  # just for convenience, keep the earlier time in its own variable
                d_state_test.append(
                    d_state_test_c
                )  # the masks just give us earlier and later times within each condition
                fit_error_M_test.append(
                    d_state_test_c.T - M * pre_state_test_c.T
                )  # this is the error in the fit

            var_d_state_test_d = np.sum(
                d_state_test.flatten() ** 2, axis=0
            )  # original data variance
            # how much is explained by the overall fit via M
            R2[i] = (
                var_d_state_test_d -
                np.sum((fit_error_M_test.flatten() ** 2), axis=0)
            ) / var_d_state_test_d

    return R2


def check_grad(f: str, x: np.ndarray, e: float, *args):
    """
    checkgrad checks the derivatives in a function, by comparing them to finite
    differences approximations. The partial derivatives and the approximation
    are printed and the norm of the diffrence divided by the norm of the sum is
    returned as an indication of accuracy.

    usage: checkgrad('f', X, e, P1, P2, ...)

    where X is the argument and e is the small perturbation used for the finite
    differences. and the P1, P2, ... are optional additional parameters which
    get passed to f. The function f should be of the type

    [fX, dfX] = f(X, P1, P2, ...)

    where fX is the function value and dfX is a vector of partial derivatives.

    Carl Edward Rasmussen, 2001-08-01.
    """

    # get the function that we want to check
    methodToCall = globals()[f]
    # get the partial derivatives of the function, dy
    y, dy = methodToCall(x, *args)

    dh = np.zeros((len(x), 1))
    for i in range(len(x)):
        dx = np.zeros((len(x), 1))
        dx[i] = dx[i] + e  # perturb a single dimension
        # get the partial derivatives of the function, dy2
        y2, dy2 = methodToCall(x + dx, *args)
        dx = -dx
        # get the partial derivatives of the function, dy1
        y1, dy1 = methodToCall(x + dx, *args)
        dh[i] = (y2 - y1) / (2 * e)  # the relative error
        # lost_norm = np.linalg.norm(dh[0:i] - dy[0:i]) / np.linalg.norm(
        #     dh[0:i] + dy[0:i]
        # )

    d = np.linalg.norm(dh - dy) / np.linalg.norm(dh + dy)
    return d, dy, dh


def diag_kron_sum(Ds: list) -> np.ndarray:
    """
    <TME>
    Copyright (C) 2017 Gamaleldin F. Elsayed and John P. Cunningham
        (see full notice in README)

    [kronSumDs] = diagKronSum(Ds)

    This function evaluates the efficient evaluation of kronecker (tensor)
    sums of diagonal matrices.

    Inputs:
        - Ds: is a cell where each element contains the diagonal elements
            of each matrix.

    Outputs:
        - kronSumLs: is the result of Dn \kronsum Dn-1 .....\kronsum D
    """
    tensor_size = len(Ds)
    last_kron_sum_ds = np.zeros(1)
    for i in range(tensor_size):
        new_kron_sum_ds = (last_kron_sum_ds * np.ones((1, len(Ds[i])))).T + (
            np.ones((len(last_kron_sum_ds), 1)) * Ds[i]
        ).T

        flat_kron_sum_ds = new_kron_sum_ds.flatten()
        flat_kron_sum_ds = np.expand_dims(flat_kron_sum_ds, axis=-1)
        last_kron_sum_ds = flat_kron_sum_ds.copy()

    return flat_kron_sum_ds


def extract_features(
    data_tensor, mean_tensor=None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict, dict]:
    """
    No comment in Matlab code
    """
    T = data_tensor.shape[0]
    N = data_tensor.shape[1]
    C = data_tensor.shape[2]
    M = {"T": [], "TN": [], "TNC": []}
    mu = {}

    meanT = sum_tensor(data_tensor, (1, 2)) / (C * N)
    meanN = sum_tensor(data_tensor, (0, 2)) / (C * T)
    meanC = sum_tensor(data_tensor, (0, 1)) / (N * T)
    mu["T"] = np.vstack(meanT)
    mu["N"] = np.vstack(meanN)
    mu["C"] = np.vstack(meanC)

    if mean_tensor is None:
        # Mean is caluclated by subtracting each reshaping mean
        # There were some other methods in the original code but they were
        # not used in the implementation, so will ignore.

        data_tensor_0 = data_tensor.copy()
        meanT = sum_tensor(data_tensor_0, (1, 2)) / (C * N)
        data_tensor_0 = data_tensor_0 - meanT
        meanN = sum_tensor(data_tensor_0, (0, 2)) / (C * T)
        data_tensor_0 = data_tensor_0 - meanN
        meanC = sum_tensor(data_tensor_0, (0, 1)) / (N * T)
        data_tensor_0 = data_tensor_0 - meanC

        M["TNC"] = data_tensor - data_tensor_0
        M["TN"] = np.tile(
            np.expand_dims(sum_tensor(M["TNC"], 2) / (C), axis=-1), (1, 1, C)
        )
        M["T"] = np.tile(
            np.expand_dims(
                np.tile(
                    np.expand_dims(
                        (sum_tensor(M["TNC"], (1, 2)) / (N * C)), axis=-1), N
                ),
                axis=-1,
            ),
            (1, 1, C),
        )
        meanTensor = M["TNC"]

    # subtract the mean tensor and calculate the covariances
    XT = np.reshape((data_tensor - meanTensor), (T, -1)).T
    XN = np.reshape(
        np.swapaxes(np.swapaxes(
            (data_tensor - meanTensor), 0, 1), 1, 2), (N, -1)
    ).T
    XC = np.reshape(np.swapaxes(data_tensor - meanTensor, 0, 2), (C, -1)).T

    sigma_T = XT.T @ XT
    sigma_N = XN.T @ XN
    sigma_C = XC.T @ XC

    return (sigma_T, sigma_N, sigma_C, M, mu)


def kron_mvprod(As: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    x = kron_mvprod(As, b)

    This function evaluates the efficient multiplication of matrix (A) that
    has kronecker product structure A = kron(An, ...., A2, A1) by a vector
    (b). This is the algorithm one from:
    Scaling multidimensional inference for structured Gaussian processes
    E Gilboa, Y Saatçi, JP Cunningham - Pattern Analysis and Machine
    Intelligence, IEEE ?, 2015

    Inputs:
        - As: is a cell where each element contains the matrices
            A1,A2,...An
        - b: a vector.

    Outputs:
        - x: is the result of A*b
    """
    x = b.copy()
    num_draws = b.shape[1]
    CTN = x.shape[0]
    for d in range(len(As)):
        cur_A = As[d]
        Gd = len(cur_A)
        X = np.reshape(x, (Gd, int((CTN * num_draws) / Gd)), order="F")
        Z = np.dot(cur_A, X)
        Z = Z.T
        x = np.reshape(Z, (CTN, num_draws), order="F")

    x = np.reshape(x, (CTN * num_draws, 1), order="F")
    x = np.reshape(x, (num_draws, CTN), order="F").T
    return x


# def minimize(X, f, length: list, *args):
#     """
#     Minimize a differentiable multivariate function using conjugate gradients.

#     Usage: [X, fX, i] = minimize(X, f, length, P1, P2, P3, ... )

#     X       initial guess; may be of any type, including struct and cell array
#     f       the name or pointer to the function to be minimized. The function
#             f must return two arguments, the value of the function, and it's
#             partial derivatives wrt the elements of X. The partial derivative
#             must have the same type as X.
#     length  length of the run; if it is positive, it gives the maximum number of
#             line searches, if negative its absolute gives the maximum allowed
#             number of function evaluations. Optionally, length can have a second
#             component, which will indicate the reduction in function value to be
#             expected in the first line-search (defaults to 1.0).
#     P1, P2, ... parameters are passed to the function f.

#     X       the returned solution
#     fX      vector of function values indicating progress made
#     i       number of iterations (line searches or function evaluations,
#             depending on the sign of "length") used at termination.

#     The function returns when either its length is up, or if no further progress
#     can be made (ie, we are at a (local) minimum, or so close that due to
#     numerical problems, we cannot get any closer). NOTE: If the function
#     terminates within a few iterations, it could be an indication that the
#     function values and derivatives are not consistent (ie, there may be a bug in
#     the implementation of your "f" function).

#     The Polack-Ribiere flavour of conjugate gradients is used to compute search
#     directions, and a line search using quadratic and cubic polynomial
#     approximations and the Wolfe-Powell stopping criteria is used together with
#     the slope ratio method for guessing initial step sizes. Additionally a bunch
#     of checks are made to make sure that exploration is taking place and that
#     extrapolation will not be unboundedly large.

#     See also: checkgrad

#     Copyright (C) 2001 - 2010 by Carl Edward Rasmussen, 2010-01-03
#     """

#     def my_unwrap(s):
#         # Extract the numerical values from "s" into the column vector "v". The
#         # variable "s" can be of any type, including struct and cell array.
#         # Non-numerical elements are ignored. See also the reverse rewrap.m.
#         v = []

#         # handle depending on the type
#         if type(s) == np.ndarray:
#             s = np.vstack(s)
#             for d in s:
#                 v.append([d])  # numeric values are recast to column vector
#         elif type(s) == float:
#             v.append([d])
#         # elif type(s) == dict:
#         #     v = my_unwrap(s.values()) # alphabetize, conv to cell, recurse
#         elif type(s) == list:
#             # cell array elements are handled sequentially
#             for d in s:
#                 v.append(my_unwrap(s))

#         return np.array(v)

#     def my_rewrap(s, v):
#         # Map the numerical elements in the vector "v" onto the variables "s" which can
#         # be of any type. The number of numerical elements must match; on exit "v"
#         # should be empty. Non-numerical entries are just copied. See also unwrap.m.
#         if type(s) == np.ndarray:
#             if v.size < s.size:
#                 sys.exit("The length of v is too short.")
#                 # error('The vector for conversion contains too few elements')
#             s = np.reshape(v[0 : s.size], s.shape)  # numeric values are reshaped
#             v = v[s.size + 1 : -1]  # remaining arguments passed on

#         elif type(s) == float:
#             s = v[0]

#         elif type(s) == list:
#             s_tmp = []
#             for i in range(s.size):  # cell array elements are handled sequentially
#                 s_tmp.append(my_rewrap(s[i], v))
#             s = s_tmp

#         # other types are not processed
#         # decided not to support dict like objects
#         return s, v

#     verbose = False  # set to true if one want to display iteration by iteration info.
#     INT = 0.1  # don't reevaluate within 0.1 of the limit of the current bracket
#     EXT = 3.0  # extrapolate maximum 3 times the current step-size
#     MAX = 20  # max 20 function evaluations per line search
#     RATIO = 10  # maximum allowed slope ratio
#     SIG = 0.1
#     RHO = SIG / 2  # SIG and RHO are the constants controlling the Wolfe-
#     # Powell conditions. SIG is the maximum allowed absolute ratio between
#     # previous and new slopes (derivatives in the search direction), thus setting
#     # SIG to low (positive) values forces higher precision in the line-searches.
#     # RHO is the minimum allowed fraction of the expected (from the slope at the
#     # initial point in the linesearch). Constants must satisfy 0 < RHO < SIG < 1.
#     # Tuning of SIG (depending on the nature of the function to be optimized) may
#     # speed up the minimization; it is probably not worth playing much with RHO.

#     # The code falls naturally into 3 parts, after the initial line search is
#     # started in the direction of steepest descent. 1) we first enter a while loop
#     # which uses point 1 (p1) and (p2) to compute an extrapolation (p3), until we
#     # have extrapolated far enough (Wolfe-Powell conditions). 2) if necessary, we
#     # enter the second loop which takes p2, p3 and p4 chooses the subinterval
#     # containing a (local) minimum, and interpolates it, unil an acceptable point
#     # is found (Wolfe-Powell conditions). Note, that points are always maintained
#     # in order p0 <= p1 <= p2 < p3 < p4. 3) compute a new search direction using
#     # conjugate gradients (Polack-Ribiere flavour), or revert to steepest if there
#     # was a problem in the previous line-search. Return the best value so far, if
#     # two consecutive line-searches fail, or whenever we run out of function
#     # evaluations or line-searches. During extrapolation, the "f" function may fail
#     # either with an error or returning Nan or Inf, and minimize should handle this
#     # gracefully.]

#     if len(length) == 2:
#         red = length[1]
#         length = length[0]
#     else:
#         red = 1
#         length = length[0]

#     if length > 0:
#         S = "Linesearch"
#     else:
#         S = "Function evaluation"

#     methodToCall = globals()[f]

#     i = 0  # zero the run length counter
#     ls_failed = 0  # no previous line search has failed
#     f0, df0 = methodToCall(X, *args)  # get function value and gradient
#     Z = X
#     X = np.expand_dims(np.squeeze(my_unwrap(X)), axis=1)
#     df0 = np.expand_dims(np.squeeze(my_unwrap(df0)), axis=1)
#     if verbose:
#         print("%s %6i;  Value %4.6e", S, i, f0)

#     fX = f0
#     i = i + int(length < 0)  # count epochs?!
#     s = -df0
#     d0 = (-s.T) @ s  # initial search direction (steepest) and slope
#     x3 = red / (1 - d0)  # initial step is red/(|s|+1)

#     while i < abs(length):  # while not finished
#         i = i + int(length > 0)  # count iterations?!
#         # make a copy of current values
#         X0 = X
#         F0 = f0
#         dF0 = df0

#         if length > 0:  # line search
#             M = MAX
#         else:  # function evaluation
#             M = min(MAX, -length - i)

#         while 1:  # keep extrapolating as long as necessary
#             x2 = 0
#             f2 = f0
#             d2 = d0
#             f3 = f0
#             df3 = df0
#             success = 0

#             while (not success) and (M > 0):
#                 M = M - 1
#                 i = i + int(length < 0)  # count epochs?!
#                 try:
#                     # need to get self-done functions unwrap and rewrap
#                     f3, df3 = methodToCall(my_rewrap(Z, X + x3 * s)[0], *args)
#                     df3 = np.expand_dims(np.squeeze(my_unwrap(df3)), axis=1)
#                     if (
#                         f3 == np.NaN
#                         or f3 == np.Inf
#                         or np.any(np.isnan(df3) + np.isinf(df3))
#                     ):
#                         raise ValueError("Error: f3 is NaN or Inf")
#                     success = 1
#                 except:  # catch any errorwhich occured in f
#                     x3 = (x2 + x3) / 2  # bisect and try again

#             if f3 < F0:  # keep best values
#                 X = X + x3 * s
#                 f0 = f3
#                 df0 = df3
#             # new slope
#             d3 = df3.T @ s
#             # seems like the function breaks too early
#             # f0 does not match
#             if (
#                 (d3 > SIG * d0) or (f3 > f0 + x3 * RHO * d0) or (M == 0)
#             ):  # are we done extrapolating?
#                 break
#             # move point 2 to point 1
#             x1 = x2
#             f1 = f2
#             d1 = d2
#             # move point 3 to point 2
#             x2 = x3
#             f2 = f3
#             d2 = d3

#             # make cubic extrapolation
#             A = 6 * (f1 - f2) + 3 * (d2 + d1) * (x2 - x1)
#             B = 3 * (f2 - f1) - (2 * d1 + d2) * (x2 - x1)
#             # num. error possible, ok!
#             x3 = x1 - (
#                 (d1 * (x2 - x1) ** 2) / (B + np.sqrt(B * B - A * d1 * (x2 - x1)))
#             )
#             # num prob | wrong sign?
#             if (
#                 (not np.all(np.isreal(x3)))
#                 or np.any(np.isnan(x3))
#                 or np.any(np.isinf(x3))
#                 or (x3 < 0)
#             ):
#                 x3 = x2 / EXT  # extrapolate maximum amount
#             # new point beyond extrapolation limit?
#             elif x3 > x2 * EXT:
#                 x3 = x2 * EXT  # extrapolate maximum amount
#             # new point too close to previous point?
#             elif x3 < (x2 + INT * (x2 - x1)):
#                 x3 = x2 + INT * (x2 - x1)
#         # end extrapolation

#         # keep interpolating
#         # d3, f3, f0 don't match shape
#         # x3 is incorrect val
#         while ((abs(d3) > -SIG * d0) or (f3 > f0 + x3 * RHO * d0)) and M > 0:
#             if (d3 > 0) or (f3 > f0 + x3 * RHO * d0):  # choose subinterval
#                 x4 = x3
#                 f4 = f3
#                 d4 = d3  # move point 3 to point 4
#             else:
#                 x2 = x3
#                 f2 = f3
#                 d2 = d3  # move point 3 to point 2

#             # if we're not extrapolating use parabolic extrapolation
#             if f4 > 0:
#                 x3 = x2 - (
#                     (0.5 * d2 * (x4 - x2) ** 2) / (f4 - f2 - d2 * (x4 - x2))
#                 )  # quadratic extrapolation
#             else:
#                 A = 6 * (f2 - f4) / (x4 - x2) + 3 * (d4 + d2)  # cubic extrapolation
#                 B = 3 * (f4 - f2) - (2 * d2 + d4) * (x4 - x2)
#                 x3 = (
#                     x2 + (np.sqrt(B * B - A * d2 * (x4 - x2) ** 2) - B) / A
#                 )  # num. error possible, ok!

#             if np.any(np.isnan(x3)) or np.any(np.isinf(x3)):
#                 x3 = (x2 + x4) / 2  # if we had a numerical problem, then bisect

#             x3 = max(
#                 min(x3, x4 - INT * (x4 - x2)), x2 + INT * (x4 - x2)
#             )  # don't accept too close
#             f3, df3 = methodToCall(my_rewrap(Z, X + x3 * s)[0], *args)
#             df3 = np.expand_dims(np.squeeze(my_unwrap(df3)), axis=1)

#             if f3 < F0:
#                 X0 = X + x3 * s  # keep best values
#                 F0 = f3
#                 dF0 = df3
#             M = M - 1
#             i = i + int(length < 0)  # count epochs?!
#             d3 = df3.T @ s  # new slope
#         # end interpolation

#         # if line search succeeded
#         if abs(d3) < -SIG * d0 and f3 < f0 + x3 * RHO * d0:
#             # perform the update
#             X = X + x3 * s
#             f0 = f3
#             # NOTE: may run into issues with this line
#             fX = np.append(fX.T, f0).T
#             if verbose:
#                 print("%s %6i;  Value %4.6e", S, i, f0)
#             s = (df3.T * df3 - df0.T * df0) / (
#                 df0.T * df0
#             ) * s - df3  # Polack-Ribiere CG direction
#             df0 = df3  # swap derivatives
#             d3 = d0
#             d0 = df0.T @ s
#             if d0 > 0:  # new slope must be negative
#                 s = -1 * df0  # otherwise use steepest direction
#                 d0 = (-1 * s.T) * s
#             x3 = x3 * min(RATIO, d3 / d0)  # slope ratio but max RATIO
#             ls_failed = 0  # this line search did not fail
#         else:
#             # restore best point so far
#             X = X0
#             f0 = F0
#             df0 = dF0
#             # line search failed twice in a row
#             # or we ran out of time, so we give up
#             if ls_failed or i > abs(length):
#                 break
#             s = -1 * df0  # try steepest
#             d0 = (-s.T) @ s
#             x3 = 1 / (1 - d0)
#             ls_failed = 1  # this line search failed

#     X = np.expand_dims(my_rewrap(Z, X)[0], axis=1)
#     if verbose:
#         print()

#     return X, fX, i


def sum_tensor(A: np.ndarray, sumDim: tuple) -> np.ndarray:
    """
    [sumA] = sumTensor(A, sumDim)

    This function evaluates the sum of tensor at specific dimensions

    Inputs:
        - A: is the input n-dimesnional tensor
        - sumDim: the dimensions to sum over.

    Outputs:
        - sumA: an n-dimensional tensor of the sum of tensor A at the
        specified dimensions. The dimensions specified by sumDim will be of
        size 1.
    """
    return np.expand_dims(np.sum(A, axis=sumDim), axis=sumDim)


def calc_TME(
    data, data_sp, cur_freq, num_surrogates=1000, surrogate_type="surrogate-TNC"
):
    # this is where I will put the code from the demo

    # randomize the seed
    np.random.seed(1762)

    # quantify the linear dynamical structure of original data by a summary statistic (R2)
    model_dim = [data.shape[-1]]
    r2_data = summarize_LDS(data, model_dim)
    print(r2_data)
    # quantify primary features of the original data
    target_sigma_T, target_sigma_N, target_sigma_C, M, mu = extract_features(
        data)

    # sample many surrogates and build null distribution of summary statistics
    params = {}
    # params["readout_mode"] = 2  # select readout mode (eg neuron mode)
    # params["shfl_mode"] = 3  # shuffle across tensor mode (eg condition mode)
    # params["fix_mode"] = 2  # shuffle per mode (shuffle for each neuron independently)

    if surrogate_type == "surrogate-T":
        params["margCov"] = [target_sigma_T, [], []]
        params["meanTensor"] = np.squeeze(M["T"])
    elif surrogate_type == "surrogate-TN":
        params["margCov"] = [target_sigma_T, target_sigma_N, []]
        params["meanTensor"] = np.squeeze(M["TN"])
    elif surrogate_type == "surrogate-TNC":
        params["margCov"] = [target_sigma_T, target_sigma_N, target_sigma_C]
        params["meanTensor"] = M["TNC"]
    else:
        raise ValueError(
            'surrogate_type must be one of "surrogate-T", "surrogate-TN", "surrogate-TNC"'
        )

    # fit the maximum entropy distribution
    max_entropy = fit_max_entropy(params)
    r2_surr = np.zeros((num_surrogates, 1))
    data_dims = data.shape
    all_surr_data = np.zeros(
        (num_surrogates, data_dims[0], data_dims[1], data_dims[2]))
    for i in tqdm(range(num_surrogates)):
        # generate TME random surrogate data.
        surr_tensor = sample_TME(max_entropy, num_surrogates)
        r2_surr[i] = summarize_LDS(surr_tensor, model_dim)
        all_surr_data[i, ...] = surr_tensor

    # evaluate a P value
    p = np.mean(r2_data <= r2_surr)  # (upper-tail test)

    if p >= 0.05:
        print("P value = ", p)
    else:
        print(
            "significant, P value < ",
            (int(p < 0.001) * 0.001)
            + (int((p < 0.01) & (p >= 0.001)) * 0.01)
            + (int((p < 0.05) & (p >= 0.01)) * 0.05),
        )

    #  plot null distribution
    x = np.linspace(0, 1, 34)
    # h = np.histogram(r2_surr, bins=x)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(r2_surr, bins=x, density=True, label="Null distribution")
    ax.plot(r2_data, 0, color="black", marker="o",
            label="Real Data", markersize=10)
    ax.set_xlabel("summary statistic (R^2)")
    ax.set_ylabel("count")
    ax.set_xlim([0, 1])
    plt.legend()
    plt.savefig(data_sp + cur_freq + "_null_distrib_histogram.png")
    plt.close()

    return all_surr_data
