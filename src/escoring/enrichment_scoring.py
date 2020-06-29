import gc
import itertools
import multiprocessing as mp
from math import ceil

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import issparse

from escoring._utils import _chunk_indices
from escoring._utils import _linreg_get_beta
from escoring._utils import _permute
from escoring._utils import _std_sparse
from escoring.support_funcs import pairwise_similarities

gc.enable()


def _enrichment_scoring(X, S, scale_exp, i, n_chunks=None):
    """Perform the actual enrichment scoring"""
    n_chunks = 100 if n_chunks is None else n_chunks

    if len(S.shape) == 2:
        params = {}
        for c in range(S.shape[1]):
            if issparse(X):
                pars = []
                for chunk in _chunk_indices(X, n_chunks, axis=1):
                    pars.extend(np.apply_along_axis(_linreg_get_beta, 0,
                                                    X[:, chunk].A, S[:, c],
                                                    scale_exp))
                params[c] = pars
            else:
                params[c] = np.apply_along_axis(_linreg_get_beta, 0, X,
                                                S[:, c], scale_exp)
    else:
        if issparse(X):
            params = []
            for chunk in _chunk_indices(X, n_chunks, axis=1):
                params.extend(np.apply_along_axis(_linreg_get_beta, 0,
                                                  X[:, chunk].A, S, scale_exp))
        else:
            params = np.apply_along_axis(_linreg_get_beta, 0, X, S, scale_exp)
    return i, params


def calculate_escores(X, query, metric=None, S=None, scale_exp=None,
                      optim_over=None, ncpu=None, n_chunks=None,
                      make_copy=None):
    """Calculate the gene/orc enrichment scores for all genes in X with
    respect to the cells in the query.

    X: matrix-like object (n_samples, n_features)
        An array where the rows are samples (i.e. cells) and the columns
        are features (i.e. genes). Accepts pandas dataframes,
        numpy arrays. If
        binarize is False, X should be log-normalized.
    query: list-like object
        An iterable which contains the names or indices of the cells
        with respect to which enrichment scoring should be performed. If
        providing a pandas dataframe, these should be strings. If
        providing a numpy array or sparse matrix, these should be
        indices (int).
    metric: str
        The metric to use for the similarity calculation. Available
        metrics can be found in the sklearn.metrics.pairwise_distances
        and sklearn.metrics.pairwise.pairwise_kernels modules.
    S: matrix-like object (n_samples, n_features), optional
        If desired, one can also provide a similarity matrix manually,
        in which columns are the reference cells (query) and rows
        are all cells in the data (X).
    scale_exp: boolean
        Whether to scale the expression vector before performing
        linear regression.
    ncpu: int
        Number of CPUs the use when using parallel processing. Defaults
        to the maximum number of CPUs available.
    optim_over: str
        Pick from "cols" or "rows".
    n_chunks: int
        How many chuncks to devide the genes or open chromatin regions
        into when using a sparse matrix. If the memory limit is reached
        when running the function, use more chunks. Defaults to the
        n_cols * 0.01 rounded up to first integer.
    """

    # set the default parameters
    metric = "cosine" if metric is None else metric
    scale_exp = True if scale_exp is None else scale_exp
    if optim_over is None:
        if X.shape[1] > len(query):
            optim_over = "cols"
        else:
            optim_over = "rows"
    if n_chunks is None:
        n_chunks = ceil(X.shape[1] * 0.01) if n_chunks is None else n_chunks
    ncpu = mp.cpu_count() if ncpu is None else ncpu
    make_copy = True if make_copy is None else make_copy

    if make_copy is True:
        X = X.copy()

    if isinstance(X, pd.DataFrame):
        query = [X.index.get_loc(i) for i in query]
        cells, genes = X.index, X.columns
        X = X.values

    if S is None:
        print("Calculating pairwise similarities")
        S = pairwise_similarities(X, query, metric)
    else:
        S = S

    if ncpu > 1:
        print("Start enrichment scoring using {0} CPUs".format(ncpu))
        print("Creating process pool")
        with mp.Pool(processes=ncpu) as pool:
            if optim_over == "cols":
                i_chunks = _chunk_indices(X, n=ncpu, axis=1)
                mpres = [pool.apply_async(_enrichment_scoring,
                                          args=(X[:, i], S, scale_exp, i,
                                                n_chunks)) for i in i_chunks]
            else:
                i_chunks = _chunk_indices(S, n=ncpu, axis=1)
                mpres = [pool.apply_async(_enrichment_scoring,
                                          args=(X, S[:, i], scale_exp, i,
                                                n_chunks)) for i in i_chunks]
            print("Run enrichment scoring")
            mpres = [r.get() for r in mpres]
            pool.close()
            pool.join()
            print("Enrichment scoring complete")
    else:
        print("Start enrichment scoring")
        i_chunks = _chunk_indices(X, n=2, axis=1)
        mpres = [_enrichment_scoring(X[:, i], S, scale_exp, i, n_chunks)
                 for i in i_chunks]
        print("Enrichment scoring complete")

    if "cells" in locals():
        rows = [list(mpres[i][0]) for i in len(len(mpres))]
        rows = genes[itertools.chain.from_iterable(rows)]
    else:
        rows = [list(mpres[i][0]) for i in range(len(mpres))]
        rows = list(itertools.chain.from_iterable(rows))
    cols = query

    scores = [pd.DataFrame(mpres[i][1]) for i in range(len(mpres))]
    if (optim_over == "rows") and (ncpu > 1):
        scores = pd.concat(scores, axis=1)
        if "genes" in locals():
            scores.index, scores.columns = genes, cols
        else:
            scores.columns = cols
    else:
        scores = pd.concat(scores, axis=0)
        scores.index, scores.columns = rows, cols

    return scores


def permute(X, n=None, axis=None, seed=None):
    """Permute the dataframe X n times (defaults to 100 times). Seed for
    reproducibility (defaults to 42)."""

    n = 100 if n is None else n
    seed = 42 if seed is None else seed
    axis = 0 if axis is None else axis

    return _permute(X, n=n, axis=axis, seed=seed)


def sig_interval(pscores, n_sds, query=None):
    """Return the signficance cut-off at n SDs above and below the
    mean as the significance cut-off as a dictionary where keys are
    the query cells and values are (lower cut-off, upper cut-off).

    pscores: matrix-like
        A matrix containing the permuted cell x feature matrix obtained
        through the permute() function.
    n_sds: int
        The number of SDs away from the mean at which a score is to
        be called significant. Defaults to 5.
    query: list-like
        A list of reference cells corresponding to the columns in a
        sparse matrix. If using the function calculate_set_perm_scores
        this is the second output.
    """

    n_sds = 5 if n_sds is None else n_sds

    if issparse(pscores):
        if query is None:
            print("Outputting cut-offs in order. Please provide a" +
                  " query in order if you want to use labels as keys.")
        if pscores.getformat() not in ["csr", "csc"]:
            pscores = pscores.to_csr()
        mu = np.array(pscores.mean(axis=0))
        std = _std_sparse(pscores, axis=0, ddof=1)
    else:
        mu, std = np.mean(pscores, axis=0), np.std(pscores, axis=0, ddof=1)

    if not issparse(pscores):
        query = pscores.columns
    else:
        query = list(range(pscores.shape[1]))

    return dict(zip(query, zip(mu - std * n_sds, mu + std * n_sds)))


def _min_set(X, sets, i=None):
    """Return the min value of each set as expression value"""
    if issparse(X):
        return i, csr_matrix([np.amin(X[:, s].A, 1) for s in sets]).T
    else:
        return i, csr_matrix([np.amin(X[:, s], 1) for s in sets]).T


def _max_set(X, sets, i=None):
    """Return the max value of each set as expression value"""
    if issparse(X):
        return i, csr_matrix([np.amax(X[:, s].A, 1) for s in sets]).T
    else:
        return i, csr_matrix([np.amax(X[:, s], 1) for s in sets]).T


def _median_set(X, sets, i=None):
    """Return the median value of each set as expression value"""
    if issparse(X):
        return i, csr_matrix([np.median(X[:, s].A, 1) for s in sets]).T
    else:
        return i, csr_matrix([np.median(X[:, s], 1) for s in sets]).T


def _interaction_set(X, sets, i=None):
    """Return the expression product of each set as expression value"""
    if issparse(X):
        return i, csr_matrix([np.prod(X[:, s].A, 1) for s in sets]).T
    else:
        return i, csr_matrix([np.prod(X[:, s], 1) for s in sets]).T


def _binary_set(X, sets, i=None):
    if issparse(X):
        gse = csr_matrix([np.sum(X[:, s].A, 1) for s in sets]).T
    else:
        gse = csr_matrix([np.sum(X[:, s], 1) for s in sets]).T
    gse[gse > 1] = 1
    return i, gse


def feature_set_values(X, sets, combtype):
    """Combines feature values for all elements in a set."""

    if (issparse(X)) and (X.getformat() not in ["csr", "csc"]):
        X = X.tocsr()

    print("Constructing expression set")
    if combtype == "min":
        return _min_set(X, sets)[1]
    elif combtype == "max":
        return _max_set(X, sets)[1]
    elif combtype == "median":
        return _median_set(X, sets)[1]
    elif combtype == "interaction":
        return _interaction_set(X, sets)[1]
    elif combtype == "binary":
        return _binary_set(X, sets)[1]
