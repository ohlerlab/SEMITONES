from itertools import chain
from itertools import islice
from itertools import repeat
from math import ceil

import numpy as np
from scipy.sparse import issparse
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import scale

kernels = ["linear", "poly", "polynomial", "rbf", "laplacian", "sigmoid",
           "cosine"]
oneminus = ["braycurtis", "correlation", "dice", "jaccard", "kulsinksi",
            "rogerstanimoto", "russelrao", "rbf", "chi2", "laplacian",
            "sigmoid"]


def _knn_dists(X, metric=None, n_neighbors=None, p=None,
               metric_params=None):
    """Compute the Jaccard distance over the kNN graph.
    The metric parameter can be used to specify which metric
    is used to construct the kNN graph."""

    n_neighbors = 5 if n_neighbors is None else n_neighbors
    metric = "euclidean" if metric is None else metric

    # get the kNN graph
    knn_graph = kneighbors_graph(X, n_neighbors, mode="distance",
                                 metric=metric, p=p,
                                 metric_params=metric_params).toarray()

    return _similarities(knn_graph, metric="jaccard")


def _distances(X1, X2=None, metric=None, metric_params=None):
    """Calls sklearn.pairwise.pairwise_distances or
    sklearn.pairwise_pairwise_kernels and returns the distance
    between X1 and X2."""

    metric = "euclidean" if metric is None else metric

    if metric in kernels:
        if metric == "cosine":
            return pairwise_distances(X1, X2, metric="cosine")
        else:
            if metric_params is None:
                S = pairwise_kernels(X1, X2, metric)
            else:
                S = pairwise_kernels(X1, X2, metric, **metric_params)
            if metric == "additive_chi2":
                return - 1 * S
            else:
                return np.max(S) - S
    elif metric == "knn_jaccard":
        S = _similarities(X1, X2, metric="knn_jaccard",
                          **metric_params)
        return 1 - S
    else:
        return pairwise_distances(X=X1, Y=X2, metric=metric)


def _similarities(X1, X2=None, metric=None, knn_metric=None, n_neighbors=None,
                  sym=None, p=None, metric_params=None):
    """Calls sklearn.pairwise.pairwise_distances or
    sklearn.pairwise_pairwise_kernels and returns the similarity
    between X1 and X2.

    n_neighbors, sym, p, and metric_params are only for knn_metrics."""

    metric = "euclidean" if metric is None else metric
    knn_metric = "euclidean" if knn_metric is None else knn_metric

    if metric in kernels:
        if metric_params is None:
            return pairwise_kernels(X1, X2, metric)
        else:
            return pairwise_kernels(X1, X2, metric, **metric_params)
    elif metric == "knn_jaccard":
        if X2 is None:
            return _knn_dists(X1, method="jaccard", metric=knn_metric,
                              n_neighbors=n_neighbors, sym=sym, p=p,
                              metric_params=metric_params)
        else:
            print("Not implemented for two matrices")
            return None
    else:
        D = pairwise_distances(X1, X2, metric)
        if metric in oneminus:
            return 1 - D
        else:
            return 1 / (1 + D)


def _permute(X, n=None, axis=None, seed=None):
    """Permute a frame n times along a given axis."""

    X = X.copy()

    if (issparse(X)) and (X.getformat() not in ["csr", "csc"]):
        X = X.tocsr()

    n = 100 if n is None else n
    axis = 0 if axis is None else axis
    seed = 42 if seed is None else seed

    np.random.seed(seed)

    indices = np.random.permutation(X.shape[axis])
    P = X[:, indices] if axis == 1 else X[indices, :]
    for _ in repeat(None, n - 1):
        indices = np.random.permutation(indices)
        P = P[:, indices] if axis == 1 else X[indices, :]

    return P


def _linreg_get_beta(x, y, scale_exp):
    """Use Scipy linregress to get the regression coeffiecient."""
    from scipy.stats import linregress

    if scale_exp is True:
        x = scale(x)

    return linregress(x, y)[0]


def _chunk_indices(X, n, axis=None):
    """A generator to return n chunks of an array."""
    axis = 0 if axis is None else axis

    if (axis != 0) and (axis != 1):
        print("Please provide a valid axis (0 or 1)")

    length = X.shape[0] if axis == 0 else X.shape[1]

    size = ceil(length / n)
    for i in range(0, length, size):
        yield range(length)[i:i + size]


def _make_generator(iterable):
    for i in iterable:
        yield i


def _chunk_generator(generator, size=None):
    for g in generator:
        yield chain([g], islice(generator, size - 1))


def _std_sparse(X, axis=None, ddof=None):
    axis = 0 if axis is None else axis
    ddof = 0 if ddof is None else ddof

    def _variance(array):
        N = len(array)
        return 1 / (N - ddof) * (np.sum(np.abs(array - array.mean()) ** 2))

    if axis == 0:
        c = X.shape[1]
        var = np.array([_variance(X[:, i].data) for i in range(c)])
        return np.sqrt(var)
    else:
        c = X.shape[0]
        var = np.array([_variance(X[i, :].data) for i in range(c)])
        return np.sqrt(var)
