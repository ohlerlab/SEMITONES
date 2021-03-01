from itertools import combinations

import numpy as np
import pandas as pd
import tables
from scipy import sparse
from scipy.sparse import issparse

from SEMITONES._utils import _similarities


def binarize(X, t=None):
    """A function to binarize a matrix-like object.

    Parameters
    ----------
    X: matrix-like (n_samples, n_features)
        A matrix-like numpy ndarray scipy compressed
        sparse row matrix.
    t: float
        The value above which an entry is set to 1
        (x > t = 1, else 0)

    Returns
    -------
    A binarized copy of X."""

    t = 0 if t is None else t

    X = X.copy()

    if issparse(X):
        condition = X.data > t
        X.data[condition] = 1
        X.data[np.logical_not(condition)] = 0
    else:
        X[X > t], X[np.logical_not(X > t)] = 1, 0
    return X


def pairwise_similarities(X, query, metric=None, knn_metric=None,
                          n_neighbors=None, p=None, metric_params=None):
    """Returns the similarity matrix of every sample entry in
       X and the sample entries in the query.

    Parameters
    ----------
    X: matrix-like (n_samples, n_features)
        A numpy ndarray, pandas dataframe or scipy
        compressed sparse row matrix.
    query: list-like
        A list of row identifiers.
    metric: str
        The metric to be used for the distance calculations.
        Any metric from sklearn.metrics.pairwise_distances or
        sklearn.metrics.pairwise.pairwise_kernels is
        available. Alternatively, “knn_jaccard” is implemented,
        where the jaccard similarity over the kNN-graph is
        calculated.
    knn_metric: str
        If using "knn_jaccard", the metric to use for knn-graph
        construction using sklearn.neighbors.kneighbors_graph.
    n_neighbours: int
        If using “knn_jaccard” the number of nearest neighbours
        used for kNN-graph construction.
    p: int
        If using "knn_jaccard" the power parameter to pass
        to sklearn.neighbors.kneighbors_graph.
    metric_params: dict
        A dictionary of arguments to pass to the sklearn
        sklearn.metrics.pairwise_distances or
        sklearn.metrics.pairwise.pairwise_kernels.

    Returns
    -------
        A matrix (n_samples, query) of pairwise similarities
        between all samples and the query samples."""

    metric = "euclidean" if metric is None else metric

    if isinstance(X, pd.DataFrame):
        query = [X.index.get_loc(i) for i in query]
        X = X.values

    if "knn" in metric:
        S = _similarities(X, metric=metric, knn_metric=knn_metric,
                          n_neighbors=n_neighbors,
                          p=p, metric_params=metric_params)
        return S[:, query]
    else:
        return _similarities(X, X[query, :], metric=metric,
                             metric_params=metric_params)


def sig_bool(scores, cutoffs):
    """Returns a boolean copy of the enrichment score
    dataframe where True means a feature (row) is
    significantly enriched in a reference cell (column).

    scores: pandas dataframe
        A pandas dataframe of enrichment scores obtained
        from enrichment_scoring.calculate_escores().
    cutoffs: dict
        A dictionary of significance cut-offs obtained
        from enrichment_scoring.sig_interval().

    Returns
    -------
        A boolean dataframe with each entry indicating
        significance (True) or non-significance (False)."""

    scores = scores.copy().values
    sigframe = np.empty(scores.shape)
    for i, k in enumerate(cutoffs.keys()):
        sigframe[:, i] = ((scores[:, i] < cutoffs[k][0]) |
                          (scores[:, i] > cutoffs[k][1]))
    return sigframe.astype(bool)


def sig_dictionary(scores, cutoffs, retrieve=None, sign=None):
    """Retrieve either the genes that are significantly
    enriched in a certain cell (retrieve=”rows”) or
    the cells in which a feature is significantly
    enriched (retrieve=”cols”).

    Parameters
    ----------
    scores: pandas dataframe
        A pandas dataframe of enrichment scores
        obtained from 
        enrichment_scoring.calculate_escores().   
    cutoffs: dict
         A dictionary of significance cut-offs
         obtained from
         enrichment_scoring.sig_interval()
    retrieve: "rows" or "cols"
        If “rows”, a dictionary where keys are cells
        and values are the features that are
        significantly enriched in this cell are
        returned. If “cols”, the keys are features
        and the values are the cells in which these
        features are significantly enriched.
    sign: "positive”, “negative”, or “both”
        If “positive”, only positive enrichment scores
        will be considered as significant. If “negative”,
        only negative enrichment scores will be
        considered as significant. If “both”, both positive
        and negative enrichment scores will be considered.

    Returns
    -------
    A dictionary of {cell: enriched features} if retrieve
    is “rows” or {feature: cells} if retrieve is “cols”."""

    retrieve = "rows" if retrieve is None else retrieve
    if retrieve == "cols":
        sign = "positive" if sign is None else sign
    else:
        sign = "both" if sign is None else sign

    X = scores.copy()
    sigs = dict()
    if retrieve == "rows":
        for k in cutoffs.keys():
            if sign == "both":
                c1 = (X.loc[:, k] < cutoffs[k][0])
                c2 = (X.loc[:, k] > cutoffs[k][1])
                sigs[k] = X.loc[:, k][c1 | c2].index.tolist()
            else:
                if sign == "positive":
                    c = (X.loc[:, k] > cutoffs[k][1])
                else:
                    c = (X.loc[:, k] < cutoffs[k][0])
                sigs[k] = X.loc[:, k][c].index.tolist()

    else:
        for r in X.index:
            if sign == "positive":
                t = [cutoffs[k][1] for k in cutoffs.keys()]
                sigs[r] = X.columns[np.where(X.loc[r, :] > t)[0]]
            elif sign == "negative":
                t = [cutoffs[k][0] for k in cutoffs.keys()]
                sigs[r] = X.columns[np.where(X.loc[r, :] < t)[0]]
            else:
                t1 = [cutoffs[k][1] for k in cutoffs.keys()]
                t2 = [cutoffs[k][0] for k in cutoffs.keys()]
                c1 = X.columns[np.where(X.loc[r, :] > t1)[0]]
                c2 = X.columns[np.where(X.loc[r, :] < t2)[0]]
                sigs[r] = list(c1) + list(c2)

    return sigs


def get_sets(sigs, n, get_list=None):
    """Get a dictionary of all possible n-element sets
    from all features in “sigs”.

    Parameters
    ----------
    sigs: dictionary
        A dictionary of {cell: iterable} where the
        iterable is an iterable of the elements from
        which sets should be constructed. Could be the
        output of sig_dictionary() if retrieve=”rows”
        was used.
    n: int
        The number of elements in each set.
    get_list: boolean
        Whether to return the sets iterable as a list
        or not.

    Returns
    -------
    A dictionary of {cell: sets} for all keys (cells)
    in sigs."""

    get_list = False if get_list is None else get_list

    sets = {}
    for k, v in sigs.items():
        if get_list is True:
            sets[k] = list(combinations(v, n))
        else:
            sets[k] = combinations(v, n)
    return sets


def load_sparse_h5(node, filename):
    """Load a scipy compressed sparse row (CSR)
     matrix saved in the hdf5 format.

    Parameters
    ----------
    node: str
        The node identifier used to save the
        scipy CSR matrix.
    filename: str
        The path to the location of the file.

    Returns
    -------
    A CSR matrix. 

    Adapted from harryscholes @ stackoverflow"""

    with tables.open_file(filename) as f:
        attrs = []
        for attr in ("data", "indices", "indptr", "shape"):
            attrs.append(getattr(f.root, "{0}_{1}".format(node, attr)).read())

    X = sparse.csr_matrix(tuple(attrs[:3]), shape=attrs[3])

    return X


def save_sparse_h5(X, node, filename):
    """Save a scipy sparse CSR matrix.

    Parameters
    ----------
    X: scipy CSR matrix
        -
    node: str
        The node identifier of the
        matrix in the hdf5 file.
    filename: str
        The path to the location of the file.

    Returns
    -------
        -

    Adapted from harryscholes @ stackoverflow"""

    assert(X.__class__ == sparse.csr.csr_matrix)
    with tables.open_file(filename, "a") as f:
        for attr in ("data", "indices", "indptr", "shape"):
            attr_name = "{0}_{1}".format(node, attr)

            try:
                n = getattr(f.root, attr_name)
                n._f_remove()
            except AttributeError:
                pass

            arr = np.array(getattr(X, attr))
            atom = tables.Atom.from_dtype(arr.dtype)
            ds = f.create_carray(f.root, attr_name, atom, arr.shape)
            ds[:] = arr
