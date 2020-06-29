from itertools import combinations

import numpy as np
import pandas as pd
import tables
from scipy import sparse
from scipy.sparse import issparse

from escoring._utils import _similarities


def binarize(X, t=None):
    """Binarizes data so that values > t are 1."""
    t = 0 if t is None else t

    X = X.copy()

    if issparse(X):
        condition = X.data > t
        X.data[condition] = 1
        X.data[np.logical_not(condition)] = 0
    else:
        X[X > t], X[np.logical_not(X > t)] = 1, 0
    return X


def pairwise_similarities(X, query, metric=None, pca=None, n_pcs=None,
                          n_neighbors=None, metric_params=None):
    """Calls sklearn.pairwise.pairwise_distances or
    sklearn.pairwise_pairwise_kernels and returns the similarity
    between X1 and X2."""

    metric = "euclidean" if metric is None else metric

    if isinstance(X, pd.DataFrame):
        query = [X.index.get_loc(i) for i in query]
        X = X.values

    if "knn" in metric:
        S = _similarities(X, metric=metric, n_neighbors=n_neighbors)
        return S[:, query]
    else:
        return _similarities(X, X[query, :], metric=metric,
                             metric_params=metric_params)


def sig_bool(scores, cutoffs):
    """Returns a boolean frame to indicate significant scores."""
    scores = scores.copy().values
    sigframe = np.empty(scores.shape)
    for i, k in enumerate(cutoffs.keys()):
        sigframe[:, i] = ((scores[:, i] < cutoffs[k][0]) |
                          (scores[:, i] > cutoffs[k][1]))
    return sigframe.astype(bool)


def sig_dictionary(scores, cutoffs, retrieve=None, sign=None):
    """Retrieve either the genes expressed in a certain cell
    or the cells in which a given feature is signficantly expressed.

    Parameters
    ----------
    scores: pandas dataframe
         A dataframe of enrichment scores where columns are cells
         and rows are genes.
    cutoffs: dict
         A dictionary of maximum and miniumum signficance cut-offs
         as provided by the function enrichment_scoring.sig_interval()
    retrieve: str
        Can be either "rows" or "cols". If "rows" the genes which
        are signficant in a given reference cell are returned. If
        "cols" the cells in which a given gene are enriched are
        returned.
    sign: str
        Can be either "positive", "negative" or "both". If "positive"
        selection is only performed for enriched genes. If "negative"
        selected is only performed for depleted genes. If "both",
        both are returned.

    Returns
    -------
    A dictionary where keys are either cells (if retrieve == "rows")
    or features (if retrieve == "cols") and values are either features
    (if retrieve == "rows") or cells (if retrieve == "cols").
    """

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
    """If sigs is a dictionary of {cell: signifcant_genes}"""
    get_list = False if get_list is None else get_list

    sets = {}
    for k, v in sigs.items():
        if get_list is True:
            sets[k] = list(combinations(v, n))
        else:
            sets[k] = combinations(v, n)
    return sets


def load_sparse_h5(node, filename):
    """from harryscholes @ stackoverflow"""
    with tables.open_file(filename) as f:
        attrs = []
        for attr in ("data", "indices", "indptr", "shape"):
            attrs.append(getattr(f.root, "{0}_{1}".format(node, attr)).read())

    X = sparse.csr_matrix(tuple(attrs[:3]), shape=attrs[3])

    return X


def save_sparse_h5(X, node, filename):
    """from harryscholes @ stackoverflow"""
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
