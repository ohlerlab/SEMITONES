import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import issparse


def _term_frequency(X, tf_type, threshold=None, K=None):
    if tf_type == "binary":
        if issparse(X):
            X.data[X.data > threshold] = 1
            X.data[np.logical_not(X.data > threshold)] = 0
            X.eliminate_zeros()
        else:
            X[X > threshold], X[np.logical_not(X > threshold)] = 1, 0
        return X
    elif tf_type == "term_frequency":
        if issparse(X):
            return X.multiply(csr_matrix(1 / X.sum(1)))
        else:
            return X / X.sum(1)[:, np.newaxis]
    elif tf_type == "log_norm":
        return np.log1p(X)
    elif tf_type == "k_aug_freq":
        if issparse(X):
            X = X.multiply(csr_matrix(1 / np.amax(X, 1).data).T)
            X = X.multiply(1 - K)
            X.data += K
            return X
        else:
            return (K + (1 - K)) * (X / np.amax(X, 1)[:, np.newaxis])
    return X


def _inverse_document_frequency(X, idf_type):
    if not idf_type == "unary":
        N = X.shape[0]
        if issparse(X):
            X.eliminate_zeros()
            nt = X.getnnz(axis=0)
        else:
            nt = np.count_nonzero(X, 0)
        if idf_type == "inv_doc_freq":
            return np.log2(N / nt)
        elif idf_type == "smooth":
            return np.log2(N / (1 + nt)) + 1
        elif idf_type == "probablistic":
            return np.log2((N - nt) / nt)
    else:
        return 1


def TFIDF(X, tf_type=None, idf_type=None, threshold=None, K=None):
    """This function returns the tf-idf transformed matrix.

    Term frequency and inverse document frequency variants are
    taken from https://en.wikipedia.org/wiki/Tf-idf

    In the inverse document frequencies, N is the number of documents
    (i.e. cells) in the corpus (i.e. dataset) and nt is the number
    of documents (i.e. cells) in which a term (i.e. gene) appears (i.e
    is expressed).

    Parameters
    ----------
    X: matrix-like object (n_samples, n_features)
        An array where the rows are samples (i.e. cells) and the columns
        are features (i.e. genes or peaks). Accepts pandas dataframes,
        numpy arrays, and scipy compressed sparse row matrix.
        are features (i.e. genes or peaks). Accepts pandas dataframes,
        numpy arrays, and scipy compressed sparse row matrix.
    tf_type: str
       The name of the term frequency weight type, choose from:
           - "raw": raw counts.
           - "binary": binarized counts using a threshold provided by
                       the "threshold" parameter.
           - "term_frequency": raw counts corrected by document length.
           - "log_norm": log-normalized counts.
           - "k_aug_freq": counts corrected for the maximum count value,
                           prevents bias towards longer documents, using
                           the parameter "k" in
                           K + (1 - K) * (frequency / max raw frequency).
    idf_type: str
        The name of the inverse document frequency type, choose from:
            - "unary": set the idf to 1
            - "inv_doc_freq": log2(N / nt)
            - "smooth": log2(N / (1 + nt)) + 1
            - "probablistic": log2((N - nt) / nt)
    threshold: float
        The binarization cut-off to use if tf_type == "binary". Defaults
        to 1 if tf_type == "binary" and no threhsold is provided.
    k: float
        The value of "K" to use if tf_type == "k_aug_freq". Defaults to
        0.5.

    Returns
    -------
    tfidf: matrix-like object (n_samples, n_features)
        A tfidf transformed version of matrix X of shape X.shape."""

    valid_tf = ["raw", "binary", "term_frequency", "log_norm", "k_aug_freq"]
    valid_idf = ["unary", "inv_doc_freq", "smooth", "probablistic"]

    if tf_type is None:
        tf_type = "term_frequency"
    if idf_type is None:
        idf_type = "inv_doc_freq"
    if (tf_type == "binary") and (threshold is None):
        print("No binarization threshold provided." +
              " Setting the threshold to the default of 0.")
        threshold = 0
    if (tf_type == "k_aug_freq") and (K is None):
        print("No value for K was provided." +
              " Setting the value of K to the default of 0.5.")
        K = 0.5
    if tf_type not in valid_tf:
        print("{0} is not a valid string for tf_type.".format(tf_type) +
              " Please choose from {0}".format(valid_tf))
        return None
    if idf_type not in valid_idf:
        print("{0} is not a valid string for idf_type.".format(idf_type) +
              " Please choose from {0}".format(valid_idf))
        return None

    X = X.copy()

    if (issparse(X) is True) and (X.getformat() not in ["csr", "csc"]):
        print("If providing a scipy sparse matrix please use the compressed" +
              "sparse row (CSR) format." +
              "Transforming X to scipy.sparse.csr_matrix()")
        X = csr_matrix(X)
    if isinstance(X, pd.DataFrame):
        X = X.values

    idf = _inverse_document_frequency(X, idf_type)
    X = _term_frequency(X, tf_type, threshold, K)

    if issparse(X):
        return X.multiply(idf).tocsr()
    else:
        return X * idf
