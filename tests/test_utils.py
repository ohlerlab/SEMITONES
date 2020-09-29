import inspect

import numpy as np
from scipy.sparse import csr_matrix

from SEMITONES._utils import _chunk_generator
from SEMITONES._utils import _chunk_indices
from SEMITONES._utils import _distances
from SEMITONES._utils import _knn_dists
from SEMITONES._utils import _linreg_get_beta
from SEMITONES._utils import _make_generator
from SEMITONES._utils import _permute
from SEMITONES._utils import _similarities
from SEMITONES._utils import _std_sparse


def test_knn_dists():

    minitest = np.array([
        [0.68174634, 0.09686306, 0.10710519, 0.19573925, 0.40389495,
         0.15040522, 0.55552467, 0.09655868, 0.68762446, 0.29262945],
        [0.13217678, 0.90647729, 0.99156602, 0.97288113, 0.77431692,
         0.73174956, 0.02167488, 0.13749407, 0.82368798, 0.25286906],
        [0.03524563, 0.88047327, 0.11248924, 0.71070194, 0.77642191,
         0.88408589, 0.79397867, 0.25119032, 0.12547259, 0.17557831],
        [0.27156203, 0.37688026, 0.27905244, 0.59463537, 0.52840277,
         0.17076835, 0.64341349, 0.07799788, 0.75185855, 0.90018081],
        [0.70656525, 0.98731994, 0.38704339, 0.27097026, 0.86827684,
         0.18572232, 0.67162928, 0.52089563, 0.3823818, 0.65332658]])

    true = np.array([[1., 0.33333333, 1, 0.33333333, 0.33333333],
                     [0.33333333, 1., 0.33333333, 0, 0.33333333],
                     [1, 0.33333333, 1., 0.33333333, 0.33333333],
                     [0.33333333, 0, 0.33333333, 1., 0.33333333],
                     [0.33333333, 0.33333333, 0.33333333, 0.33333333, 1.]])

    test = _knn_dists(minitest, n_neighbors=2)

    np.testing.assert_allclose(test, true)


def test_distances():

    from sklearn.metrics import pairwise_distances
    from sklearn.metrics.pairwise import pairwise_kernels

    minitest = np.random.random((5, 1))
    distmets = ["euclidean", "cosine", "jaccard"]
    for metric in distmets:
        d1 = pairwise_distances(minitest, metric=metric)
        d2 = _distances(minitest, metric=metric)
        np.testing.assert_allclose(d1, d2)
    kernels = ["linear", "rbf", "laplacian"]
    for metric in kernels:
        d1 = pairwise_kernels(minitest, metric=metric)
        if metric == "rbf":
            d1 = 1 - d1
        else:
            d1 = np.max(d1) - d1
        d2 = _distances(minitest, metric=metric)
        np.testing.assert_allclose(d1, d2)


def test_similarities():

    from sklearn.metrics.pairwise import pairwise_distances
    from sklearn.metrics.pairwise import pairwise_kernels

    minitest = np.random.random((5, 1))
    distmets = ["euclidean",
                "cosine",
                "jaccard"]
    for metric in distmets:
        if not metric == "cosine":
            d1 = pairwise_distances(minitest, metric=metric)
            if metric == "euclidean":
                s1 = 1 / (1 + d1)
            if metric == "jaccard":
                s1 = 1 - d1
        else:
            s1 = pairwise_kernels(minitest, metric=metric)
        s2 = _similarities(minitest, metric=metric)
        np.testing.assert_allclose(s1, s2)
    kernels = ["linear", "rbf", "laplacian"]
    for metric in kernels:
        s1 = pairwise_kernels(minitest, metric=metric)
        s2 = _similarities(minitest, metric=metric)
        np.testing.assert_allclose(s1, s2)


def test_permute():

    minitest = np.random.random((5, 10))

    P = _permute(minitest)

    np.testing.assert_raises(AssertionError, np.testing.assert_allclose,
                             minitest, P)
    assert P.shape == minitest.shape


def test_linreg():

    x, y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    assert _linreg_get_beta(x, y, scale_exp=False) == 1.0
    assert _linreg_get_beta(x, y, scale_exp=True) > 1.0


def test_chunk_indices():

    minitest = np.zeros((50, 10))
    indx1 = _chunk_indices(minitest, 2, axis=0)
    test1 = []
    for idxs in indx1:
        for i in idxs:
            test1.append(i)
    indx2 = _chunk_indices(minitest, 2, axis=1)
    test2 = []
    for idxs in indx2:
        for i in idxs:
            test2.append(i)

    assert list(range(minitest.shape[0])) == test1
    assert list(range(minitest.shape[1])) == test2


def test_make_generator():
    assert inspect.isgeneratorfunction(_make_generator)


def test_chunk_generator():

    x = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]
    z = _make_generator(x)
    z = _chunk_generator(z, size=5)
    z = [list(i) for i in z]
    y = []
    for vals in z:
        for i in vals:
            y.append(i)

    assert x == y


def test_std_sparse_cols():

    minitest = np.random.random((50, 10))
    minitest_sparse = csr_matrix(minitest)
    std1 = _std_sparse(minitest_sparse, axis=0, ddof=0)
    std2 = np.std(minitest, axis=0, ddof=0)

    np.testing.assert_allclose(std1, std2)


def test_std_sprase_rows():

    minitest = np.random.random((50, 10))
    minitest_sparse = csr_matrix(minitest)
    std1 = _std_sparse(minitest_sparse, axis=1, ddof=0)
    std2 = np.std(minitest, axis=1, ddof=0)

    np.testing.assert_allclose(std1, std2)
