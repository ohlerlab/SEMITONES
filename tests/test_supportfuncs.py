import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_kernels

from SEMITONES.support_funcs import binarize
from SEMITONES.support_funcs import pairwise_similarities
from SEMITONES.support_funcs import sig_bool
from SEMITONES.support_funcs import sig_dictionary


def test_binarize():

    minitest = np.zeros((5, 10))
    rows, cols = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4], range(0, 10)
    for i, j in zip(rows, cols):
        minitest[i, j] = 1

    test = binarize(minitest, t=0)

    np.testing.assert_array_equal(test, minitest)

    test = binarize(csr_matrix(minitest), t=0).toarray()

    np.testing.assert_array_equal(test, minitest)


def test_pairwise_similarities():

    minitest = np.zeros((10, 5))
    query = [0, 1, 2, 3, 4]
    y = minitest[query, :]

    true = pairwise_kernels(minitest, y, metric="cosine")
    test = pairwise_similarities(minitest, query, metric="cosine")

    np.testing.assert_array_equal(test, true)


def test_sigbool():

    minitest = np.array([[1, 1, 1],
                         [2, 2, 2],
                         [0, 2, 0]])
    cutoffs = {0: (-.5, .5),
               1: (-1, 1),
               2: (1, 3)}
    sigbool = sig_bool(pd.DataFrame(minitest), cutoffs)

    true = np.array([[True, False, False],
                     [True, True, False],
                     [False, True, True]])

    np.testing.assert_array_equal(true, sigbool)


def test_sigdict():

    def get_vals(scores, sigdict, retrieve):
        allvals = []
        if retrieve == "rows":
            for k, v in sigdict.items():
                allvals.extend(scores.loc[v, k])
        else:
            for k, v in sigdict.items():
                allvals.extend(scores.loc[k, v])
        return allvals

    minitest = pd.DataFrame(4 * np.random.random((20, 5)) - 2)
    cutoffs = {0: (-0.5, 0.5),
               1: (-0.5, 0.5),
               2: (-0.5, 0.5),
               3: (-0.5, 0.5),
               4: (-0.5, 0.5)}

    test = sig_dictionary(minitest, cutoffs, retrieve="rows", sign="both")
    assert len(test.keys()) == minitest.shape[1]
    test = get_vals(minitest, test, retrieve="rows")
    assert all((v < -0.5) or (v > 0.5) for v in test)

    test = sig_dictionary(minitest, cutoffs, retrieve="rows", sign="positive")
    test = get_vals(minitest, test, retrieve="rows")
    assert all(v > 0.5 for v in test)

    test = sig_dictionary(minitest, cutoffs, retrieve="rows", sign="negative")
    test = get_vals(minitest, test, retrieve="rows")
    assert all(v < -0.5 for v in test)

    test = sig_dictionary(minitest, cutoffs, retrieve="cols", sign="both")
    assert len(test.keys()) == minitest.shape[0]
    test = get_vals(minitest, test, retrieve="cols")
    assert all((v < -0.5) or (v > 0.5) for v in test)

    test = sig_dictionary(minitest, cutoffs, retrieve="cols", sign="positive")
    test = get_vals(minitest, test, retrieve="cols")
    assert all(v > 0.5 for v in test)

    test = sig_dictionary(minitest, cutoffs, retrieve="cols", sign="negative")
    test = get_vals(minitest, test, retrieve="cols")
    assert all(v < -0.5 for v in test)
