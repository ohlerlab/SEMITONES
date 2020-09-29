import numpy as np
from scipy.sparse import csr_matrix

from SEMITONES.enrichment_scoring import _enrichment_scoring
from SEMITONES.enrichment_scoring import _interaction_set
from SEMITONES.enrichment_scoring import _max_set
from SEMITONES.enrichment_scoring import _median_set
from SEMITONES.enrichment_scoring import _min_set
from SEMITONES.enrichment_scoring import calculate_escores
from SEMITONES.enrichment_scoring import feature_set_values
from SEMITONES.enrichment_scoring import permute
from SEMITONES.enrichment_scoring import sig_interval


def test_permute():

    minitest = np.random.random((5, 10))

    P = permute(minitest)

    np.testing.assert_raises(AssertionError, np.testing.assert_allclose,
                             minitest, P)


def test_features_set_interaction():

    minitest = np.zeros((5, 10))
    rows = [0, 1, 2, 3, 4, 0, 1, 2, 4, 3, 4, 2, 1, 2, 3, 4]
    cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 4, 2, 6, 1, 3]
    for i, j in zip(rows, cols):
        minitest[i, j] = 1

    sets = [[1, 3], [2, 3], [9, 1]]

    # test interaction
    test = _interaction_set(minitest, sets)[1].toarray()
    true = np.asarray([[0., 0., 0.],
                       [0., 0., 0.],
                       [0., 0., 0.],
                       [1., 0., 1.],
                       [0., 0., 0.]])
    np.testing.assert_allclose(test, true, atol=1e-6, rtol=1e-6)


def test_feature_set_median():

    minitest = np.zeros((5, 10))
    rows = [0, 1, 2, 3, 4, 0, 1, 2, 4, 3, 4, 2, 1, 2, 3, 4]
    cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 4, 2, 6, 1, 3]
    for i, j in zip(rows, cols):
        minitest[i, j] = 1

    sets = [[1, 3], [2, 3], [9, 1]]
    # test median
    test = _median_set(minitest, sets)[1].toarray()
    true = np.asarray([[0., 0., 0.],
                       [0.5, 0.5, 0.5],
                       [0., 0.5, 0.],
                       [1., 0.5, 1.],
                       [0.5, 0.5, 0.]])
    np.testing.assert_allclose(test, true, atol=1e-6, rtol=1e-6)


def test_feature_set_min():

    minitest = np.zeros((5, 10))
    rows = [0, 1, 2, 3, 4, 0, 1, 2, 4, 3, 4, 2, 1, 2, 3, 4]
    cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 4, 2, 6, 1, 3]
    for i, j in zip(rows, cols):
        minitest[i, j] = 1

    sets = [[1, 3], [2, 3], [9, 1]]

    # test min
    test = _min_set(minitest, sets)[1].toarray()
    true = np.asarray([[0., 0., 0.],
                       [0., 0., 0.],
                       [0., 0., 0.],
                       [1., 0., 1.],
                       [0., 0., 0.]])
    np.testing.assert_allclose(test, true, atol=1e-6, rtol=1e-6)


def test_feature_set_max():

    minitest = np.zeros((5, 10))
    rows = [0, 1, 2, 3, 4, 0, 1, 2, 4, 3, 4, 2, 1, 2, 3, 4]
    cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 4, 2, 6, 1, 3]
    for i, j in zip(rows, cols):
        minitest[i, j] = 1

    sets = [[1, 3], [2, 3], [9, 1]]
    test = _max_set(minitest, sets)[1].toarray()
    true = np.asarray([[0., 0., 0.],
                       [1., 1., 1.],
                       [0., 1., 0.],
                       [1., 1., 1.],
                       [1., 1., 0.]])
    np.testing.assert_allclose(test, true, atol=1e-6, rtol=1e-6)


def test_feature_set_binary():

    minitest = np.zeros((5, 10))
    rows = [0, 1, 2, 3, 4, 0, 1, 2, 4, 3, 4, 2, 1, 2, 3, 4]
    cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 4, 2, 6, 1, 3]
    for i, j in zip(rows, cols):
        minitest[i, j] = 1

    sets = [[1, 3], [2, 3], [9, 1]]
    test = _max_set(minitest, sets)[1].toarray()
    true = np.asarray([[0., 0., 0.],
                       [1., 1., 1.],
                       [0., 1., 0.],
                       [1., 1., 1.],
                       [1., 1., 0.]])
    np.testing.assert_allclose(test, true, atol=1e-6, rtol=1e-6)


def test_feature_set():

    minitest = np.zeros((5, 10))
    rows = [0, 1, 2, 3, 4, 0, 1, 2, 4, 3, 4, 2, 1, 2, 3, 4]
    cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 4, 2, 6, 1, 3]
    for i, j in zip(rows, cols):
        minitest[i, j] = 1

    sets = [[1, 3], [2, 3], [9, 1]]

    # test interaction
    test = feature_set_values(minitest, sets, combtype="interaction").toarray()
    true = np.asarray([[0., 0., 0.],
                       [0., 0., 0.],
                       [0., 0., 0.],
                       [1., 0., 1.],
                       [0., 0., 0.]])
    np.testing.assert_allclose(test, true, atol=1e-6, rtol=1e-6)

    # test max
    test = feature_set_values(minitest, sets, combtype="max").toarray()
    true = np.asarray([[0., 0., 0.],
                       [1., 1., 1.],
                       [0., 1., 0.],
                       [1., 1., 1.],
                       [1., 1., 0.]])
    np.testing.assert_allclose(test, true, atol=1e-6, rtol=1e-6)

    # test min
    test = feature_set_values(minitest, sets, combtype="min").toarray()
    true = np.asarray([[0., 0., 0.],
                       [0., 0., 0.],
                       [0., 0., 0.],
                       [1., 0., 1.],
                       [0., 0., 0.]])
    np.testing.assert_allclose(test, true, atol=1e-6, rtol=1e-6)

    # test median
    test = feature_set_values(minitest, sets, combtype="median").toarray()
    true = np.asarray([[0., 0., 0.],
                       [0.5, 0.5, 0.5],
                       [0., 0.5, 0.],
                       [1., 0.5, 1.],
                       [0.5, 0.5, 0.]])
    np.testing.assert_allclose(test, true, atol=1e-6, rtol=1e-6)

    # test binary
    test = feature_set_values(minitest, sets, combtype="binary").toarray()
    true = np.asarray([[0., 0., 0.],
                       [1., 1., 1.],
                       [0., 1., 0.],
                       [1., 1., 1.],
                       [1., 1., 0.]])
    np.testing.assert_allclose(test, true, atol=1e-6, rtol=1e-6)


def test_escoring():

    minitest = np.array([
        [0.34065522, 0.50168601, 0.44391542, 0.6388779, 0.35760797,
         0.40253224, 0.31333822, 0.39277549, 0.59726848, 0.63088571],
        [0.43803599, 0.36222281, 0.27606492, 0.67858647, 0.97190305,
         0.69773913, 0.49869304, 0.96105146, 0.30361556, 0.45692298],
        [0.09826217, 0.16485451, 0.95936339, 0.02990735, 0.95325413,
         0.94589958, 0.39961259, 0.8382766, 0.6376664, 0.78479295],
        [0.16446985, 0.15305316, 0.54160174, 0.4545658, 0.44105138,
         0.35408377, 0.49301503, 0.59114443, 0.98804617, 0.73359866],
        [0.72424446, 0.7914756, 0.45841415, 0.96167232, 0.32146698,
         0.53233922, 0.55770827, 0.17920708, 0.61025691, 0.25102257]])
    cells = [1, 3]
    test_s = np.array([
        [0.23042758, 0.26242135],
        [0.07697681, 0.32526972],
        [0.68100719, 0.77398591],
        [0.47088978, 0.05630848],
        [0.94069831, 0.4730802]])
    true = np.array([
        [0.08152949, -0.00720362],
        [0.10558164, 0.01163322],
        [0.15488303, 0.15993126],
        [0.00778551, -0.09954508],
        [-0.09972833, 0.11872709],
        [0.04684367, 0.21091833],
        [0.12456712, -0.02400143],
        [-0.1680262, 0.04354533],
        [0.11860297, -0.08559907],
        [-0.07660861, 0.00104565]])
    test = calculate_escores(minitest, cells, S=test_s, scale_exp=True)
    np.testing.assert_allclose(test, true, atol=1e-6, rtol=1e-6)


def test_escoring_full_csr():

    minitest = np.random.random((10, 50))
    S = np.random.random((10, 2))
    minitest_sparse = csr_matrix(minitest)

    i, test1 = _enrichment_scoring(minitest, S, True, 0)
    i, test2 = _enrichment_scoring(minitest_sparse, S, True, 0, 2)
    vals1, vals2 = [], []
    for i, j in zip(test1.values(), test2.values()):
        vals1.append(i)
        vals2.append(j)
    np.testing.assert_allclose(vals1, vals2)


def test_escoring_per_cell_iterative():

    minitest = np.random.random((10, 50))
    S = np.random.random((10, 2))
    S1, S2 = S[:, 0], S[:, 1]

    i, test1 = _enrichment_scoring(minitest, S, True, 0)
    i, test2a = _enrichment_scoring(minitest, S1, True, 0)
    i, test2b = _enrichment_scoring(minitest, S2, True, 0)

    vals1 = []
    for ia in test1.values():
        for i in ia:
            vals1.append(i)

    vals2 = []
    for i in test2a:
        vals2.append(i)
    for i in test2b:
        vals2.append(i)

    assert vals1 == vals2


def test_interval_sparse():

    import pandas as pd

    minitest = pd.DataFrame(np.random.random((5, 10)))
    minitest_sparse = csr_matrix(minitest)

    dense = sig_interval(minitest, 3)
    dense = pd.DataFrame(dense).values[0]
    sparse = sig_interval(minitest_sparse, 3)
    sparse = pd.DataFrame(sparse).values[0][0]

    np.testing.assert_allclose(dense, sparse)
