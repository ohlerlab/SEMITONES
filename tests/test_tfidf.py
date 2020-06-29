import numpy as np
from scipy.sparse import csr_matrix

from escoring.tfidf import TFIDF
from escoring.tfidf import _inverse_document_frequency
from escoring.tfidf import _term_frequency


def test_tfidf():

    minitest = np.zeros((5, 10))
    rows, cols = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4], range(0, 10)
    for i, j in zip(rows, cols):
        minitest[i, j] = 1

    test = TFIDF(minitest)

    true = np.asarray(
        [[1.16096405, 0., 0., 0., 0.,
          1.16096405, 0., 0., 0., 0.],
         [0., 1.16096405, 0., 0., 0.,
          0., 1.16096405, 0., 0., 0.],
         [0., 0., 1.16096405, 0., 0.,
          0., 0., 1.16096405, 0., 0.],
         [0., 0., 0., 1.16096405, 0.,
          0., 0., 0., 1.16096405, 0.],
         [0., 0., 0., 0., 1.16096405,
          0., 0., 0., 0., 1.16096405]])

    np.testing.assert_allclose(test, true, atol=1e-6, rtol=1e-6)


def test_tf():

    minitest = np.zeros((5, 10))
    rows, cols = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4], range(0, 10)
    for i, j in zip(rows, cols):
        minitest[i, j] = 1

    test = _term_frequency(minitest, "term_frequency")

    true = np.asarray([[0.5, 0., 0., 0., 0., 0.5, 0., 0., 0., 0.],
                       [0., 0.5, 0., 0., 0., 0., 0.5, 0., 0., 0.],
                       [0., 0., 0.5, 0., 0., 0., 0., 0.5, 0., 0.],
                       [0., 0., 0., 0.5, 0., 0., 0., 0., 0.5, 0.],
                       [0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0.5]])

    np.testing.assert_allclose(test, true, atol=1e-6, rtol=1e-6)


def test_idf():

    minitest = np.zeros((5, 10))
    rows, cols = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4], range(0, 10)
    for i, j in zip(rows, cols):
        minitest[i, j] = 1

    test = _inverse_document_frequency(minitest, "inv_doc_freq")

    true = np.log2(5)

    assert all(i == true for i in test)


def test_individual():

    minitest = np.zeros((5, 10))
    rows, cols = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4], range(0, 10)
    for i, j in zip(rows, cols):
        minitest[i, j] = 1

    test1 = _term_frequency(minitest, "term_frequency")
    test2 = _inverse_document_frequency(minitest, "inv_doc_freq")
    test3 = TFIDF(minitest)

    np.testing.assert_allclose(test1 * test2, test3)


def test_sparse():

    minitest = np.zeros((5, 10))
    rows, cols = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4], range(0, 10)
    for i, j in zip(rows, cols):
        minitest[i, j] = 1
    sparse = csr_matrix(minitest)

    test1 = TFIDF(minitest)
    test2 = TFIDF(sparse).toarray()

    np.testing.assert_allclose(test1, test2)


def test_unary():

    minitest = np.zeros((5, 10))
    rows, cols = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4], range(0, 10)
    for i, j in zip(rows, cols):
        minitest[i, j] = 1

    tfidf = TFIDF(minitest, idf_type="unary")
    tf = _term_frequency(minitest, "term_frequency")

    np.testing.assert_allclose(tfidf, tf)


def test_raw():

    minitest = np.zeros((5, 10))
    rows, cols = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4], range(0, 10)
    for i, j in zip(rows, cols):
        minitest[i, j] = 1

    tf = _term_frequency(minitest, "raw")

    np.testing.assert_allclose(minitest, tf)


def test_binary():

    minitest = np.zeros((5, 10))
    rows, cols = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4], range(0, 10)
    for i, j in zip(rows, cols):
        minitest[i, j] = 1

    tf = _term_frequency(minitest, "binary", threshold=0.5)

    np.testing.assert_allclose(minitest, tf)


def test_log_norm():

    minitest = np.zeros((5, 10))
    rows, cols = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4], range(0, 10)
    for i, j in zip(rows, cols):
        minitest[i, j] = 1

    tf = _term_frequency(minitest, "log_norm")

    np.testing.assert_allclose(np.log1p(minitest), tf)


def test_k_aug_freq_sparse():

    minitest = np.zeros((5, 10))
    rows, cols = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4], range(0, 10)
    for i, j in zip(rows, cols):
        minitest[i, j] = 1
    sparse = csr_matrix(minitest)

    tf1 = _term_frequency(minitest, "k_aug_freq", K=0.5)
    tf2 = _term_frequency(sparse, "k_aug_freq", K=0.5).toarray()

    np.testing.assert_allclose(tf1, tf2)


def test_k_aug_freq():

    minitest = np.zeros((5, 10))
    rows, cols = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4], range(0, 10)
    for i, j in zip(rows, cols):
        minitest[i, j] = 1

    a = 0.5 + (1 - 0.5)
    b = minitest / np.amax(minitest, 1)[:, np.newaxis]
    tf1 = a * b
    tf2 = _term_frequency(minitest, "k_aug_freq", K=0.5)

    np.testing.assert_allclose(tf1, tf2)
