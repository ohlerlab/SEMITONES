import numpy as np

from escoring.cell_selection import from_knn_dist


def test_alg1():

    minitest = np.random.random((3000, 50))

    # make sure you return exactly n_ret cells
    for true_n in np.random.randint(1, 2000, 10):
        test_n = len(from_knn_dist(minitest, n_ret=true_n, roundup=False))
        assert true_n == test_n

    # make sure roundup = True works when selecting max 1% of cells
    for true_n in range(2, 30):
        test_n = len(from_knn_dist(minitest, n_ret=true_n, roundup=True))
        assert true_n == test_n
