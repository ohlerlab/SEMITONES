import warnings
from math import ceil

import numpy as np
import pandas as pd
from plotly import graph_objects
from scipy.sparse import issparse

from escoring._utils import _distances


def from_gui(umap, figsize=(None, None)):
    if figsize == (None, None):
        figsize = (1500, 1500)
    else:
        figsize = figsize
    t = "Please select cells of interest"

    data = [graph_objects.Scattergl(x=umap[:, 0], y=umap[:, 1], mode="markers")]
    layout = graph_objects.Layout(template="plotly_dark",
                                  width=figsize[0], height=figsize[1],
                                  title=graph_objects.layout.Title(text=t),
                                  plot_bgcolor="#000000",
                                  xaxis_showgrid=False, xaxis_zeroline=False,
                                  yaxis_showgrid=False, yaxis_zeroline=False,
                                  xaxis_showticklabels=False,
                                  yaxis_showticklabels=False,
                                  font={"family": "Courier New, monospace",
                                        "size": 18, "color": "#ff8d00"})

    fig = graph_objects.FigureWidget(data, layout)
    fig.layout.hovermode = "closest"

    scatter = fig.data[0]
    colors = ["#ffffff"] * umap.shape[0]
    scatter.marker.color = colors
    scatter.marker.line = dict(width=0)
    scatter.marker.opacity = [0.25] * umap.shape[0]
    scatter.marker.size = [5] * umap.shape[0]

    def _update_point(trace, points, selector):
        c = list(scatter.marker.color)
        s = list(scatter.marker.size)
        o = list(scatter.marker.opacity)
        for i in points.point_inds:
            c[i] = "#ff8d00"
            s[i] = 15
            o[i] = 1
            with fig.batch_update():
                scatter.marker.color = c
                scatter.marker.size = s
                scatter.marker.opacity = o
    scatter.on_click(_update_point)

    return fig


def get_cells_from_gui(fig):

    return np.where(np.array(fig.data[0].marker.opacity) != 0.25)[0]


def from_knn_dist(X, start=None, n_ret=None, metric=None,
                  seed=None, metric_params=None, roundup=None):
    """This function selects a set of cells of length n_ret cells
    based on the distance to the previously selected cells without
    considering each selected cell's nearest neighbours. The number
    of neighbours to be considered is dependent on the number of cells
    that is to be returned.

    Parameters
    ----------
    X: matrix-like object (n_samples, n_features)
        An array where the rows are samples (i.e. cells) and the columns
        are features (i.e. genes or peaks). Accepts pandas dataframes,
        numpy arrays, and scipy compressed sparse row matrix.
    start: str or int
        Either 1) a string indicating whether a random cell ("random")
        or the medoid ("medoid") should be selected as the
        initialization cell or 2) the index of the initialization cell.
    n_ret: int
        The number of cells to be returned. Defaults to 0.01 * n_cells.
    metric: str
        A metric which can be passed to
        sklearn.metrics.pairwise_distances or
        sklearn.metrics.pairwise.pairwise_kernels. Defaults to euclidean.
    seed: int
        Seed used for random selections. Defaults to 42.
    metric_params: dict
        A dictionary of keyword arguments to pass to pairwise_kernels
        if used.

    Returns
    -------
    A set of indices of reference cells of size n_ret"""

    seed = 42 if seed is None else seed
    np.random.seed(seed)
    n_ret = int(X.shape[0] * 0.01) if n_ret is None else n_ret
    metric = "euclidean" if metric is None else metric
    roundup = True if roundup is None else roundup
    if start is None:
        print("No method for start provided. Picking a random cell" +
              " for initialization")
        start = np.random.randint(X.shape[0])

    if not issparse(X) and isinstance(X, pd.DataFrame):
        X = X.values

    # set parameters
    n = X.shape[0]  # number of cells
    n_ret = n_ret  # number of cells to return
    if roundup is True:
        nNN = ceil((n - n_ret) / n_ret)  # number of NNs to exlcude
    else:
        nNN = int((n - n_ret) / n_ret)  # number of cells to return

    ex, total = (nNN * (n_ret - 1) + n_ret), X.shape[0]
    if (roundup is True) & (ex > total):
        warnings.warn("The number of nearest neighbours to exclude is" +
                      " rounded up and less than n_ret cells will" +
                      " be selected.")
    if (roundup is False) & (n - ex > nNN):
        warnings.warn("The number of nearest neighbours is rounded down" +
                      f" The algorithm will return exactly {n_ret} cells" +
                      " but some cells will be neither selected nor excluded.")

    selected = [start]  # initialize return vector
    eliminated = [start]  # initalize exlusion vector
    d = _distances(X, X[start, :].reshape(1, -1), metric=metric,
                   metric_params=metric_params)
    i_max = np.where(d == np.nanmax(d))[0]  # select all cells = max(d)
    s = np.random.choice(i_max, 1)[0]  # randomly pick one
    selected.append(s)  # add to selection
    eliminated.append(s)  # do not select same cell again
    nNNs = np.argpartition(d.ravel(), nNN)[: nNN]  # NNs to the starting cell
    eliminated.extend(nNNs)

    if roundup is True:
        condition = X.shape[0] - len(eliminated) > nNN
    else:
        condition = len(selected) < n_ret
    while condition:
        d = _distances(X, X[selected[-1], :].reshape(1, -1), metric=metric,
                       metric_params=metric_params)
        d[eliminated, :] = np.nan
        nNNs = np.argpartition(d.ravel(), nNN)[: nNN]  # NNs to eliminate
        eliminated.extend(nNNs)
        d[eliminated, :] = np.nan  # do not select NNs
        i_max = np.where(d == np.nanmax(d))[0]  # select all cells = max(d)
        s = np.random.choice(i_max, 1)[0]  # randomly pick one
        selected.append(s)  # select s
        eliminated.append(s)  # eliminate s

        if np.nan in d[nNNs, :]:
            print("help!")

        if roundup is True:
            condition = X.shape[0] - len(eliminated) > nNN
        else:
            condition = len(selected) < n_ret

    return selected


def from_2D_embedding(X, g=(None, None), d=None):
    """This function returns a set of cells by fitting a grid
    to a 2D embedding. The number of selected cells is dependent on
    the size of the grid and the minimum distance between the cells.

    Parameters
    ----------
    X: matrix-like object (n_samples, 2)
        A matrix-like object which contains the 2D embedding of the
        original data, where rows are the cells and the columns
        are the two dimensions.
    g: tuple (int, int)
        A tuple of the grid size (g, g) to fit over the 2D embedding.
        Since cells are selected based on the intersection of grid
        lines, this determines the maximum number of cells to be
        selected.
    d: float
        A parameter which determines how large the distance between
        the cells should be. Defaults to 0.25, which defaults to the
        distance between two cells having to be at least 0.25 * the
        distance between the bottom left point of the grid and
        the first point along the diagnoal to the top right point
        of the grid.

    Returns
    -------
    A set of indices."""

    g = (5, 5) if g == (None, None) else g
    d = 0.25 if d is None else d

    # define a grid of size g
    x, y = X[:, 0], X[:, 1]
    xmin, xmax, ymin, ymax = np.min(x), np.max(x), np.min(y), np.max(y)
    xstep, ystep = (xmax - xmin) / g[0], (ymax - ymin) / g[1]
    xx, yy = np.meshgrid(np.arange(xmin, xmax + xstep, xstep),
                         np.arange(ymin, ymax + ystep, ystep))
    grid = np.array((xx.ravel(), yy.ravel())).T

    # calculate the minimum distance between cells
    pdif = grid[0] - (grid[0][0] + xstep, grid[0][1] + ystep)
    d_min = np.sqrt(np.einsum('i,i->', pdif, pdif)) * d

    # calculate the distance between the cell embedding and the grid
    D = _distances(grid, X)
    D = D[np.where(np.min(D, 1) < d_min)[0], :]

    return np.unique(np.argmin(D, 1))


def from_kmeans_pp(X, n_cells, seed=None):
    """This in an implementation of the kmeans++ algorithm to find
    cluster center for kmeans clustering using the sklearn
    implementation.

    Parameters
    ----------
    X: matrix-like object (n_samples, n_features)
        An array where the rows are samples (i.e. cells) and the columns
        are features (i.e. genes or peaks). Accepts pandas dataframes,
        numpy arrays, and scipy compressed sparse row matrix.
    n_cells: int
        The number of cells to return from the population.
    seed: int
        The random_state seed to enable reporudcibility.
    """

    seed = 42 if seed is None else seed

    from sklearn.cluster import MiniBatchKMeans as mbk
    centers = mbk(n_cells, random_state=seed).fit(X).cluster_centers_
    c_indxs = [np.argmin(_distances(X, centers[i].reshape(1, -1))) for
               i in range(n_cells)]
    return c_indxs
