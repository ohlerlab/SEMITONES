import matplotlib.pyplot as plt


def visualize_cells(EMB, r_cells, figsize=None, dpi=None):
    """Show the reference cells in orange on
    the 2D embedding of the cells.

    Parameters
    ----------
    EMB: 2D array
        The coordinates of the 2D embedding of
        the cells in an experiment.
    r_cells: list
        A list of (row) indices of the selected
        reference cells.
    figsize: tuple
        The figsize to pass to
        matplotlib.pyplot.figure()
    dpi: int
        The dpi value to pass to
        matplotlib.pyplot.figure()

    Returns
    -------
    -"""

    figsize = (10, 7) if figsize is None else figsize
    dpi = 150 if dpi is None else dpi

    plt.figure(figsize=figsize, dpi=dpi)

    plt.scatter(EMB[:, 0], EMB[:, 1], s=2, color="darkgrey")
    plt.scatter(EMB[r_cells, 0], EMB[r_cells, 1], s=5, color="#eb7b14")

    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    plt.show()


def visualize_expressed_gene(X, EMB, gene_idx, figsize=None, dpi=None):
    """Visualize the expression of a certain
    gene in the 2D cell embedding space.

    Parameters
    ----------
    X: matrix-like
        A numpy matrix where rows are cells and columns are features.
    EMB: 2D array
        The coordinates of the 2D embedding of the cells in an experiment.
    gene_idx: int
        The column index of the gene that is to be visualized.
    figsize: tuple
        The figsize to pass to matplotlib.pyplot.figure()
    dpi: int
        The dpi value to pass to matplotlib.pyplot.figure()

    Returns
    -------
    -"""

    figsize = (10, 7) if figsize is None else figsize
    dpi = 150 if dpi is None else dpi

    plt.figure(figsize=figsize, dpi=dpi)

    plt.scatter(EMB[:, 0], EMB[:, 1], c=X[:, gene_idx],
                cmap="Oranges", s=1)

    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    plt.show()
