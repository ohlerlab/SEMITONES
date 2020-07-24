import matplotlib.pyplot as plt


def visualize_cells(EMB, r_cells, figsize=None, dpi=None):
    """
    EMB: 2D matrix
        The 2D embedding of the cells in an experiment.
    r_cells: list
        A list of cell indices to visualize
    figsize: tuple
        A tuple of (width, height)
    dpi: int
        The dpi to use for the figure
    """

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
