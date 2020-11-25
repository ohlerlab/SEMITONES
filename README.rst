===================================================================
SEMITONES (Single-cEll Marker IdentificaTiON by Enrichment Scoring)
===================================================================

Description
===========

SEMITONES identifies highly enriched features in single-cell omics data without prior clustering. For a detailed description of SEMITONES, please check out the `manuscript <https://www.biorxiv.org/content/10.1101/2020.11.17.386664v1.full>`. For analysis scripts and notebooks relating to the manuscript, please consult the `manuscript's GitHub page <https://github.com/ohlerlab/SEMITONES_paper>`.

Installation
============

You can install the in-development version with::

    pip install https://github.com/ohlerlab/SEMITONES/archive/master.zip

Usage
=====

Usage examples for the basic SEMITONES functionality are provided by means of Jupyter Notebooks in the tutorial directory.

1. `Reference cell selection <https://github.com/ohlerlab/SEMITONES/blob/master/tutorial/notebooks/1_select_reference_cells.ipynb>`_
2. `Enrichment scoring <https://github.com/ohlerlab/SEMITONES/blob/master/tutorial/notebooks/2_enrichment_scoring.ipynb>`_
3. `Gene set enrichment scoring <https://github.com/ohlerlab/SEMITONES/blob/master/tutorial/notebooks/3_gene_set_enrichment_scoring.ipynb>`_
4. `Enrichment scoring for feature selection <https://github.com/ohlerlab/SEMITONES/blob/master/tutorial/notebooks/4_enrichment_scoring_for_feature_selection.ipynb>`_

If you have trouble opening the Jupyter Notebooks on GitHub, try accessing them through the `Jupyter Notebook viewer <https://nbviewer.jupyter.org/>`_

Documentation on individual functions can be found in the project`s `wiki <https://github.com/ohlerlab/SEMITONES/wiki>`_.

System
======

SEMITONES has been tested on a Linux system running CentOS 7 and developed in and test for Python 3.6.

To use the figure widget for cell selection in Jupyter Lab or Jupyter Notebooks, please consult the `plotly installation instructions <https://github.com/plotly/plotly.py>`_.

License
=======

The package is available under the GPL-v3 license. 
