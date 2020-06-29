========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis|
        | |codecov|
    * - package
      - | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/escoring/badge/?style=flat
    :target: https://readthedocs.org/projects/escoring
    :alt: Documentation Status

.. |travis| image:: https://api.travis-ci.org/ohlerlab/escoring.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/ohlerlab/escoring

.. |codecov| image:: https://codecov.io/gh/ohlerlab/escoring/branch/master/graphs/badge.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/ohlerlab/escoring

.. |commits-since| image:: https://img.shields.io/github/commits-since/ohlerlab/escoring/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/ohlerlab/escoring/compare/v0.0.0...master



.. end-badges

A method for the identification of informative features in scRNA-seq and scATAC-seq data.

* Free software: MIT license

Installation
============

::

    pip install escoring

You can also install the in-development version with::

    pip install https://github.com/ohlerlab/escoring/archive/master.zip


Documentation
=============


https://escoring.readthedocs.io/


Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
