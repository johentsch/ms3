==============
Installing ms3
==============

The library is hosted at `PyPI <https://pypi.org/project/ms3/>`__ and can be installed via ``pip``.

Unix-based systems
==================

Open up your console / command-line terminal and make sure to enter the Python 3 environment where you want to install the library.
You can check which Python executable is currently used by typing

.. code-block:: console

    which python

Using Python >= 3.6
-------------------

Check the Python of your environment by typing

.. code-block:: console

    python -V

If it's a Python 2 version, replace ``python`` by ``python3``:

.. code-block:: console

    which python3
    python3 -V

If Python 3 doesn't seem to be installed, we recommend installing it via the small `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`__
or the large `Anaconda <https://docs.anaconda.com/anaconda/install/>`__. The latter comes with many libraries pre-installed,
e.g. Jupyter notebooks.

Installing via ``pip``
----------------------

First, update your package manager

.. code-block:: console

    python3 -m pip install –-upgrade pip

Then you're ready to install ms3:

.. code-block:: console

    python3 -m pip install ms3

