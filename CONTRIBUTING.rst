============
Contributing
============

Welcome to ``ms3`` contributor's guide.

This document focuses on getting any potential contributor familiarized
with the development processes, but `other kinds of contributions`_ are also
appreciated.

If you are new to using git_ or have never collaborated in a project previously,
please have a look at `contribution-guide.org`_. Other resources are also
listed in the excellent `guide created by FreeCodeCamp`_ [#contrib1]_.

Please notice, all users and contributors are expected to be **open,
considerate, reasonable, and respectful**. When in doubt, `Python Software
Foundation's Code of Conduct`_ is a good reference in terms of behavior
guidelines.


Issue Reports
=============

If you experience bugs or general issues with ``ms3``, please have a look
on the `issue tracker`_. If you don't see anything useful there, please feel
free to fire an issue report.

.. tip::
   Please don't forget to include the closed issues in your search.
   Sometimes a solution was already reported, and the problem is considered
   **solved**.

New issue reports should include information about your programming environment
(e.g., operating system, Python version) and steps to reproduce the problem.
Please try also to simplify the reproduction steps to a very minimal example
that still illustrates the problem you are facing. By removing other factors,
you help us to identify the root cause of the issue.


Documentation Improvements
==========================

You can help improve ``ms3`` docs by making them more readable and coherent, or
by adding missing information and correcting mistakes.

``ms3`` documentation uses Sphinx_ as its main documentation compiler.
This means that the docs are kept in the same repository as the project code, and
that any documentation update is done in the same way was a code contribution.
The documentation is written in reStructuredText_ and includes the myst-nb_ extension.

Documentation pages are written in reStructuredText_ (as are the docstrings that are automatically compiled to the
API docs).

.. tip::
  Please notice that the `GitHub web interface`_ provides a quick way of
  propose changes in ``ms3``'s files. While this mechanism can
  be tricky for normal code contributions, it works perfectly fine for
  contributing to the docs, and can be quite handy.

  If you are interested in trying this method out, please navigate to
  the ``docs`` folder in the source repository_, find which file you
  would like to propose changes and click in the little pencil icon at the
  top, to open `GitHub's code editor`_. Once you finish editing the file,
  please write a message in the form at the bottom of the page describing
  which changes have you made and what are the motivations behind them and
  submit your proposal.


When working on documentation changes in your local machine, you can
compile them using |tox|_::

    tox -e docs

and use Python's built-in web server for a preview in your web browser
(``http://localhost:8000``)::

    python3 -m http.server --directory 'docs/_build/html'


Code Contributions
==================

.. admonition:: TL;DR

   * Fork the repository.
   * (Create a virtual environment, :ref:`see below <virtenv>`).
   * Head into the local clone of your fork and hit ``pip install -e ".[dev]"`` (where ``.`` is the current directory).
   * Install the precommit hooks via ``pre-commit install``.
   * Implement the changes and create a Pull Request against the ``development`` branch.
   * Thank you!


Submit an issue
---------------

Before you work on any non-trivial code contribution it's best to first create
a report in the `issue tracker`_ to start a discussion on the subject.
This often provides additional considerations and avoids unnecessary work.

Create an environment
---------------------

Before you start coding, we recommend creating an isolated `virtual
environment`_ to avoid any problems with your installed Python packages.
This can easily be done via either |virtualenv|_::

    virtualenv <PATH TO VENV>
    source <PATH TO VENV>/bin/activate

or Miniconda_::

    conda create -n ms3 python=3.10
    conda activate ms3

Clone the repository
--------------------

#. Create an user account on |the repository service| if you do not already have one.
#. Fork the project repository_: click on the *Fork* button near the top of the
   page. This creates a copy of the code under your account on |the repository service|.
#. Clone this copy to your local disk::

    git clone git@github.com:YourLogin/ms3.git

#. You should run::

    cd ms3
    pip install pip -e .

   to be able to import the package under development in the Python REPL.



Implement your changes
----------------------

#. Create a branch to hold your changes::

    git checkout -b my-feature

   and start making changes. Never work on the main branch!

#. Start your work on this branch. Don't forget to add docstrings_ to new
   functions, modules and classes, especially if they are part of public APIs.

#. Add yourself to the list of contributors in ``AUTHORS.rst``.

#. When you’re done editing, do::

    git add <MODIFIED FILES>
    git commit

   to record your changes in git_.

   Please make sure to see the validation messages from |pre-commit|_ and fix
   any eventual issues.
   This should automatically use flake8_/black_ to check/fix the code style
   in a way that is compatible with the project.

   .. important:: Don't forget to add unit tests and documentation in case your
      contribution adds an additional feature and is not just a bugfix.

      Moreover, writing a `descriptive commit message`_ is highly recommended.
      In case of doubt, you can check the commit history with::

         git log --graph --decorate --pretty=oneline --abbrev-commit --all

      to look for recurring communication patterns.

#. Please check that your changes don't break any unit tests with::

    tox

   (after having installed |tox|_ with ``pip install tox`` or ``pipx``).

   You can also use |tox|_ to run several other pre-configured tasks in the
   repository. Try ``tox -av`` to see a list of the available checks.

Submit your contribution
------------------------

#. If everything works fine, push your local branch to |the repository service| with::

    git push -u origin my-feature

#. Go to the web page of your fork and click |contribute button|
   to send your changes for review.

   Find more detailed information in `creating a PR`_. You might also want to open
   the PR as a draft first and mark it as ready for review after the feedbacks
   from the continuous integration (CI) system or any required fixes.


Coding Conventions
------------------

Please make sure to run ``pre-commit install`` in your local clone of the repository. This way, many coding
conventions are automatically applied before each commit!

Commit messages
~~~~~~~~~~~~~~~

``ms3`` uses `Conventional Commits <https://www.conventionalcommits.org/>`__ to determine the next SemVer version number. Please make sure to prefix each
message with one of:

+---------------+--------------------------+-------------------------------------------------------------------------------------------------------------+--------+
| Commit Type   | Title                    | Description                                                                                                 | SemVer |
+===============+==========================+=============================================================================================================+========+
| ``feat``      | Features                 | A new feature                                                                                               | MINOR  |
+---------------+--------------------------+-------------------------------------------------------------------------------------------------------------+--------+
| ``fix``       | Bug Fixes                | A bug Fix                                                                                                   | PATCH  |
+---------------+--------------------------+-------------------------------------------------------------------------------------------------------------+--------+
| ``docs``      | Documentation            | Documentation only changes                                                                                  | PATCH  |
+---------------+--------------------------+-------------------------------------------------------------------------------------------------------------+--------+
| ``style``     | Styles                   | Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)      | PATCH  |
+---------------+--------------------------+-------------------------------------------------------------------------------------------------------------+--------+
| ``refactor``  | Code Refactoring         | A code change that neither fixes a bug nor adds a feature                                                   | PATCH  |
+---------------+--------------------------+-------------------------------------------------------------------------------------------------------------+--------+
| ``perf``      | Performance Improvements | A code change that improves performance                                                                     | PATCH  |
+---------------+--------------------------+-------------------------------------------------------------------------------------------------------------+--------+
| ``test``      | Tests                    | Adding missing tests or correcting existing tests                                                           | PATCH  |
+---------------+--------------------------+-------------------------------------------------------------------------------------------------------------+--------+
| ``build``     | Builds                   | Changes that affect the build system or external dependencies (example scopes: gulp, broccoli, npm)         | PATCH  |
+---------------+--------------------------+-------------------------------------------------------------------------------------------------------------+--------+
| ``ci``        | Continuous Integrations  | Changes to our CI configuration files and scripts (example scopes: Travis, Circle, BrowserStack, SauceLabs) | PATCH  |
+---------------+--------------------------+-------------------------------------------------------------------------------------------------------------+--------+
| ``chore``     | Chores                   | Other changes that don't modify src or test files                                                           | PATCH  |
+---------------+--------------------------+-------------------------------------------------------------------------------------------------------------+--------+
| ``revert``    | Reverts                  | Reverts a previous commit                                                                                   | PATCH  |
+---------------+--------------------------+-------------------------------------------------------------------------------------------------------------+--------+

In the case of breaking changes, which result in a new major version, please add a ``!`` after the type, e.g., ``refactor!:``.
This type of commit message needs to come with a body, starting with ``BREAKING CHANGE:``, which explains in great detail everything
that will not be working anymore.
Troubleshooting
---------------

The following tips can be used when facing problems to build or test the
package:

#. Make sure to fetch all the tags from the upstream repository_.
   The command ``git describe --abbrev=0 --tags`` should return the version you
   are expecting. If you are trying to run CI scripts in a fork repository,
   make sure to push all the tags.
   You can also try to remove all the egg files or the complete egg folder, i.e.,
   ``.eggs``, as well as the ``*.egg-info`` folders in the ``src`` folder or
   potentially in the root of your project.

#. Sometimes |tox|_ misses out when new dependencies are added, especially to
   ``setup.cfg`` and ``docs/requirements.txt``. If you find any problems with
   missing dependencies when running a command with |tox|_, try to recreate the
   ``tox`` environment using the ``-r`` flag. For example, instead of::

    tox -e docs

   Try running::

    tox -r -e docs

#. Make sure to have a reliable |tox|_ installation that uses the correct
   Python version (e.g., 3.7+). When in doubt you can run::

    tox --version
    # OR
    which tox

   If you have trouble and are seeing weird errors upon running |tox|_, you can
   also try to create a dedicated `virtual environment`_ with a |tox|_ binary
   freshly installed. For example::

    virtualenv .venv
    source .venv/bin/activate
    .venv/bin/pip install tox
    .venv/bin/tox -e all

#. `Pytest can drop you`_ in an interactive session in the case an error occurs.
   In order to do that you need to pass a ``--pdb`` option (for example by
   running ``tox -- -k <NAME OF THE FALLING TEST> --pdb``).
   You can also setup breakpoints manually instead of using the ``--pdb`` option.


Maintainer tasks
================

Releases
--------

If you are part of the group of maintainers and have correct user permissions
on PyPI_, the following steps can be used to release a new version for
``ms3``:

#. Make sure all unit tests are successful.
#. Tag the current commit on the main branch with a release tag, e.g., ``v1.2.3``.
#. Push the new tag to the upstream repository_, e.g., ``git push upstream v1.2.3``
#. Clean up the ``dist`` and ``build`` folders with ``tox -e clean``
   (or ``rm -rf dist build``)
   to avoid confusion with old builds and Sphinx docs.
#. Run ``tox -e build`` and check that the files in ``dist`` have
   the correct version (no ``.dirty`` or git_ hash) according to the git_ tag.
   Also check the sizes of the distributions, if they are too big (e.g., >
   500KB), unwanted clutter may have been accidentally included.
#. Run ``tox -e publish -- --repository pypi`` and check that everything was
   uploaded to PyPI_ correctly.



.. [#contrib1] Even though, these resources focus on open source projects and
   communities, the general ideas behind collaborating with other developers
   to collectively create software are general and can be applied to all sorts
   of environments, including private companies and proprietary code bases.


.. <-- strart -->
.. todo:: Please review and change the following definitions:

.. |the repository service| replace:: GitHub
.. |contribute button| replace:: "Create pull request"

.. _repository: https://github.com/DCMLab/ms3
.. _issue tracker: https://github.com/DCMLab/ms3/issues
.. <-- end -->


.. |virtualenv| replace:: ``virtualenv``
.. |pre-commit| replace:: ``pre-commit``
.. |tox| replace:: ``tox``


.. _black: https://pypi.org/project/black/
.. _CommonMark: https://commonmark.org/
.. _contribution-guide.org: https://www.contribution-guide.org/
.. _creating a PR: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request
.. _descriptive commit message: https://chris.beams.io/posts/git-commit
.. _docstrings: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
.. _first-contributions tutorial: https://github.com/firstcontributions/first-contributions
.. _flake8: https://flake8.pycqa.org/en/stable/
.. _git: https://git-scm.com
.. _GitHub's fork and pull request workflow: https://guides.github.com/activities/forking/
.. _guide created by FreeCodeCamp: https://github.com/FreeCodeCamp/how-to-contribute-to-open-source
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _myst-nb: https://myst-nb.readthedocs.io/en/latest/
.. _other kinds of contributions: https://opensource.guide/how-to-contribute
.. _pre-commit: https://pre-commit.com/
.. _PyPI: https://pypi.org/
.. _PyScaffold's contributor's guide: https://pyscaffold.org/en/stable/contributing.html
.. _Pytest can drop you: https://docs.pytest.org/en/stable/how-to/failures.html#using-python-library-pdb-with-pytest
.. _Python Software Foundation's Code of Conduct: https://www.python.org/psf/conduct/
.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/
.. _Sphinx: https://www.sphinx-doc.org/en/master/
.. _tox: https://tox.wiki/en/stable/
.. _virtual environment: https://realpython.com/python-virtual-environments-a-primer/
.. _virtualenv: https://virtualenv.pypa.io/en/stable/

.. _GitHub web interface: https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files
.. _GitHub's code editor: https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files
