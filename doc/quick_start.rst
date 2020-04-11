#########################################
Quick Start with the mne-project-template
#########################################

This package serves as a skeleton package aiding at MNE
compatible packages.

Creating your own mne contribution package
==========================================

For illustration purposes we want to create a ``mne-foo`` project
named ``mnefoo``. Here is a table of the naming of this project and
the project you will create:



1. Download and setup your repository
-------------------------------------

To create your package, you need to clone the ``mne-project-template`` repository
and rename it to your convenience (i.e:``mne-foo``)::

    $ git clone https://github.com/mne-tools/mne-project-template.git mne-foo
    $ cd mne-foo

Before to reinitialize your git repository, you need to replace the template
information with you own. We provide you with a convenient script to speed up
the process, but you can also do it manually.

1.1.1 bootstrap your mne project using a convenience script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Open your favorite editor and change ``PKG_NAME``, ``PYTHON_NAME``, 
and ``GH_NAME`` in ``mne_project_template_bootstrap.sh`` with your own
information. Then run the bootsrap script::

   $ bash  mne_project_template_bootstrap.sh

1.1.2 Update your project manually
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Replace all occurrences of ``mne_nirs`` and ``mne-nirs``
with the name of you own contribution. You can find all the occurrences using
the following command::

    $ git grep mne_nirs
    $ git grep mne-nirs

You can do this with your favorite editor or use the ``sed`` tool.
In linux machine::

    $ git grep -l 'mne_nirs' | xargs sed -i 's/mne_nirs/mnefoo/g'
    $ git grep -l 'mne-nirs' | xargs sed -i 's/mne-nirs/mne-foo/g'

this is how to do it in Macosx machine::

    $ git grep -l 'mne_nirs' | xargs sed -i '' -e 's/mne_nirs/mnefoo/g'
    $ git grep -l 'mne-nirs' | xargs sed -i '' -e 's/mne-nirs/mne-foo/g'

   
Update the module directory name::

    $ mv mne_nirs mnefoo

1.3 Remove history and convert it into a new project
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To remove the history of the template package, you need to remove the `.git`
directory::

    $ rm -rf .git

Then, you need to initialize your new git repository::

    $ git init
    $ git add .
    $ git commit -m 'Initial commit'

Finally, you create an online repository on GitHub and push your code online::

    $ git remote add origin https://github.com/your_remote/mne-foo.git
    $ git push origin master


2. Edit the documentation
-------------------------

.. _Sphinx: http://www.sphinx-doc.org/en/stable/

The documentation is created using Sphinx_. In addition, the examples are
created using ``sphinx-gallery``. Therefore, to generate locally the
documentation, you are required to install the following packages::

    $ pip install sphinx sphinx-gallery sphinx_rtd_theme matplotlib numpydoc pillow

The documentation is made of:

* a home page, ``doc/index.rst``;
* an API documentation, ``doc/api.rst`` in which you should add all public
  objects for which the docstring should be exposed publicly.
* a User Guide documentation, ``doc/user_guide.rst``, containing the narrative
  documentation of your package, to give as much intuition as possible to your
  users.
* examples which are created in the `examples/` folder. Each example
  illustrates some usage of the package. the example file name should start by
  `plot_*.py`.

The documentation is built with the following commands::

    $ cd doc
    $ make html

