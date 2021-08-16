# -*- coding: utf-8 -*-
#
# project-template documentation build configuration file, created by
# sphinx-quickstart on Mon Jan 18 14:44:12 2016.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.


import multiprocessing as mp
import sys
from distutils.version import LooseVersion
import sphinx
import os
from sphinx_gallery.sorting import FileNameSortKey

sys.path.append("../")
import mne
from mne_nirs import __version__  # noqa: E402
from mne.tests.test_docstring_parameters import error_ignores


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath('.'))

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx_fontawesome',
    'sphinx_multiversion',
    'sphinx_gallery.gen_gallery',
    'sphinx_bootstrap_divs',
    'numpydoc',
    'sphinxcontrib.bibtex',
]

smv_branch_whitelist = r'^(?!refs/heads/).*$'
# v0.0.1 config is not compatible with sphinx-multiversion, so use 2 onwards
# smv_tag_whitelist = r'^v\d+\.\d+.\d+$'
smv_tag_whitelist = None
# Mark vX.Y.Z as releases
smv_released_pattern = r'^.*v.*$'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# generate autosummary even if no references.
autosummary_generate = True
autodoc_default_options = {'inherited-members': None}

# The suffix of source filenames.
source_suffix = '.rst'

# Generate the plots for the gallery
plot_gallery = 'True'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'MNE-NIRS'
copyright = u'2020, Robert Luke'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = __version__
# The full version, including alpha/beta/rc tags.
release = __version__

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', '_templates']


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# NumPyDoc configuration -----------------------------------------------------

numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = True
numpydoc_validate = True
numpydoc_validation_checks = {'all'} | set(error_ignores)
numpydoc_validation_exclude = {  # set of regex
    # dict subclasses
    r'\.clear', r'\.get$', r'\.copy$', r'\.fromkeys', r'\.items', r'\.keys',
    r'\.pop', r'\.popitem', r'\.setdefault', r'\.update', r'\.values',
    # list subclasses
    r'\.append', r'\.count', r'\.extend', r'\.index', r'\.insert', r'\.remove',
    r'\.sort',
    # we currently don't document these properly (probably okay)
    r'\.__getitem__', r'\.__contains__', r'\.__hash__', r'\.__mul__',
    r'\.__sub__', r'\.__add__', r'\.__iter__', r'\.__div__', r'\.__neg__',
    # copied from sklearn
    r'mne\.utils\.deprecated',
}


# sphinxcontrib-bibtex
bibtex_bibfiles = ['./references.bib']
bibtex_style = 'unsrt'
bibtex_footbibliography_header = ''


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'pydata_sphinx_theme'

# Remove left side bar
html_sidebars = {
  "**": []
}

# variables to pass to HTML templating engine
html_context = {
    'build_dev_html': bool(int(os.environ.get('BUILD_DEV_HTML', False))),
    'versions_dropdown': {
        'v0.1.0': 'v0.1.0 (stable)',
        'v0.0.6': 'v0.0.6',
        'v0.0.5': 'v0.0.5',
        'v0.0.4': 'v0.0.4',
        'v0.0.3': 'v0.0.3',
        'v0.0.2': 'v0.0.2',
    }
}


# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
# html_theme_options = {}
html_theme_options = {
    "navbar_end": ["version-switcher.html", "navbar-icon-links.html", "search-field.html"],
    'github_url': 'https://github.com/mne-tools/mne-nirs',
    "show_toc_level": 1,
    "google_analytics_id": "UA-188272121-1",
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# Output file base name for HTML help builder.
htmlhelp_basename = 'mnenirsdoc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    # 'papersize': 'letterpaper',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    ('index', 'project-template.tex', u'project-template Documentation',
     u'Robert Luke', 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'project-template', u'project-template Documentation',
     [u'Robert Luke'], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    ('index', 'project-template', u'project-template Documentation',
     u'Robert Luke', 'project-template', 'One line description of project.',
     'Miscellaneous'),
]

# Example configuration for intersphinx: refer to the Python standard library.
# intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/{.major}'.format(
        sys.version_info), None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'mne': ('https://mne.tools/stable', None),
    'nilearn': ('http://nilearn.github.io/', None),
    'sklearn': ('http://scikit-learn.org/stable', None),
    'mne_bids': ('https://mne.tools/mne-bids/stable', None),
    'statsmodels': ('https://www.statsmodels.org/stable', None)
}

filepath_prefix = 'examples/mne-nirs-website-examples'

# sphinx-gallery configuration
sphinx_gallery_conf = {
    'doc_module': 'mne_nirs',
    'backreferences_dir': os.path.join('generated'),
    'image_scrapers': ('pyvista', 'matplotlib'),
    'reference_url': {
        'mne_nirs': None},
    'download_all_examples': False,
    'show_memory': True,
    'within_subsection_order': FileNameSortKey,
    'binder': {
    # Required keys
    'org': 'rob-luke',
    'repo': 'mne-nirs-docker',
    'branch': 'main',  # noqa: E501 Can be any branch, tag, or commit hash. Use a branch that hosts your docs.
    'binderhub_url': 'https://mybinder.org',  # noqa: E501 Any URL of a binderhub deployment. Must be full URL (e.g. https://mybinder.org).
    'filepath_prefix': 'examples',  # noqa: E501 A prefix to prepend to any filepaths in Binder links.
    'notebooks_dir': 'mne-nirs-website-examples',  # noqa: E501 A prefix to prepend to any filepaths in Binder links.
        'dependencies': [
            '../requirements.txt',
            '../requirements_doc.txt',
        ],
    }
}


def setup(app):
    # a copy button to copy snippet of code from the documentation
    app.add_js_file('js/copybutton.js')
    app.add_css_file('font-awesome.css')
