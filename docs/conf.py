# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Other packages
import datetime

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('sphinxext'))

# -- Project information -----------------------------------------------------

project = 'Einsums'
copyright = f'2022-{datetime.datetime.today().year}, Einsums Developers'
author = 'Einsums Developers'

# The full version, including alpha/beta/rc tags
release = '0.1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
  'sphinx.ext.mathjax',
  'breathe',
  'exhale',
  'sphinx_design'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'sphinxext']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
    "github_url": "https://github.com/Einsums/Einsums",

    "logo": {
        "image_light": "einsums-logo.png",
        "image_dark": "einsums-logo.png",
        "text": "Einsums",
    },

    "show_toc_level": 2,
    "header_links_before_dropdown": 4,

    "secondary_sidebar_items": ["page-toc", "sourcelink"]
}

html_css_files = ['css/custom.css']

# -- Setup the breathe extension ---------------------------------------------

breathe_projects = {
  "Einsums": "./_doxygen/xml"
}
breathe_default_project = "Einsums"

# -- Setup the exhale extension ----------------------------------------------

exhale_args = {
  # These are required
  "containmentFolder": "./api",
  "rootFileName": "library_root.rst",
  "doxygenStripFromPath": "..",
  # Heavily encouraged optional arguments
  "rootFileTitle": "Library API",
  # Suggested optional arguments
  "createTreeView": True,
  # If using the sphinx-bootstrape-theme, you need this next one
  # "treeViewIsBootstrap": True,
  "exhaleExecutesDoxygen": True,
  # Use of CLANG_* requires a Doxygen compiled with clang support
  # Use of CLAND_DATA_PATH requires a configuration of einsums to be completed.
  "exhaleDoxygenStdin": """INPUT = ../src/include
  PREDEFINED += EINSUMS_EXPORT=
  PREDEFINED += DOXYGEN_SHOULD_SKIP_THIS
  PREDEFINED += BEGIN_EINSUMS_NAMESPACE_HPP(x)=namespace x {
  PREDEFINED += END_EINSUMS_NAMESPACE_HPP(x)=}
  """
  # CLANG_ASSISTED_PARSING = YES
  # CLANG_DATABASE_PATH = ../build
}

# Tell sphinx what the primary language being documented is
primary_domain = "cpp"

# Tell sphinx what the pygems highlight language should be
highlight_language = "cpp"
