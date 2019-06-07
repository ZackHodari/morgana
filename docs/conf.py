# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

import inspect
from os.path import relpath, dirname
import pkg_resources
import sys

import sphinx_readable_theme

import morgana


__version__ = pkg_resources.get_distribution('morgana').version

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'morgana'
copyright = '2019, Zack Hodari'
author = 'Zack Hodari'

# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

master_doc = 'index'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'numpydoc',
    'sphinx.ext.linkcode',
    'sphinxcontrib.fulltoc',
    'sphinx.ext.todo',
    'sphinxarg.ext',
    'sphinx.ext.autosectionlabel',
]

autodoc_default_flags = ['members']
autosummary_generate = True
numpydoc_show_class_members = False

autodoc_member_order = 'bysource'
autodoc_inherit_docstrings = False

default_role = 'code'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme_path = [sphinx_readable_theme.get_html_theme_path()]
html_theme = 'readable'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_sidebars = {'**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html']}


intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'np': ('http://docs.scipy.org/doc/numpy/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'tensorboardX': ('https://tensorboardx.readthedocs.io/en/latest/', None),
}


def setup(app):
    app.add_stylesheet('css/custom.css')


def linkcode_resolve(domain, info):
    r"""Determine the URL corresponding to Python object."""
    if domain != 'py':
        return None

    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    # strip decorators, which would resolve to the source of the decorator
    # possibly an upstream bug in getsourcefile, bpo-1764286
    try:
        unwrap = inspect.unwrap
    except AttributeError:
        pass
    else:
        obj = unwrap(obj)

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        lineno = None

    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        linespec = ""

    fn = relpath(fn, start=dirname(morgana.__file__))

    if 'dev' in __version__:
        return "https://github.com/ZackHodari/morgana/blob/master/morgana/%s%s" % (
           fn, linespec)
    else:
        return "https://github.com/ZackHodari/morgana/blob/v%s/morgana/%s%s" % (
           __version__, fn, linespec)

