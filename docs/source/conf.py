# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

sys.path.insert(0, os.path.abspath('../..'))  # Path to your code

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'graph-nd'
copyright = '2025, Zach Blumenfeld'
author = 'Zach Blumenfeld, Alex Gilmore'
release = '0.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # Automatically generate API documentation
    'sphinx.ext.viewcode',  # Add links to view source code
    'sphinx.ext.napoleon',  # Support for Google-style docstrings
    'sphinx.ext.intersphinx',  # Link to other project's documentation
    'sphinx_autodoc_typehints',  # Include type hints in API documentation
    'sphinx_copybutton',  # Add copy button to code blocks
    'myst_parser',  # Support for Markdown
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'  # Use Read the Docs theme
html_theme_options = {
    'github_user': 'zach-blumenfeld',
    'github_repo': 'graph-nd',
    'github_button': True,
    'github_banner': True,
    'github_type': 'star',
}
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
}

# -- Napoleon settings for Google style docstrings --------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
