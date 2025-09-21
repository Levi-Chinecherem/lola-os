"""
File: Sphinx configuration for LOLA OS TMVP 1 documentation.

Purpose: Configures Sphinx to build HTML documentation from Markdown files.
How: Uses myst_parser for Markdown, sphinx_rtd_theme for styling, and sets up project metadata.
Why: Enables developer-friendly documentation, per Developer Sovereignty tenet.
Full Path: lola-os/docs/conf.py
"""

# Configuration file for the Sphinx documentation builder.

# Project information
project = 'LOLA OS'
copyright = '2025, Levi Chinecherem Chidi'
author = 'Levi Chinecherem Chidi'
release = '1.0.0'

# General configuration
extensions = [
    'myst_parser',  # Enable Markdown parsing
    'sphinx.ext.autodoc',  # Auto-generate API docs
    'sphinx.ext.napoleon',  # Support Google/NumPy docstrings
    'sphinx.ext.viewcode',  # Link to source code
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML output options
html_theme = 'sphinx_rtd_theme'
html_static_path = []  # Remove _static to avoid missing directory warning

# MyST-Parser configuration for Markdown
myst_enable_extensions = [
    'colon_fence',  # Support ::: for directives
    'deflist',  # Definition lists
    'html_admonition',  # Admonition blocks
]
myst_heading_anchors = 3  # Auto-generate anchors for h1-h3

# Source file suffix
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}