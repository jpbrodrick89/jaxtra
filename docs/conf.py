import os
import sys
import types

# Make the jaxtra source tree importable without installing the package.
sys.path.insert(0, os.path.abspath(".."))

# ---------------------------------------------------------------------------
# Stub out the compiled C extensions so Sphinx can import jaxtra without a
# built _jaxtra.so present (e.g. on ReadTheDocs).
# _load_extension() in _core.py checks sys.modules first, so pre-populating
# it here is enough to short-circuit the file search.
# ---------------------------------------------------------------------------
for _mod in ("jaxtra._jaxtra", "jaxtra._jaxtra_cuda"):
    _stub = types.ModuleType(_mod)
    _stub.initialize = lambda: None
    _stub.registrations = lambda: {}
    sys.modules[_mod] = _stub

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------
project = "jaxtra"
author = "jpbrodrick89"
copyright = "2026, jpbrodrick89"
release = "0.1.0"
root_doc = "index"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
]

autosummary_generate = True

# Support Google-style docstrings (used throughout the package).
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# autodoc: show type annotations in signatures, not repeated in description.
autodoc_typehints = "signature"
autodoc_member_order = "bysource"

# ---------------------------------------------------------------------------
# Intersphinx — cross-link to JAX, NumPy, and Python docs
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
}

# ---------------------------------------------------------------------------
# HTML output — sphinx-book-theme (same as JAX)
# ---------------------------------------------------------------------------
html_theme = "sphinx_book_theme"
html_title = "jaxtra"
html_theme_options = {
    "repository_url": "https://github.com/jpbrodrick89/jaxtra",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": False,
    "path_to_docs": "docs",
    "show_navbar_depth": 2,
}

exclude_patterns = ["_build", "_autosummary", "Thumbs.db", ".DS_Store"]
