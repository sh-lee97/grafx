import os
import sys

dirname = os.path.dirname(__file__)
basedir = os.path.abspath(os.path.join(dirname, "..", "..", "src"))
sys.path.insert(0, basedir)

project = "GRAFX"
copyright = "2024, Sungho Lee"
author = "Sungho Lee"
version = "0.5.0"
html_title = "GRAFX Documentation"

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.ifconfig",
    "sphinx.ext.githubpages",
    "sphinx_toolbox.collapse",
    "sphinxcontrib.bibtex",
    "sphinx_math_dollar",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "furo"
html_static_path = ["_static"]

html_css_files = ["css/custom.css"]
add_module_names = False

autoclass_signature = "separated"  # new
todo_include_todos = True
bibtex_bibfiles = ["references/refs.bib"]
autodoc_member_order = "bysource"
toc_object_entries_show_parents = "hide"

napoleon_use_ivar = True

html_theme_options = {
    "top_of_page_buttons": [],
    #    "light_css_variables": {
    #        "api-font-size": "100%",
    #    },
}

html_favicon = "favicon.ico"
# pygments_style = "xcode"
pygments_style = "tango"
# pygments_style = "manni"
# pygments_style = "emacs"
# pygments_style = "perldoc"
# pygments_style = "murphy"
