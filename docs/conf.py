# Sphinx configuration for the g1_barcode_sim documentation (Chinese).
# Build locally:  sphinx-build -b html docs docs/_build/html
# The English documentation is a separate Sphinx project in docs/en/
# (sphinx-build -b html docs/en docs/_build/html/en).

project = "g1_barcode_sim"
author = "Yuxuan Li"
copyright = "2026, Yuxuan Li"

extensions = [
    "myst_parser",
    "sphinx.ext.githubpages",
]

source_suffix = {".md": "markdown"}
root_doc = "index"
language = "zh_CN"
exclude_patterns = ["_build", "en", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
    "attrs_inline",
]

html_theme = "sphinx_book_theme"
html_title = "g1_barcode_sim 文档"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "repository_url": "https://github.com/yuxuanli224284/g1_barcode_sim",
    "use_repository_button": True,
    "show_toc_level": 2,
    # Language switch shown in the header of every page.
    "announcement": (
        '中文 | <a href="https://yuxuanli224284.github.io/g1_barcode_sim/en/">English</a>'
    ),
}
