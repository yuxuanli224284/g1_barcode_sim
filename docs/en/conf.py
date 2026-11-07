# Sphinx configuration for the English g1_barcode_sim documentation.
# Build locally:  sphinx-build -b html docs/en docs/_build/html/en
# Images and the stylesheet are shared with the Chinese project in docs/.

project = "g1_barcode_sim"
author = "Yuxuan Li"
copyright = "2026, Yuxuan Li"

extensions = [
    "myst_parser",
    "sphinx.ext.githubpages",
]

source_suffix = {".md": "markdown"}
root_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
    "attrs_inline",
]

html_theme = "sphinx_book_theme"
html_title = "g1_barcode_sim documentation"
html_static_path = ["../_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "repository_url": "https://github.com/yuxuanli224284/g1_barcode_sim",
    "use_repository_button": True,
    "show_toc_level": 2,
    # language button next to the repository button (top right of every page)
    "icon_links": [{"name": "中文", "url": "https://yuxuanli224284.github.io/g1_barcode_sim/", "icon": "fa-solid fa-language", "type": "fontawesome"}],
    # Language switch shown in the header of every page.
    "announcement": (
        '<a href="https://yuxuanli224284.github.io/g1_barcode_sim/">中文</a> | English'
    ),
}
