from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

with (ROOT / "pyproject.toml").open("rb") as pyproject_file:
    project_metadata = tomllib.load(pyproject_file)["project"]

project = "Terrarium"
author = "Terrarium contributors"
version = project_metadata["version"]
release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_preserve_defaults = True
autoclass_content = "both"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = False

# Optional provider SDKs are not required to build the core documentation.
autodoc_mock_imports = [
    "anthropic",
    "azure",
    "azure.identity",
    "google",
    "google.genai",
    "openai",
    "problem_layer",
    "problem_layer.base",
    "problem_layer.meeting_scheduling",
    "problem_layer.personal_assistant",
    "problem_layer.personal_assistant.problem",
    "problem_layer.smart_grid",
    "torch",
    "vllm",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

html_theme = "furo"
html_title = "Terrarium Documentation"
html_baseurl = os.environ.get(
    "TERRARIUM_DOCS_BASEURL",
    "https://umass-aisec.github.io/Terrarium/latest/",
)
html_copy_source = False
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/leaf_favicon.svg"
html_theme_options = {
    "light_logo": "terrarium_logo_rounded.png",
    "dark_logo": "terrarium_logo_rounded.png",
    "source_repository": "https://github.com/umass-aisec/Terrarium/",
    "source_branch": "main",
    "source_directory": "docs/",
}

myst_heading_anchors = 3

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
