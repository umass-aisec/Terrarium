---
orphan: true
---

# Terrarium Docs

This folder contains the Sphinx documentation source for Terrarium.

## Build the Documentation

Install the documentation requirements and the local package:

```bash
uv pip install -r docs/requirements.txt -e .
```

Build once with clean URLs:

```bash
python -m sphinx -b dirhtml docs docs/_build/dirhtml
```

Rebuild automatically while editing:

```bash
sphinx-autobuild -b dirhtml --watch terrarium docs docs/_build/dirhtml
```
