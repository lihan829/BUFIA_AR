# Tone-Mora-Syllable Association Visualizer

This project provides tools to represent and visualize tonal phonology structures through Python classes and Graphviz diagrams. It includes functions to update mora and tone indices in a word’s tonal structure and to generate clean, non-redundant visualizations.

---

## Features

- **Update mora indices globally** across syllables for accurate representation.
- **Update tone indices** to reflect their sequential order.
- **Draw tone-mora-syllable association graphs** with Graphviz.
- Avoid duplicate edges in visualization.
- Save graphs as PNG images (default) or display inline in Jupyter notebooks.
- Configurable output directory for saved graphs.

---

## Requirements

- Python 3.6+
- [graphviz](https://graphviz.org/) installed on your system and in your PATH
- Python package: `graphviz` (install via pip)
