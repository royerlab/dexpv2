# DEXPV2

[![DOI](https://zenodo.org/badge/626141936.svg)](https://zenodo.org/doi/10.5281/zenodo.13761266)

<!---
[![License BSD 3-Clause License](https://img.shields.io/pypi/l/dexpv2.svg?color=green)](https://github.com/royerlab/dexpv2/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/dexpv2.svg?color=green)](https://pypi.org/project/dexpv2)
[![Python Version](https://img.shields.io/pypi/pyversions/dexpv2.svg?color=green)](https://python.org)
[![tests](https://github.com/royerlab/dexpv2/workflows/tests/badge.svg)](https://github.com/royerlab/dexpv2/actions)
[![codecov](https://codecov.io/gh/royerlab/dexpv2/branch/main/graph/badge.svg)](https://codecov.io/gh/royerlab/dexpv2)
--->

Image processing library for large microscopy images.

# Installation

```
pip install git+https://github.com/royerlab/dexpv2.git
```

# Philosophy

DEXPv2 is a simplified version of [DEXP](https://github.com/royerlab/dexp).

Rather than implementing the whole processing pipeline as the original DEXP, version 2 focuses on building image processing components for large image processing.

These image processing functions are implemented using Array API standard when permitted, making them available for CPU or GPU when `cupy` and `cucim` is available.
