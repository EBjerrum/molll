# molll: Data Driven Estimation of Molecular Log-Likelihood using Fingerprint Key Counting

This software provides models for estimating the likelihood of a molecule belonging to a specific dataset based on simple fingerprint key counting. The models, AtomLL and MolLL, are designed for outlier detection and class membership assignment. They offer potential applications in molecular generation and optimization. PropLL is included and uses scikit kernel density estimates on RDKit-derived and user-selectable properties.

## Installation

Clone and install directly from the main directory:

```bash
pip install .
```

or directly from the repository without cloning:

```bash
pip install git+https://github.com/EBjerrum/molll.git
```

(PyPI package is underway)

## Usage

The code works on lists of RDKit Mol objects:

```python
from molll import MolLL
molll = MolLL()
molll.analyze_dataset(mols_list)
molll.calculate_lls(other_or_same_mols)
#Or a single Mol object
molll.calculate_ll(single_mols)
```

Saving and loading from a text-based format:

```python
molll.save("MySaveFile.json")

molll_clone = MolLL()
molll_clone.load("MySaveFile.json")
```

For convenience, some classes with precomputed data are available, currently based on LibInvent train data:

```python
from molll import LibInventMolLLr1
molll = LibInventMolLLr1()
molll.calculate_lls(mols_list)
```

<!--
## Tests

TBD
-->

## Additional Reading

There's a preprint on ChemRxiv with some example usages: [https://doi.org/10.26434/chemrxiv-2024-hzddj](https://doi.org/10.26434/chemrxiv-2024-hzddj)
