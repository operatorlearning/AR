# Alternating RandONets

Alternating RandONets is an efficient operator learning framework based on random projection and alternating optimization.  
Instead of training all network parameters using gradient descent, the framework randomly initializes hidden-layer weights and computes output-layer weights analytically through closed-form least-squares updates. This design significantly improves training efficiency while retaining strong approximation capability.

This repository contains several implementations and experiments of Alternating RandONets for benchmark operator learning tasks, including supervised and physics-informed settings.

---

## Highlights

- Fast training through analytical output-weight updates
- Alternating optimization for branch and trunk output layers
- Random projection based architecture with low computational overhead
- Compact parameterization with strong predictive performance
- Support for both supervised and physics-informed learning
- Built-in visualization and MATLAB export utilities

---

## Method Overview

Alternating RandONets builds on the idea of random projection and introduces an alternating optimization strategy:

1. Hidden-layer parameters are randomly initialized and fixed.
2. The output weights associated with the branch network are updated by solving a regularized least-squares problem.
3. The output weights associated with the trunk network are then updated in closed form.
4. These two steps are alternated during training.

Compared with conventional fully gradient-based operator learning methods, this framework greatly reduces training cost and improves efficiency.

---

## Repository Structure

```text
.
├── LICENSE
├── README.md
├── burgerstest.py
├── heattest.py
├── pi-possiontest.py
├── poisson5d.py
├── possiontest.py
└── rdtest.py
```

## Requirements

The code was tested with the following environment:

- Python 3.9+
- PyTorch 2.0+
- NumPy
- SciPy
- Matplotlib

Install the required packages with:

```bash
pip install torch numpy scipy matplotlib
```
