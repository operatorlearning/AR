# Alternating RandONets

Alternating RandONets is an efficient operator learning framework based on random projection and alternating optimization.  
Instead of training all network parameters with gradient descent, the method randomly initializes hidden-layer weights and updates output-layer weights analytically through closed-form least-squares solutions. This design preserves high training efficiency while improving flexibility and predictive performance.

This repository contains the implementation of Alternating RandONets for benchmark operator learning tasks, together with visualization and export utilities.

---

## Highlights

- **Fast training** with analytical output-weight updates
- **Alternating optimization** for both branch and trunk output layers
- **Random projection framework** with reduced computational overhead
- **Efficient implementation** with compact trainable parameterization
- **Built-in visualization tools** for predictions, references, and pointwise errors
- **MATLAB export support** for interactive post-processing

---

## Method Overview

Alternating RandONets builds on the random projection idea used in RandONets and introduces an alternating optimization strategy:

1. Hidden-layer weights are randomly initialized and kept fixed.
2. The branch-side output weights are updated analytically using a regularized least-squares solution.
3. The trunk-side output weights are then updated analytically in closed form.
4. These two steps are alternated during training.

Compared with conventional gradient-based operator learning models, this approach significantly reduces training time while maintaining strong approximation capability.

---

## Repository Structure

```text
.
├── main.py                  # Main training / testing script
├── README.md                # Project documentation
└── ...                      # Additional scripts, data, or utilities
