# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

University assignment for an LLM Architectures course. The main notebook is `LLM_Architectures_hometask_1_ilay_nisim_boris_kogan.ipynb`. It covers optimization in PyTorch: gradient descent, SGD, numerical stability, and L1 regularization, applied to sentiment classification (SST-2 dataset).

`TextClassification.ipynb` is a lesson demo — not part of the assignment.

## Assignment Structure (Total: 100 points)

### Part 1 — SGD for Logistic Regression (70 pts)
- **Task 1.1 (10 pts):** Implement `binary_cross_entropy_loss` with numerical stability (epsilon clamping). Understand binary vs. softmax formulation.
- **Task 1.2 (10 pts):** Implement `LogisticRegression` nn.Module (weight init: zeros/random/tensor, forward with sigmoid, predict with 0.5 threshold).
- **Task 1.3 (10 pts):** Implement `sgd_logistic_regression` training loop — mini-batch SGD, epoch shuffling, logging history/metrics.
- **Task 1.4 (20 pts):** Run experiments across learning rates [0.01, 0.03, 0.1, 0.3, 1.0] and batch sizes [16, 64, 256], plot results, write conclusions.
- **Task 1.5 (20 pts):** Add L1 regularization to the training loop, study sparsity effects, analyze which features get zeroed out.

### Part 2 — Comparing Optimizers (30 pts)
- Implement from scratch: Gradient Descent, Momentum, AdaGrad, Adam.
- Test on two functions: convex bowl `f(x,y) = x² + 4y²` and six-hump camel function.
- Plot trajectories and convergence, compare behavior, write conclusions.

## How to Run

This is a Jupyter notebook (Python 3). Run in Google Colab or locally with:
```
pip install datasets torch numpy matplotlib
jupyter notebook LLM_Architectures_hometask_1_ilay_nisim_boris_kogan.ipynb
```

## Key Dependencies
- `datasets` (HuggingFace) — for loading SST-2
- `torch` / `torch.nn` — all implementations must use PyTorch primitives
- `numpy`, `matplotlib` — data handling and plotting

## Important Guidelines from the Assignment
- **Do not delete task formulations** from the notebook.
- Experiments and code can be delegated to AI, but **conclusions must be written by the student** in their own words with genuine understanding.
- The student is new to ML/optimization — provide intuition and short explanations in comments, and add prints/graphs to illustrate concepts.

## Data Pipeline (already provided)
1. Load SST-2 via HuggingFace `datasets`
2. `clean_text()` — lowercase, remove special chars, collapse whitespace
3. `build_vocabulary()` — top 10k tokens from training set
4. `dataset_to_vec()` — Bag-of-Words vectors (token counts)
5. Result: `X_train`, `y_train`, `X_val`, `y_val` as numpy arrays
