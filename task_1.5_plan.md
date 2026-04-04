# Plan: Implement Task 1.5 — L1 Regularization and Sparsity

## Context

Task 1.5 (20 pts) asks to study L1 regularization effects on the logistic regression model from Tasks 1.1–1.3. The `sgd_logistic_regression` function (cell 25) **already has L1 penalty support** built in during Task 1.3 — it accepts `penalty='l1'` and `reg_lambda` parameters and adds `reg_lambda * sum(|w|)` to the loss. So no core function changes are needed.

The remaining work is the **experiment and visualization code** in cell 32 (currently empty: `# <Your code here>`).

## What's Already Done

- `binary_cross_entropy_loss` (cell 19) — complete
- `LogisticRegression` (cell 22) — supports `init="zeros"` and `init="random"`
- `sgd_logistic_regression` (cell 25) — already has L1/L2 penalty logic, returns `(w, b, history, epoch_log)`
- `history` already tracks per-batch weight snapshots (`model.w` clones) — useful for weight dynamics plots

## What Needs to Be Implemented (Cell 32)

**Note:** Per your preference, this is experiment/visualization code you may want to write yourself. I'll outline the structure so you know exactly what to build.

### Step 1: Run Experiments

Loop over `reg_lambda = [0, 1e-4, 1e-3, 1e-2, 1e-1]` with two init modes (`"zeros"`, `"random"`), fixed `lr=0.1`, `batch_size=100`, `penalty='l1'`:

```
For each (init, lambda) combo:
  w, b, history, epoch_log = sgd_logistic_regression(
      X_train, y_train, X_val, y_val,
      lr=0.1, epochs=20, batch_size=100,
      init=init, penalty='l1', reg_lambda=lam,
      metric='accuracy', print_metrics=True
  )
  Store: final w, history, epoch_log, train/val metrics, sparsity count
```

Sparsity = count of weights where `|w| < 1e-7`.

### Step 2: Compare Weight Initialization (zeros vs random)

For each lambda, print/compare:
- Stability (any NaN in loss?)
- Final train & val accuracy
- Number of near-zero weights

### Step 3: Visualizations

Three plots required:

1. **Non-zero weights vs lambda** — line plot, one line per init mode. X-axis: lambda (log scale), Y-axis: count of `|w| >= 1e-7`

2. **Train/val metric vs lambda** — line plot showing train and val accuracy for each lambda. One subplot per init mode (or all on one plot with legend).

3. **Weight dynamics for eliminated features** — pick features whose weights go from non-zero to near-zero during training (under high lambda like 0.1). Plot weight value vs training step using `history` entries. Select ~5-10 such features for readability.

### Step 4: Written Analysis (Student writes)

Summarize insights about:
- How lambda affects sparsity
- Train vs val performance trade-off
- Which features get eliminated and why that makes sense
- Zeros vs random init comparison

## Files to Modify

- `LLM_Architectures_hometask_1_ilay_nisim_boris_kogan.ipynb` — cell 32 only

## Verification

- Run the notebook cells 1–32 in order
- Check that experiments complete without NaN/errors for all lambda values
- Verify plots render correctly
- Confirm sparsity increases with larger lambda
