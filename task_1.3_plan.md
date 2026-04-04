# Plan: Implement Task 1.3 (SGD Training Loop) + Task 1.4 (Experiments)

## Context

This is a university assignment on logistic regression with SGD. Tasks 1.1 (BCE loss) and 1.2 (LogisticRegression model) are already implemented. Task 1.3 asks to implement the training loop, Task 1.4 asks to run experiments across hyperparameters and visualize results. The student is new to ML, so every step needs intuitive explanation.

---

## Your Questions Answered

### 1. Which evaluation metric and why?

**Accuracy** is the right choice here. Here's why:

- SST-2 is a **roughly balanced** dataset (~50% positive, ~50% negative). The notebook already prints class counts in cell 10.
- When classes are balanced, accuracy is a fair metric — it won't be misleadingly high just because one class dominates.
- If the dataset were imbalanced (e.g., 95% positive), accuracy would be misleading (predicting all-positive gives 95% "accuracy"). In that case you'd want F1, precision/recall, or AUC-ROC.
- The assignment says "you may choose any evaluation metric ... but must briefly explain why." Accuracy with a balance argument is clean and correct.
- We'll still compute log-loss (BCE) per epoch as the assignment requires — that's a separate metric tracking optimization progress.

### 2. How to save w and b for each epoch (and batch)

The skeleton already defines `history` (batch-level) and `epoch_log` (epoch-level):

- **`history`** (list of dicts): After every single batch SGD step, we snapshot `w` and `b` via `.detach().clone()`. This is already in the skeleton. The `.clone()` is critical — without it, all entries would point to the same (final) tensor.
- **`epoch_log`** (list of dicts): After each full epoch, we record train/val loss and train/val accuracy. This is what we use for plotting convergence curves.

Both are returned from the function. **We will also save `epoch_log` to disk** as JSON files in a `logs/` folder — one file per experiment config — so Task 1.4 can reload results without re-training.

### 3. How to visualize the saved logs

For Task 1.4, the main visualizations are:

1. **Heatmaps** (required by assignment): 2D grid of (batch_size x learning_rate) showing final train/val accuracy and loss. Uses `matplotlib.pyplot.imshow` or `seaborn.heatmap`.
2. **Convergence curves** (supplementary): Loss vs. epoch for each config, to show how fast/stable training is.

These are generated from `epoch_log` data.

### 4. What about batch-wise history of w and b? Does it need saving?

The assignment says to "save the model parameters w and b after each batch update" and "return the batch-wise history." So **yes, we must collect it** — the function returns it. But:

- **Don't save batch-wise history to disk.** With 10k features, 20 epochs, and 100s of batches per epoch, each experiment would produce gigabytes of weight snapshots. It's impractical and not useful.
- **Don't save it to the `logs/` folder.** It exists only in-memory, returned for potential inspection (Task 1.5 uses it to study sparsity).
- The `epoch_log` (compact, one entry per epoch) is what goes to disk.

For Task 1.5 later, the batch-wise history becomes relevant for watching weights get zeroed out by L1 regularization — but even then, you'd visualize a small subset of weights, not dump everything to disk.

### 5. Logs folder

We'll create `logs/` in the project root. Each experiment saves a JSON file like `logs/lr0.01_bs50.json` containing the `epoch_log` list.

---

## Implementation Plan

### Step 1: Create `logs/` directory
- `mkdir logs/` in the project root

### Step 2: Fill in Task 1.3 — `sgd_logistic_regression` (Cell 25)

Fill in all TODOs in the skeleton:

```python
# Shuffle training data each epoch
perm = torch.randperm(n_samples)
X_train_epoch = X_train_tensor[perm]
y_train_epoch = y_train_tensor[perm]

# Mini-batch selection
X_batch = X_train_epoch[start:end]
y_batch = y_train_epoch[start:end]

# Forward pass
y_pred = model(X_batch)

# BCE loss
data_loss = binary_cross_entropy_loss(y_pred, y_batch)

# No regularization for now (Task 1.5 adds it)
loss = data_loss

# Backward + step (NOTE: zero_grad BEFORE backward, not after)
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Epoch evaluation
y_pred_train = model(X_train_tensor)
y_pred_val = model(X_val_tensor)
train_loss = binary_cross_entropy_loss(y_pred_train, y_train_tensor)
val_loss = binary_cross_entropy_loss(y_pred_val, y_val_tensor)

y_hat_train = (y_pred_train >= 0.5).float()
y_hat_val = (y_pred_val >= 0.5).float()

# Accuracy metric
train_metric = (y_hat_train == y_train_tensor).float().mean().item()
val_metric = (y_hat_val == y_val_tensor).float().mean().item()
```

Key bug fix in skeleton: `optimizer.zero_grad()` is placed AFTER the batch update in the skeleton — it must come BEFORE `loss.backward()`. The correct order is: zero_grad -> backward -> step.

Also add: save `epoch_log` to `logs/` folder as JSON at the end of the function.

### Step 3: Add F1 metric support

The skeleton has an `if metric == 'f1'` branch. Implement it using manual computation (TP/FP/FN from tensors) so no sklearn dependency is needed.

### Step 4: Implement Task 1.4 — Experiments (Cell 28)

```python
learning_rates = [0.01, 0.03, 0.1, 0.3, 1.0]
batch_sizes = [50, 100, 200]

# Run all combinations, collect results
results = {}
for bs in batch_sizes:
    for lr in learning_rates:
        w, b, history, epoch_log = sgd_logistic_regression(
            X_train, y_train, X_val, y_val,
            lr=lr, epochs=20, batch_size=bs,
            init="zeros", print_metrics=True
        )
        results[(lr, bs)] = epoch_log
        # Save epoch_log to logs/
        with open(f'logs/lr{lr}_bs{bs}.json', 'w') as f:
            json.dump(epoch_log, f)
```

Then generate:
1. **Heatmaps**: Final val accuracy, final val loss, final train accuracy, final train loss
2. **Convergence curves**: Val loss over epochs for each (lr, bs) combo

### Step 5: Add brief metric explanation as markdown

Add a markdown cell before the code explaining why accuracy is appropriate for balanced SST-2.

---

## Files to modify
- `/Users/YOUR_USER/dev/cloud/nebius/lesson4_27_03_assignment/LLM_Architectures_hometask_1_ilay_nisim_boris_kogan.ipynb` — Cells 25 and 28
- Create directory: `logs/`

## Verification
- Run all cells top-to-bottom in the notebook
- Check that `logs/` folder contains JSON files after Task 1.4 experiments
- Verify heatmaps render correctly
- Confirm training loss decreases over epochs (sanity check)
