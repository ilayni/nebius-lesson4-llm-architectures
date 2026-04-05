# Plan: Implement Task 2 — Comparing Optimization Algorithms

## Context
Part 2 of the assignment (30 pts) requires implementing 4 optimizers from scratch and comparing them on two test functions. The notebook (Cell 40) has skeleton code with `TODO` placeholders. Cell 39 has the test functions and a plotting helper. Cell 41 is empty (for visualization/analysis).

**Bowl function mismatch:** The markdown description (Cell 37) defines the bowl as `f(x,y) = x² + 4y²`, but the actual code in Cell 39 implements `x**2 + 2*y**2`. These are different functions (4y² has steeper curvature in y than 2y²). Need to decide which is correct — either update the code to match the formula, or treat the code as the intended version. Currently keeping the code as-is.

## Files to Modify
- `LLM_Architectures_hometask_1_ilay_nisim_boris_kogan.ipynb` — Cells 40 and 41

## Step 1: Fill in Optimizer Implementations (Cell 40)

All four follow the same pattern: compute loss → backward → update with `torch.no_grad()` → zero grad.

### gradient_descent
```python
loss = f(theta)
loss.backward()
with torch.no_grad():
    theta -= lr * theta.grad
```

### momentum
```python
loss = f(theta)
loss.backward()
with torch.no_grad():
    v = beta * v + theta.grad          # velocity = decay * old_v + gradient
    theta -= lr * v
```

### adagrad
```python
loss = f(theta)
loss.backward()
with torch.no_grad():
    G = G + theta.grad ** 2            # accumulate squared gradients
    theta -= lr * theta.grad / (torch.sqrt(G) + eps)
```

### adam
```python
loss = f(theta)
loss.backward()
with torch.no_grad():
    m = beta1 * m + (1 - beta1) * theta.grad        # 1st moment
    v = beta2 * v + (1 - beta2) * theta.grad ** 2    # 2nd moment
    m_hat = m / (1 - beta1 ** step)                   # bias correction
    v_hat = v / (1 - beta2 ** step)
    theta -= lr * m_hat / (torch.sqrt(v_hat) + eps)
```

## Step 2: Set Hyperparameters for Experiments (Cell 40, bottom)

### Starting point
`theta0 = [-2, -1.5]` — as recommended by the assignment. This is a non-optimal point on both functions' surfaces, far enough from the minima to show interesting optimizer behavior.

### Learning rate choices

**Bowl (convex, smooth gradients):**
- `lr=0.1` for GD, Momentum, Adam — the bowl is smooth and convex, so gradients are well-behaved and a moderate lr converges quickly without overshooting.
- `lr=0.5` for AdaGrad — AdaGrad divides by accumulated squared gradients, which shrinks the effective lr over time. It needs a higher starting lr to compensate, otherwise it stalls before reaching the minimum.

**Camel (non-convex, steep ridges):**
- `lr=0.01` for GD, Momentum, Adam — the camel function has sharp curvature changes and large gradients near ridges. A smaller lr prevents divergence and overshooting past local/global minima.
- `lr=0.1` for AdaGrad — same adaptive-shrinking reason as above, but still lower than the bowl since the landscape is more aggressive.

### Beta (momentum coefficient)
- `beta=0.9` is the standard textbook default — it retains 90% of the previous velocity, giving enough inertia to smooth out noisy gradients and accelerate through flat regions, without so much momentum that it overshoots minima.

### Hyperparameter values
```python
# Bowl
results_bowl = {
    "GD": gradient_descent(bowl, theta0, lr=0.1),
    "Momentum": momentum(bowl, theta0, lr=0.1, beta=0.9),
    "AdaGrad": adagrad(bowl, theta0, lr=0.5),
    "Adam": adam(bowl, theta0, lr=0.1)
}

# Camel
results_camel = {
    "GD": gradient_descent(camel, theta0, lr=0.01),
    "Momentum": momentum(camel, theta0, lr=0.01, beta=0.9),
    "AdaGrad": adagrad(camel, theta0, lr=0.1),
    "Adam": adam(camel, theta0, lr=0.01)
}
```

## Step 3: Visualization Code (Cell 41)

Two plots per function (4 total):

1. **Loss vs. iteration** — all optimizers on one plot, using `plt.semilogy` for the bowl (values span orders of magnitude)
2. **Trajectory in (x,y) plane** — use the provided `plot_trajectories_camel_log` for camel; create similar contour plot for bowl

```python
# --- Plot 1: Loss curves ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for name, (traj, vals) in results_bowl.items():
    axes[0].semilogy(vals, label=name)
axes[0].set_title("Bowl: f(x,y) = x² + 2y²")
axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("f value (log)")
axes[0].legend(); axes[0].grid(True)

for name, (traj, vals) in results_camel.items():
    axes[1].plot(vals, label=name)
axes[1].set_title("Six-Hump Camel")
axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("f value")
axes[1].legend(); axes[1].grid(True)
plt.tight_layout(); plt.show()

# --- Plot 2: Trajectories ---
plot_trajectories_camel_log(bowl, results_bowl, xlim=(-2, 2), ylim=(-2, 2), title="Bowl Trajectories")
plot_trajectories_camel_log(camel, results_camel, title="Camel Trajectories")
```

## Step 4: Analysis Placeholder (Cell 41, markdown or comment)

Leave a markdown cell with prompting questions for the student to write conclusions (per CLAUDE.md: conclusions must be written by the student).

## Verification
1. Run Cell 39 (defines functions + plotting helper)
2. Run Cell 40 (optimizers + experiments) — should complete without errors
3. Run Cell 41 (plots) — should show 4 plots
4. Check: bowl optimizers should converge to (0, 0); camel should find minima near (0.0898, -0.7126) or (-0.0898, 0.7126)
5. Verify trajectories look reasonable (no NaN, no divergence)
