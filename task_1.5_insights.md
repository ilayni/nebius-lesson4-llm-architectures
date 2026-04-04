#### Task 1.5 — Insights 

**L1 reduces overfitting — the main takeaway**

In Task 1.4 without regularization, we saw a ~15% train-val gap
with best val accuracy of 0.7787.
With L1 (lambda=0, same hyperparameters), the gap dropped to ~2%
(train 0.7681 vs val 0.7466 for zeros init).
With lambda=0.001, the gap shrinks even further —
val accuracy (0.7041) is close to train accuracy (0.6777).
The higher accuracy in Task 1.4 might have been partly "lucky" —
the model memorized training noise that happened to match
validation data, rather than learning genuinely useful patterns.

**Accuracy drops with stronger regularization**

The best val accuracy here was 0.7511 (random, lambda=0).
As lambda increases, accuracy drops —
at lambda=0.1 the model barely learns (~0.50, random guessing).
Too much regularization suppresses all weights
so the model can't pick up any useful signal.

**Sparsity was surprisingly low**

Only lambda=0.0001 pushed some weights near zero
(19 for zeros init, 10 for random init).
Higher lambdas didn't eliminate any weights at all.
This is a known SGD limitation with L1 —
weights oscillate around zero instead of converging to it.
Proximal gradient methods would be needed for true sparsity.

**Initialization didn't matter much**

Both zeros and random init gave very similar results
at every lambda value.
Regularization strength dominates over initialization choice.

**Weight dynamics: shrinkage but not elimination**

The raw weight plot shows heavy oscillation from mini-batch SGD noise.
The smoothed graphs tell a clearer story —
weights trend toward smaller values over training,
but they bounce around zero without settling.
This confirms that SGD with L1 gives weight shrinkage
but not true feature elimination.

**All experiments were numerically stable**

No NaN values appeared in any configuration.
