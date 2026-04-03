# Task 1.4 — Conclusions and Explanations

## How to Read the Heatmaps

### Structure
- Each heatmap is a grid: rows = batch size (50, 100, 200), columns = learning rate (0.01 → 1.0)
- Each cell = one experiment's final result after 20 epochs
- Color intensity = the value. Darker = more extreme

### Top row (Accuracy) — darker blue = better
- **Left→right** (increasing lr): accuracy generally improves. lr=0.01 is too slow (didn't converge in 20 epochs), lr=1.0 is best.
- **Top→down** (increasing batch size): accuracy drops slightly. Smaller batches (50) do more weight updates per epoch, so they learn faster.
- **Best cell**: batch_size=50, lr=1.0 → 0.787 val accuracy

### Bottom row (Loss) — darker red = lower loss = better
- Same pattern: higher lr + smaller batch = lower loss
- Train loss drops more aggressively than val loss — that gap is the beginning of overfitting

## Analysis: How Learning Rate and Batch Size Affect Training

### 1. Convergence Speed

From the heatmaps: low learning rates (0.01, 0.03) show significantly worse final accuracy than higher ones. This means they haven't converged in 20 epochs — they're still learning when training stops.

From the convergence curves: lr=1.0 drops to low loss within 2-3 epochs, while lr=0.01 is still slowly declining at epoch 20.

Smaller batch sizes also converge faster. With 67k training samples: batch_size=50 gives ~1,340 weight updates per epoch, batch_size=200 gives only ~335. More updates per epoch = faster learning.

### 2. Stability

From the convergence curves: lr=1.0 (especially with batch_size=100) shows jagged, spiky loss curves — bouncing up and down between epochs. The steps are so large that the model overshoots the optimum, then corrects, then overshoots again.

lr=0.01 curves are perfectly smooth but painfully slow. lr=0.1 and 0.3 offer a good balance: decent speed without wild oscillations.

Smaller batches also add noise — each batch is a less representative sample of the full data, so gradient estimates are noisier. Comparing lr=1.0 across batch sizes in the convergence curves, it's most jagged at batch_size=50.

### 3. Final Performance

From the heatmaps: the best val accuracy is 0.787 (lr=1.0, batch_size=50). This means roughly 4 out of 5 reviews are classified correctly — decent for a simple bag-of-words logistic regression.

However, lr=1.0 is the most unstable (as seen in the convergence curves). A practical choice would be **lr=0.3, batch_size=50** — nearly as good accuracy with much more stable training.

0.787 accuracy is reasonable given the model's limitations. Logistic regression is linear — it can only learn individual word associations. Sentences like "not bad" (double negative = positive) will fool it because it sees "not" and "bad" as separate features without understanding their combination.

### 4. The Train-Val Gap

The train-val accuracy gap is ~15% (train ~0.935, val ~0.787). This means the model fits the training data better than it generalizes to new data. We still need to address this gap.

## What is Overfitting?

**Overfitting** = the model learns patterns specific to the training data that don't apply to new data. It memorizes noise and quirks instead of learning general rules.

Think of it like studying for an exam:
- **Train accuracy** = how well you answer questions from your study notes (you've seen these before)
- **Val accuracy** = how well you answer new questions on the exam (you haven't seen these)

If you score 93% on practice questions but 78% on the real exam, you memorized the practice questions more than you learned the underlying concepts.

### Detecting Overfitting: The Train-Val Gap

The **gap** between train and val performance is how you detect it:
- Small gap (train=75%, val=73%) → model generalizes well
- Large gap (train=93%, val=78%) → model is starting to overfit

The gap in our results is moderate (~15%). With more epochs or a more complex model, the gap could grow worse — train accuracy keeps climbing toward 100% while val accuracy plateaus or even drops.

In the heatmaps, the gap is **larger at high learning rates** — the model learns fast and hard, which means it also starts memorizing training-specific patterns faster.
