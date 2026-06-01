* how quickly each optimizer converges
For bowl: all 4 converged to the global minimum. GD converged first, then AdaGrad, both around 250-300
steps. Adam and Momentum were slower with about 1000 steps.
That might be because the bowl's simple convex shape doesn't benefit much from momentum or adaptive methods.

* whether the optimization path is smooth or oscillatory
For bowl most of the optimizers seemed stable besides Momentum which overshoots and oscillates because the velocity accumulates and carries the parameters past the minimum before correcting. 
Adam was smoother than Momentum despite also using momentum internally, because its adaptive per-parameter learning rate dampens oscillations.

* whether an optimizer that works well on the convex bowl also works well on Camel
GD did well on bowl but not on the camel. 
On Camel, GD, Adam and AdaGrad all converged to the same local min at (-1.607, -0.569) with f=2.104, 
while the global minimum is f=-1.0316 — so the local min they found is actually much worse.
It is surprising that Adam missed the global minimum, it usually does well but from this starting point it got trapped.
Only Momentum found one of the global minima at (0.0898, -0.7127) with f=-1.0316. Probably because its accumulated velocity carried it past the local basin into the global minimum's basin. 

* how the geometry of the function affects the optimization process
The camel's geometry (multiple valleys separated by ridges) means that
without enough inertia to cross ridges, optimizers get trapped in the nearest local basin.
The bowl has no such traps — its single smooth valley means all optimizers reach the same answer,
just at different speeds.
For camel, all optimizers converged quickly (within 200 steps), but not necessarily to the global minimum —
this is dependent on the starting point. Momentum was the only one to reach the global minimum.
For bowl, Adam and Momentum took more steps (around 1000) compared to GD and AdaGrad (250-300).


**Analysis**

Summarize your findings in 2-3 text paragraphs. Things to ponder:

* Which optimizer performs best on the convex bowl?
GD. Because the bowl is smooth and convex, the extra machinery of Momentum/Adam adds overhead without benefit. Simple gradient descent with a good learning rate is optimal here.

* Which optimizer performs best on the Camel function? Will any of them reliably find global optima or will all of them get trapped in local minima from time to time, depending on the starting point.

Momentum performed best, but probably because it got lucky since its velocity helped it escape the local basin.
From a different starting point, Momentum could also get trapped.
None of the optimizers guarantee finding global optima on non-convex functions.
That would require methods like random restarts, simulated annealing, or grid search.
I demonstrated that results depend on the starting point by changing it to (0.2, -0.5), near the global minimum — where all 4 optimizers found the global minimum.
when the starting point is already in the basin of the global minimum, even simple GD finds it. The
advantage of Momentum only shows when you need to cross from one basin to another.     

* Do the same hyperparameters work equally well for both functions?
No, and this is why I chose different ones for each case.
**Bowl (convex, smooth gradients):**
- `lr=0.1` for GD, Momentum, Adam — the bowl is smooth and convex, so gradients are well-behaved and a moderate lr converges quickly without overshooting.
- `lr=0.5` for AdaGrad — AdaGrad divides by accumulated squared gradients, which shrinks the effective lr over time. It needs a higher starting lr to compensate, otherwise it stalls before reaching the minimum.
**Camel (non-convex, steep ridges):**
- `lr=0.01` for GD, Momentum, Adam — the camel function has sharp curvature changes and large gradients near ridges. A smaller lr prevents divergence and overshooting past local/global minima.
- `lr=0.1` for AdaGrad — same adaptive-shrinking reason as above, but still lower than the bowl since the landscape is more aggressive.
### Beta (momentum coefficient)
- `beta=0.9` is the standard textbook default — it retains 90% of the previous velocity, giving enough inertia to smooth out noisy gradients and accelerate through flat regions, without so much momentum that it overshoots minima.


* What advantages do Momentum, AdaGrad, and Adam provide compared to plain Gradient Descent
While in Gradient Descent learning rate is fixed, they all can adapt in different ways:
- AdaGrad: adapts lr per-parameter — parameters with large past gradients get smaller lr.
  Good for sparse features. Downside: lr only shrinks, never grows — can stall.
- Momentum: does NOT adapt the learning rate. It adds inertia (velocity) that helps
  cross flat regions faster and can escape shallow local minima. Downside: oscillates and can
  overshoot minima due to inertia — this could be a pro or a con : it helped escape the local min
  on camel, but could also skip past a global minimum in other cases.
- Adam: combines BOTH momentum AND adaptive lr. Usually the most robust default choice.
  But as we saw, it still got trapped in the local min here — adaptive lr doesn't help
  escape local minima, it just navigates smoother within a basin.
