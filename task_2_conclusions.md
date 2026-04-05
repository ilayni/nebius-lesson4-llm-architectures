* how quickly each optimizer converges
for bowl functions: gd converged first, than adagrad both around 250-300
steps, adam and momentum took about 1000 steps. most of them seem s

* whether the optimization path is smooth or oscillatory
for bowl most of the optiominer seemed stable besides momentum

* whether an optimizer that works well on the convex bowl also works well on Camel
gd did well on bowl but not on the camel 
momentun found 1 global minimum which is good, it took much more step for momentun to find the minmum on bowl.
adam and adagrad - didnt seem to stop on global minima

* how the geometry of the function affects the optimization process
in camel, functions converged very quickly but not neccsarily to the global minima
this could be dependepnt on the starting point. all until 200 step momentum much faster.
for bowl adam and momentum took more steps, around 1000.




**Analysis**

Summarize your findings in 2-3 text paragraphs. Things to ponder:

* Which optimizer performs best on the convex bowl?
gd

* Which optimizer performs best on the Camel function? Will any of them reliably find global optima or will all of them get trapped in local minina from time to time, depending on the starting point.
momentun, it seemed they all can be stuck on local mimina and depend on starting point

* Do the same hyperparameters work equally well for both functions?
no, 
**Bowl (convex, smooth gradients):**
- `lr=0.1` for GD, Momentum, Adam — the bowl is smooth and convex, so gradients are well-behaved and a moderate lr converges quickly without overshooting.
- `lr=0.5` for AdaGrad — AdaGrad divides by accumulated squared gradients, which shrinks the effective lr over time. It needs a higher starting lr to compensate, otherwise it stalls before reaching the minimum.
**Camel (non-convex, steep ridges):**
- `lr=0.01` for GD, Momentum, Adam — the camel function has sharp curvature changes and large gradients near ridges. A smaller lr prevents divergence and overshooting past local/global minima.
- `lr=0.1` for AdaGrad — same adaptive-shrinking reason as above, but still lower than the bowl since the landscape is more aggressive.
### Beta (momentum coefficient)
- `beta=0.9` is the standard textbook default — it retains 90% of the previous velocity, giving enough inertia to smooth out noisy gradients and accelerate through flat regions, without so much momentum that it overshoots minima.

* What advantages do Momentum, AdaGrad, and Adam provide compared to plain Gradient Descent

they can adaptivly change the learning rate as function of the gradient, given more accuracy 
and less prone to skip the minimum.