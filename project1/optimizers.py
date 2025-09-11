import numpy as np

class Optimizer:
    def __init__(self, lr=1e-2):
        self.lr = float(lr)
        self.iteration = 0

    def step(self, w, prob):
        raise NotImplementedError

    def run(self, w, prob, max_iters=100, log_every=10):
        history = []
        for t in range(max_iters):
            self.iteration = t
            self.step(w, prob)
            if (t % log_every) == 0:
                history.append(prob.loss(w))
        return history


class GDOptimizer(Optimizer):
    def __init__(self, lr=1e-2, weight_decay=0.0):
        super().__init__(lr)
        self.weight_decay = float(weight_decay)

    def step(self, w, prob):
        g = prob.gradient(w)
        if self.weight_decay != 0.0:
            g = g + self.weight_decay * w
        w -= self.lr * g


class SGDOptimizer(Optimizer):
    def __init__(self, lr=1e-2, momentum=0.9, weight_decay=0.0, seed=42):
        super().__init__(lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.rng = np.random.default_rng(seed)
        self.velocity = None  # lazy init
        self.idx = None
        self.ptr = 0

    def _ensure_state(self, w, prob):
        if self.velocity is None:
            self.velocity = np.zeros_like(w, dtype=np.float32)
        if self.idx is None or self.ptr >= prob.n:
            self.idx = self.rng.permutation(prob.n)  # shuffle once per epoch
            self.ptr = 0

    def step(self, w, prob):
        self._ensure_state(w, prob)
        i = int(self.idx[self.ptr])
        self.ptr += 1

        g = prob.stochastic_gradient(w, i)
        if self.weight_decay != 0.0:
            g = g + self.weight_decay * w
        # momentum update
        self.velocity = self.momentum * self.velocity + (1.0 - self.momentum) * g
        w -= self.lr * self.velocity


class MiniBatchOptimizer(Optimizer):
    def __init__(self, lr=1e-2, batch_size=64, momentum=0.9, weight_decay=0.0,
                 lr_decay=0.0, seed=42):
        super().__init__(lr)
        self.batch_size = int(batch_size)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.lr_decay = float(lr_decay)  # per-epoch decay factor (e.g., 0.1 for 10%)
        self.rng = np.random.default_rng(seed)
        self.velocity = None
        self.idx = None
        self.ptr = 0
        self.epoch = 0

    def _ensure_state(self, w, prob):
        if self.velocity is None:
            self.velocity = np.zeros_like(w, dtype=np.float32)
        if self.idx is None or self.ptr >= prob.n:
            # step LR at epoch boundary
            if self.lr_decay > 0.0 and self.epoch > 0:
                self.lr *= (1.0 - self.lr_decay)
            self.idx = self.rng.permutation(prob.n)
            self.ptr = 0
            self.epoch += 1

    def step(self, w, prob):
        self._ensure_state(w, prob)
        start = self.ptr
        end = min(start + self.batch_size, prob.n)
        batch_idx = self.idx[start:end]
        self.ptr = end

        g = prob.minibatch_gradient(w, batch_idx)
        if self.weight_decay != 0.0:
            g = g + self.weight_decay * w

        # momentum update
        self.velocity = self.momentum * self.velocity + (1.0 - self.momentum) * g
        w -= self.lr * self.velocity
