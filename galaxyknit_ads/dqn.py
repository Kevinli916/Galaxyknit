"""纯 numpy 实现的 Double DQN（MLP + Adam + Huber loss + 软更新目标网络），无需深度学习框架。"""

import json
from pathlib import Path

import numpy as np


class MLP:
    def __init__(self, sizes, rng):
        self.W = [rng.normal(0, np.sqrt(2.0 / i), (i, o)) for i, o in zip(sizes[:-1], sizes[1:])]
        self.W[-1] *= 0.1
        self.b = [np.zeros(o) for o in sizes[1:]]

    @property
    def params(self):
        return self.W + self.b

    def forward(self, x):
        acts = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = np.maximum(x, 0.0)
            acts.append(x)
        return x, acts

    def backward(self, acts, g):
        n = len(self.W)
        gW, gb = [None] * n, [None] * n
        for i in reversed(range(n)):
            if i < n - 1:
                g = g * (acts[i + 1] > 0)
            gW[i] = acts[i].T @ g
            gb[i] = g.sum(0)
            if i > 0:
                g = g @ self.W[i].T
        return gW + gb

    def copy_from(self, other, tau=1.0):
        for mine, theirs in zip(self.params, other.params):
            mine *= 1 - tau
            mine += tau * theirs


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, clip=10.0):
        self.params, self.lr, self.b1, self.b2, self.eps, self.clip = params, lr, b1, b2, eps, clip
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        norm = np.sqrt(sum((g**2).sum() for g in grads))
        if norm > self.clip:
            grads = [g * (self.clip / norm) for g in grads]
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1**self.t)
            vh = v / (1 - self.b2**self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


class ReplayBuffer:
    def __init__(self, obs_dim, capacity, rng):
        self.s = np.zeros((capacity, obs_dim))
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity)
        self.s2 = np.zeros((capacity, obs_dim))
        self.cap, self.ptr, self.size, self.rng = capacity, 0, 0, rng

    def add_batch(self, s, a, r, s2):
        n = len(a)
        idx = (self.ptr + np.arange(n)) % self.cap
        self.s[idx], self.a[idx], self.r[idx], self.s2[idx] = s, a, r, s2
        self.ptr = (self.ptr + n) % self.cap
        self.size = min(self.size + n, self.cap)

    def sample(self, batch):
        i = self.rng.integers(0, self.size, batch)
        return self.s[i], self.a[i], self.r[i], self.s2[i]


class DQNAgent:
    def __init__(self, obs_dim, n_actions, hidden=(128, 128), lr=1e-3, gamma=0.9, tau=0.01, seed=0):
        self.rng = np.random.default_rng(seed)
        sizes = [obs_dim, *hidden, n_actions]
        self.config = {"obs_dim": obs_dim, "n_actions": n_actions, "hidden": list(hidden),
                       "lr": lr, "gamma": gamma, "tau": tau}
        self.online = MLP(sizes, self.rng)
        self.target = MLP(sizes, self.rng)
        self.target.copy_from(self.online)
        self.opt = Adam(self.online.params, lr=lr)
        self.gamma, self.tau = gamma, tau
        self.obs_mean = np.zeros(obs_dim)
        self.obs_std = np.ones(obs_dim)

    def set_normalizer(self, samples):
        self.obs_mean = samples.mean(0)
        self.obs_std = samples.std(0) + 1e-6

    def _norm(self, s):
        return (s - self.obs_mean) / self.obs_std

    def q_values(self, s):
        return self.online.forward(self._norm(s))[0]

    def act(self, s, eps=0.0):
        a = self.q_values(s).argmax(1)
        if eps > 0:
            explore = self.rng.random(len(a)) < eps
            a[explore] = self.rng.integers(0, self.config["n_actions"], explore.sum())
        return a

    def update(self, batch):
        s, a, r, s2 = batch
        s, s2 = self._norm(s), self._norm(s2)
        q, acts = self.online.forward(s)
        a2 = self.online.forward(s2)[0].argmax(1)
        q2 = self.target.forward(s2)[0][np.arange(len(a)), a2]
        y = r + self.gamma * q2
        td = q[np.arange(len(a)), a] - y
        g = np.zeros_like(q)
        g[np.arange(len(a)), a] = np.clip(td, -1.0, 1.0) / len(a)  # Huber 梯度
        self.opt.step(self.online.backward(acts, g))
        self.target.copy_from(self.online, self.tau)
        return float(np.mean(np.where(np.abs(td) < 1, 0.5 * td**2, np.abs(td) - 0.5)))

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        arrays = {f"W{i}": w for i, w in enumerate(self.online.W)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.online.b)})
        np.savez(folder / "policy.npz", obs_mean=self.obs_mean, obs_std=self.obs_std, **arrays)
        (folder / "policy.json").write_text(json.dumps(self.config, indent=2))

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        cfg = json.loads((folder / "policy.json").read_text())
        agent = cls(cfg["obs_dim"], cfg["n_actions"], tuple(cfg["hidden"]), cfg["lr"], cfg["gamma"], cfg["tau"])
        z = np.load(folder / "policy.npz")
        for i in range(len(agent.online.W)):
            agent.online.W[i][...] = z[f"W{i}"]
            agent.online.b[i][...] = z[f"b{i}"]
        agent.target.copy_from(agent.online)
        agent.obs_mean, agent.obs_std = z["obs_mean"], z["obs_std"]
        return agent
