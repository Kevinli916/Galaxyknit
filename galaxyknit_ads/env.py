"""向量化竞价环境：同时模拟 n 个关键词，每步 = 一次调价决策。

观测只包含真实投放中也能拿到的信息（近 7/28 天表现、后验估计、季节、星期），
与上线时 policy.py 构造观测的方式完全一致。
"""

import numpy as np

from .config import ACTIONS, Strategy
from .market import MATCH_GROUPS, Market, expected_reward, month_dow, simulate_day

HIST = 28
METRICS = ("imp", "clicks", "spend", "orders", "sales")
BASE_DATE = np.datetime64("2025-01-01")


def build_obs(bid, p, hist, month, dow, season, g, strategy):
    """构造观测向量。p 中 ctr/cvr 使用后验均值（agent 不知道真实值）。"""
    s7 = {k: hist[k][:, -7:].sum(1) for k in METRICS}
    s28 = {k: hist[k].sum(1) for k in METRICS}
    imp3 = hist["imp"][:, -3:].sum(1)
    ctr_g, cvr_g, aov_g = g["ctr"], g["cvr"], g["aov"]
    ctr_post = p["ctr_a"] / (p["ctr_a"] + p["ctr_b"])
    cvr_post = p["cvr_a"] / (p["cvr_a"] + p["cvr_b"])
    cvr28 = (s28["orders"] + cvr_post * 20) / (s28["clicks"] + 20)
    breakeven_cpc = cvr28 * p["aov"] * strategy.max_acos
    cpc7 = np.where(s7["clicks"] > 0, s7["spend"] / np.maximum(s7["clicks"], 1) / bid, p["cpc_ratio"])
    acos7 = s7["spend"] / (s7["sales"] + aov_g * 0.2) / strategy.max_acos
    acos28 = s28["spend"] / (s28["sales"] + aov_g * 0.2) / strategy.max_acos
    nxt = season[(month + 1) % 12]
    onehot = np.eye(len(MATCH_GROUPS))[p["match_idx"]]
    feats = [
        np.log(bid / p["ref_bid"]),
        np.log(bid * p["cpc_ratio"] / breakeven_cpc),
        np.log1p(s7["imp"]) / 5,
        np.log1p(s7["clicks"]) / 3,
        np.log1p(s7["orders"]) / 2,
        np.log1p(s28["clicks"]) / 3,
        np.log1p(s28["orders"]) / 2,
        np.log((s7["clicks"] + ctr_post * 200) / (s7["imp"] + 200) / ctr_g),
        np.log((s7["orders"] + cvr_post * 10) / (s7["clicks"] + 10) / cvr_g),
        np.log(cvr28 / cvr_g),
        np.log(ctr_post / ctr_g),
        np.log(cvr_post / cvr_g),
        np.log1p(p["cvr_a"] + p["cvr_b"]) / 5,
        np.clip(acos7, 0, 5),
        np.clip(acos28, 0, 5),
        np.clip(cpc7, 0, 1.5),
        np.log((imp3 / 3 + 1) / (s7["imp"] / 7 + 1)),
        np.log(p["aov"] / aov_g),
        np.log(season[month]),
        np.log(nxt / season[month]),
        np.sin(2 * np.pi * dow / 7),
        np.cos(2 * np.pi * dow / 7),
    ]
    return np.column_stack(feats + [onehot]).astype(np.float64)


OBS_DIM = 22 + len(MATCH_GROUPS)


class BiddingEnv:
    def __init__(self, market: Market, strategy: Strategy, n_envs=256, horizon_days=28, seed=0,
                 randomize=True, init_bid_noise=0.35, start_dates=None):
        self.m = market
        self.st = strategy
        self.n = n_envs
        self.k = strategy.decision_interval
        self.horizon = max(1, horizon_days // self.k)
        self.rng = np.random.default_rng(seed)
        self.randomize = randomize
        self.init_bid_noise = init_bid_noise
        self.start_dates = start_dates  # None = 全年随机
        self.season, self.dow_idx = market.season, market.dow
        w = np.sqrt(market.params["spend"].to_numpy() + market.params["scale"].to_numpy())
        self.weights = w / w.sum()

    # ---------- 回合控制 ----------
    def reset(self, idx=None):
        rng = self.rng
        self.idx = rng.choice(len(self.m), self.n, p=self.weights) if idx is None else np.asarray(idx)
        p = self.m.arrays(self.idx)
        self.p_obs = p  # 观测用后验均值
        true = dict(p)
        if self.randomize:
            true["ctr"] = rng.beta(p["ctr_a"], p["ctr_b"])
            true["cvr"] = rng.beta(p["cvr_a"], p["cvr_b"])
            true["elasticity"] = np.clip(p["elasticity"] * np.exp(rng.normal(0, 0.25, self.n)), 0.2, 3.0)
            true["base_imp"] = p["base_imp"] * np.exp(rng.normal(0, 0.2, self.n))
        self.p_true = true
        if self.start_dates is None:
            self.date = BASE_DATE + rng.integers(0, 365, self.n).astype("timedelta64[D]")
        else:
            self.date = np.asarray(self.start_dates, dtype="datetime64[D]").copy()
        noise = np.exp(rng.normal(0, self.init_bid_noise, self.n)) if self.init_bid_noise else 1.0
        self.bid = self._clip_bid(p["ref_bid"] * noise)
        self.hist = {k: np.zeros((self.n, HIST)) for k in METRICS}
        self.date = self.date - np.timedelta64(HIST, "D")
        for _ in range(HIST):  # 用初始竞价预热 28 天历史
            self._sim_day()
        self.t = 0
        return self.obs()

    def _clip_bid(self, bid):
        ref = self.p_obs["ref_bid"]
        bid = np.clip(bid, ref * self.st.rel_bid_floor, ref * self.st.rel_bid_cap)
        return np.round(np.clip(bid, self.st.min_bid, self.st.max_bid), 2)

    def _sim_day(self):
        month, dow = month_dow(self.date)
        out = simulate_day(self.rng, self.bid, self.p_true, month, dow, self.season, self.dow_idx)
        for k in METRICS:
            self.hist[k] = np.concatenate([self.hist[k][:, 1:], out[k][:, None]], axis=1)
        self.date = self.date + np.timedelta64(1, "D")
        return out

    def obs(self):
        month, dow = month_dow(self.date)
        return build_obs(self.bid, self.p_obs, self.hist, month, dow, self.season, self.m.globals, self.st)

    # ---------- 交互 ----------
    def step(self, actions):
        mult = np.asarray(ACTIONS)[actions]
        return self.step_bids(self.bid * mult)

    def step_bids(self, new_bids):
        new_bids = self._clip_bid(new_bids)
        changed = np.abs(np.log(new_bids / self.bid))
        self.bid = new_bids
        info = {k: np.zeros(self.n) for k in METRICS}
        for _ in range(self.k):
            out = self._sim_day()
            for k in METRICS:
                info[k] += out[k]
        net = info["sales"] - info["spend"] / self.st.max_acos
        reward = net / (self.p_obs["scale"] * self.k) - self.st.change_penalty * (changed > 1e-9)
        self.t += 1
        return self.obs(), reward, self.t >= self.horizon, info

    def oracle_bids(self, grid=np.exp(np.linspace(np.log(0.3), np.log(3.0), 41))):
        """已知真实参数时的最优竞价（评估上界，实际不可得）。"""
        month, dow = month_dow(self.date)
        bids = self.p_obs["ref_bid"][:, None] * grid[None, :]
        er = expected_reward(bids, self.p_true, month, dow, self.season, self.dow_idx, self.st.max_acos)
        return bids[np.arange(self.n), er.argmax(1)]
