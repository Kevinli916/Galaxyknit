"""训练 DQN 竞价策略，并在模拟器中与基线策略对比。"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ACTIONS, HOLD_ACTION, Strategy
from .dqn import DQNAgent, ReplayBuffer
from .env import OBS_DIM, BiddingEnv
from .market import Market


def train(market: Market, strategy: Strategy, episodes=120, n_envs=256, horizon_days=28,
          updates_per_step=4, batch_size=512, seed=0, log=print):
    env = BiddingEnv(market, strategy, n_envs=n_envs, horizon_days=horizon_days, seed=seed)
    agent = DQNAgent(OBS_DIM, len(ACTIONS), seed=seed)
    rng = np.random.default_rng(seed + 1)

    # 用随机策略采样观测，确定归一化参数
    samples = [env.reset()]
    for _ in range(env.horizon):
        samples.append(env.step(rng.integers(0, len(ACTIONS), n_envs))[0])
    agent.set_normalizer(np.concatenate(samples))

    buf = ReplayBuffer(OBS_DIM, capacity=400_000, rng=rng)
    total_steps = episodes * env.horizon
    warmup = 4 * n_envs
    step, history, t0 = 0, [], time.time()
    for ep in range(episodes):
        s = env.reset()
        ep_r, losses = 0.0, []
        for _ in range(env.horizon):
            eps = max(0.05, 1.0 - step / (0.5 * total_steps))
            a = agent.act(s, eps)
            s2, r, _, _ = env.step(a)
            buf.add_batch(s, a, r, s2)  # 回合截断不是真正的终止，继续自举
            s = s2
            ep_r += r.mean()
            step += 1
            if buf.size >= warmup:
                losses += [agent.update(buf.sample(batch_size)) for _ in range(updates_per_step)]
        history.append({"episode": ep + 1, "reward": ep_r, "eps": eps, "loss": float(np.mean(losses or [0]))})
        if (ep + 1) % 10 == 0 or ep == 0:
            log(f"[train] ep {ep + 1:>4}/{episodes}  avg_reward/keyword={ep_r:7.3f}  eps={eps:.2f}  "
                f"loss={history[-1]['loss']:.4f}  {time.time() - t0:5.0f}s")
    return agent, pd.DataFrame(history)


# ---------------- 评估 ----------------

def rule_policy(strategy: Strategy):
    """常见的人工调价规则：每 3 天按近 7 天 ACoS 调一次。"""
    def act(env, t):
        h = env.hist
        clicks7, orders7 = h["clicks"][:, -7:].sum(1), h["orders"][:, -7:].sum(1)
        spend7, sales7, imp7 = h["spend"][:, -7:].sum(1), h["sales"][:, -7:].sum(1), h["imp"][:, -7:].sum(1)
        acos = np.where(sales7 > 0, spend7 / np.maximum(sales7, 1e-9), np.inf)
        m = np.ones(env.n)
        m[imp7 < 100] = 1.10
        m[(orders7 > 0) & (acos < strategy.target_acos * 0.75)] = 1.10
        m[(orders7 > 0) & (acos > strategy.target_acos * 1.25)] = 0.90
        m[(clicks7 >= 12) & (orders7 == 0)] = 0.85
        if (t * env.k) % 3:
            m[:] = 1.0
        return env.step_bids(env.bid * m)
    return act


def evaluate(market, strategy, agent=None, n_envs=1024, horizon_days=28, seed=123, start_dates=None):
    policies = {
        "保持当前竞价": lambda env, t: env.step(np.full(env.n, HOLD_ACTION)),
        "规则调价": rule_policy(strategy),
    }
    if agent is not None:
        policies["RL 智能体"] = lambda env, t: env.step(agent.act(env.obs()))
    policies["Oracle(已知真实参数)"] = lambda env, t: env.step_bids(env.oracle_bids())

    rows = []
    for name, fn in policies.items():
        # 共同随机数：每个策略面对完全相同的关键词、真实参数和起始状态
        env = BiddingEnv(market, strategy, n_envs=n_envs, horizon_days=horizon_days, seed=seed,
                         init_bid_noise=0.0, start_dates=start_dates)
        env.reset()
        tot = {k: 0.0 for k in ("imp", "clicks", "spend", "orders", "sales")}
        reward = 0.0
        for t in range(env.horizon):
            _, r, _, info = fn(env, t)
            reward += r.mean()
            for k in tot:
                tot[k] += info[k].sum()
        rows.append({"策略": name, "展示": tot["imp"], "点击": tot["clicks"], "订单": tot["orders"],
                     "花费": tot["spend"], "销售额": tot["sales"],
                     "ACoS": tot["spend"] / max(tot["sales"], 1e-9),
                     "净贡献(销售-花费/maxACoS)": tot["sales"] - tot["spend"] / strategy.max_acos,
                     "平均奖励": reward})
    df = pd.DataFrame(rows)
    base = df.iloc[0]
    df["销售额vs保持"] = df["销售额"] / base["销售额"] - 1
    df["订单vs保持"] = df["订单"] / base["订单"] - 1
    return df


def save_run(folder, agent, market, strategy, history, evals):
    folder = Path(folder)
    agent.save(folder)
    market.save(folder)
    (folder / "strategy.json").write_text(json.dumps(strategy.to_dict(), ensure_ascii=False, indent=2))
    history.to_csv(folder / "train_history.csv", index=False)
    for name, df in evals.items():
        df.to_csv(folder / f"eval_{name}.csv", index=False, encoding="utf-8-sig")
