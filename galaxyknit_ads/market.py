"""关键词级“竞价 → 曝光/点击/转化”响应模型，以及基于它的单日竞价模拟器。

模型（每个关键词）:
    展示量   ~ Poisson(Gamma(base_imp · 季节 · 星期 · min((bid / (ref_bid · 竞争)) ^ e, IMP_CAP)))
    点击     ~ Binomial(展示, ctr · 相对竞价^CTR_POS_EXP)      # 竞价越高位置越好
    CPC      = bid · cpc_ratio · 噪声（二价拍卖，不超过 bid）
    订单     ~ Binomial(点击, cvr)
    销售额   = 订单 · aov · 噪声
ctr / cvr 用 Beta 后验（向同匹配类型的全局均值收缩），训练时每个回合从后验中采样，
让策略对估计误差保持稳健。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ID_COLS, KEY_COLS, complete_panel

# 针织服装季节曲线（1–12 月，旺季 10–12 月），在数据不足 3 个月时作为先验
KNITWEAR_SEASON = np.array([0.95, 0.75, 0.60, 0.50, 0.40, 0.35, 0.40, 0.55, 0.85, 1.35, 1.85, 2.00])
KNITWEAR_SEASON = KNITWEAR_SEASON / KNITWEAR_SEASON.mean()

COMP_EXP = 0.3  # 旺季竞争加剧：同样竞价拿到的曝光份额变少
IMP_CAP = 3.0  # 加价带来的曝光增长上限（相对参考竞价）
CTR_POS_EXP = 0.10
IMP_DISP = 4.0  # 展示量的过度离散（Gamma 形状参数）
PRIOR_ELASTICITY = 1.0
DEFAULT_CPC_RATIO = 0.80
MATCH_GROUPS = ("EXACT", "PHRASE", "BROAD", "OTHER")
CTR_PRIOR_STRENGTH = 500.0  # 等价展示数
CVR_PRIOR_STRENGTH = 30.0  # 等价点击数
AOV_PRIOR_ORDERS = 3.0


def match_group(mt):
    mt = str(mt).upper()
    return mt if mt in MATCH_GROUPS else "OTHER"


def month_dow(dates):
    """datetime64[D] 数组 → (月份 0-11, 星期 0=周一)。"""
    dates = np.asarray(dates, dtype="datetime64[D]")
    month = dates.astype("datetime64[M]").astype(np.int64) % 12
    dow = (dates.astype(np.int64) + 3) % 7
    return month, dow


@dataclass
class Market:
    params: pd.DataFrame  # 每行一个关键词
    season: np.ndarray = field(default_factory=lambda: KNITWEAR_SEASON.copy())
    dow: np.ndarray = field(default_factory=lambda: np.ones(7))
    globals: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.params)

    def arrays(self, idx=None):
        p = self.params if idx is None else self.params.iloc[idx]
        cols = ["base_imp", "ref_bid", "elasticity", "ctr_a", "ctr_b", "cvr_a", "cvr_b", "aov", "cpc_ratio", "scale"]
        out = {c: p[c].to_numpy(dtype=float) for c in cols}
        out["ctr"] = out["ctr_a"] / (out["ctr_a"] + out["ctr_b"])
        out["cvr"] = out["cvr_a"] / (out["cvr_a"] + out["cvr_b"])
        out["match_idx"] = p["match_group"].map(MATCH_GROUPS.index).to_numpy()
        return out

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.params.to_csv(folder / "market_params.csv", index=False)
        meta = {"season": self.season.tolist(), "dow": self.dow.tolist(), "globals": self.globals}
        (folder / "market.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        params = pd.read_csv(folder / "market_params.csv", dtype={c: str for c in ID_COLS})
        params[KEY_COLS] = params[KEY_COLS].fillna("")
        meta = json.loads((folder / "market.json").read_text())
        return cls(params, np.array(meta["season"]), np.array(meta["dow"]), meta["globals"])


def expected_lift(bid, ref_bid, elasticity, season_m):
    rel = bid / (ref_bid * season_m**COMP_EXP)
    return np.minimum(rel**elasticity, IMP_CAP), rel


def simulate_day(rng, bid, p, month, dow, season, dow_idx):
    """模拟一天的广告表现。p 为参数数组字典（ctr/cvr 为已采样的真实值）。"""
    n = len(bid)
    s = season[month]
    lift, rel = expected_lift(bid, p["ref_bid"], p["elasticity"], s)
    mu = p["base_imp"] * s * dow_idx[dow] * lift
    imp = rng.poisson(rng.gamma(IMP_DISP, mu / IMP_DISP))
    ctr = np.clip(p["ctr"] * np.clip(rel, 0.2, 5.0) ** CTR_POS_EXP, 0.0, 0.5)
    clicks = rng.binomial(imp, ctr)
    cpc = np.minimum(bid, bid * p["cpc_ratio"] * np.exp(rng.normal(0, 0.08, n)))
    spend = clicks * cpc
    orders = rng.binomial(clicks, np.clip(p["cvr"], 0, 1))
    sales = orders * p["aov"] * np.exp(rng.normal(0, 0.10, n))
    return {"imp": imp.astype(float), "clicks": clicks.astype(float), "spend": spend,
            "orders": orders.astype(float), "sales": sales}


def expected_reward(bid, p, month, dow, season, dow_idx, max_acos):
    """给定竞价的期望单日奖励（用于 oracle 上界）。bid 可为 (n, k) 网格。"""
    def col(x):
        return x[:, None] if np.ndim(bid) == 2 else x
    s = season[month]
    lift, rel = expected_lift(bid, col(p["ref_bid"]), col(p["elasticity"]), col(s))
    mu = col(p["base_imp"] * s * dow_idx[dow]) * lift
    clicks = mu * np.clip(col(p["ctr"]) * np.clip(rel, 0.2, 5.0) ** CTR_POS_EXP, 0, 0.5)
    margin = col(p["cvr"] * p["aov"]) - bid * col(p["cpc_ratio"]) / max_acos
    return clicks * margin


def _season_index(panel):
    monthly = panel.groupby(panel["date"].dt.month - 1)["impressions"].sum()
    days = panel.groupby(panel["date"].dt.month - 1)["date"].nunique()
    season = KNITWEAR_SEASON.copy()
    if len(monthly) >= 3:
        per_day = (monthly / days).to_numpy()
        months = monthly.index.to_numpy()
        # 让观测月份的均值与先验在这些月份上的均值一致，未观测月份沿用先验
        season[months] = per_day / per_day.mean() * KNITWEAR_SEASON[months].mean()
        season = np.clip(season, 0.1, 5.0)
        season = season / season.mean()
    return season


def _dow_index(panel):
    if panel["date"].nunique() < 28:
        return np.ones(7)
    by = panel.groupby(panel["date"].dt.dayofweek)["impressions"].sum()
    weeks = panel.groupby(panel["date"].dt.dayofweek)["date"].nunique()
    idx = (by / weeks).reindex(range(7)).fillna(by.mean()).to_numpy()
    idx = idx / idx.mean()
    return 0.5 * idx + 0.5  # 向 1 收缩


def fit_market(df, min_impressions=30, default_bid=None):
    """从标准化日面板拟合 Market。"""
    panel = complete_panel(df)
    season = _season_index(panel)
    dow_idx = _dow_index(panel)
    has_bid = "bid" in panel.columns and panel["bid"].notna().any()

    mg = panel["match_type"].map(match_group)
    tot = panel.assign(mg=mg).groupby("mg")[["impressions", "clicks", "orders", "sales"]].sum()
    all_tot = tot.sum()
    ctr_g = float(all_tot["clicks"] / max(all_tot["impressions"], 1)) or 0.003
    cvr_g = float(all_tot["orders"] / max(all_tot["clicks"], 1)) or 0.08
    aov_g = float(all_tot["sales"] / max(all_tot["orders"], 1)) or 30.0
    ctr_by = {m: (r["clicks"] + ctr_g * 2000) / (r["impressions"] + 2000) for m, r in tot.iterrows()}
    cvr_by = {m: (r["orders"] + cvr_g * 100) / (r["clicks"] + 100) for m, r in tot.iterrows()}
    cpc_all = panel["spend"].sum() / max(panel["clicks"].sum(), 1)
    if default_bid is None:
        default_bid = cpc_all / DEFAULT_CPC_RATIO if cpc_all > 0 else 1.0

    rows = []
    for key, g in panel.groupby("key", sort=False):
        if g["impressions"].sum() < min_impressions:
            continue
        g = g.sort_values("date")
        group = match_group(g["match_type"].iloc[0])
        cpc = (g["spend"] / g["clicks"].where(g["clicks"] > 0)).ffill().bfill()
        if has_bid and g["bid"].notna().any():
            bid = g["bid"].ffill().bfill()
            ratio = (cpc / bid).where(g["clicks"] > 0).dropna()
            cpc_ratio = float(np.clip(ratio.median(), 0.4, 1.0)) if len(ratio) >= 3 else DEFAULT_CPC_RATIO
        else:
            bid = cpc / DEFAULT_CPC_RATIO if cpc.notna().any() else pd.Series(default_bid, index=g.index)
            cpc_ratio = DEFAULT_CPC_RATIO
        bid = bid.fillna(default_bid).to_numpy(dtype=float)
        ref_bid = float(np.median(bid[-14:]))

        month, dow = month_dow(g["date"].to_numpy().astype("datetime64[D]"))
        s = season[month]
        imp = g["impressions"].to_numpy(dtype=float)
        x = np.log(bid / s**COMP_EXP)
        y = np.log(imp + 0.5) - np.log(s * dow_idx[dow])
        ok = imp > 0
        elasticity = PRIOR_ELASTICITY
        if ok.sum() >= 10 and x[ok].std() > 0.05:
            slope = np.polyfit(x[ok], y[ok], 1)[0]
            n = ok.sum()
            elasticity = (n * slope + 20 * PRIOR_ELASTICITY) / (n + 20)
        elasticity = float(np.clip(elasticity, 0.3, 2.5))
        lift, _ = expected_lift(bid, ref_bid, elasticity, s)
        base_imp = float(np.mean(imp / (s * dow_idx[dow] * lift)))

        clicks, orders, sales = g["clicks"].sum(), g["orders"].sum(), g["sales"].sum()
        ctr0, cvr0 = ctr_by.get(group, ctr_g), cvr_by.get(group, cvr_g)
        ctr_a = ctr0 * CTR_PRIOR_STRENGTH + clicks
        ctr_b = (1 - ctr0) * CTR_PRIOR_STRENGTH + imp.sum() - clicks
        cvr_a = cvr0 * CVR_PRIOR_STRENGTH + orders
        cvr_b = (1 - cvr0) * CVR_PRIOR_STRENGTH + clicks - orders
        aov = (sales + aov_g * AOV_PRIOR_ORDERS) / (orders + AOV_PRIOR_ORDERS)
        exp_orders = base_imp * ctr_a / (ctr_a + ctr_b) * cvr_a / (cvr_a + cvr_b)
        last = g.iloc[-1]
        rows.append({
            "key": key, **{c: last[c] for c in KEY_COLS}, **{c: last[c] for c in ID_COLS},
            "match_group": group, "base_imp": base_imp, "ref_bid": ref_bid, "elasticity": elasticity,
            "ctr_a": ctr_a, "ctr_b": ctr_b, "cvr_a": cvr_a, "cvr_b": cvr_b, "aov": aov,
            "cpc_ratio": cpc_ratio, "scale": aov * max(1.0, exp_orders),
            "n_days": len(g), "impressions": imp.sum(), "clicks": clicks, "orders": orders,
            "spend": g["spend"].sum(), "sales": sales,
        })
    if not rows:
        raise ValueError("没有足够数据的关键词可供拟合（每个关键词至少需要 %d 次展示）" % min_impressions)
    params = pd.DataFrame(rows)
    glob = {"ctr": ctr_g, "cvr": cvr_g, "aov": aov_g, "bid_estimated": not has_bid,
            "last_date": str(panel["date"].max().date())}
    return Market(params, season, dow_idx, glob)
