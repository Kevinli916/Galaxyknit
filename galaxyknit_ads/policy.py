"""用训练好的策略对最新报表生成调价建议，并导出批量上传表。"""

import numpy as np
import pandas as pd

from .config import ACTIONS, HOLD_ACTION, Strategy
from .data import ID_COLS, KEY_COLS, complete_panel
from .env import HIST, build_obs
from .market import Market, month_dow

_MAP = {"impressions": "imp", "clicks": "clicks", "spend": "spend", "orders": "orders", "sales": "sales"}


def _history(panel, keys, end):
    """取每个关键词截至 end 的最近 28 天日数据，缺失补 0。"""
    dates = pd.date_range(end - pd.Timedelta(days=HIST - 1), end, freq="D")
    hist = {}
    sub = panel[panel["key"].isin(keys) & panel["date"].isin(dates)]
    for col, k in _MAP.items():
        wide = sub.pivot_table(index="key", columns="date", values=col, aggfunc="sum")
        hist[k] = wide.reindex(index=keys, columns=dates).fillna(0.0).to_numpy()
    return hist


def recommend(df, market: Market, agent, strategy: Strategy, current_market: Market = None):
    """df: 标准化日面板（最新报表）。current_market: 用最新数据拟合的市场模型（提供最新后验）。"""
    m = current_market or market
    params = m.params
    keys = params["key"].tolist()
    p = m.arrays()
    panel = complete_panel(df)
    end = panel["date"].max()
    hist = _history(panel, keys, end)

    last = panel.sort_values("date").groupby("key").tail(1).set_index("key")
    if "bid" in last.columns and last["bid"].notna().any():
        bid = last["bid"].reindex(keys).to_numpy(dtype=float)
        bid = np.where(np.isfinite(bid) & (bid > 0), bid, p["ref_bid"])
    else:
        bid = p["ref_bid"].copy()

    next_day = np.full(len(keys), np.datetime64(end.date()) + np.timedelta64(1, "D"))
    month, dow = month_dow(next_day)
    obs = build_obs(bid, p, hist, month, dow, market.season, market.globals, strategy)
    q = agent.q_values(obs)
    act = q.argmax(1)
    mult = np.asarray(ACTIONS)[act]
    ref = p["ref_bid"]
    new_bid = np.clip(bid * mult, ref * strategy.rel_bid_floor, ref * strategy.rel_bid_cap)
    new_bid = np.round(np.clip(new_bid, strategy.min_bid, strategy.max_bid), 2)

    s28 = {k: v.sum(1) for k, v in hist.items()}
    acos28 = np.where(s28["sales"] > 0, s28["spend"] / np.maximum(s28["sales"], 1e-9), np.nan)
    neg = (s28["clicks"] >= strategy.negative_clicks) & (s28["orders"] == 0)

    out = params[["key", *KEY_COLS, *ID_COLS]].copy()
    out["当前竞价"] = np.round(bid, 2)
    out["建议竞价"] = new_bid
    out["调整幅度"] = np.round(new_bid / bid - 1, 3)
    out["28天展示"] = s28["imp"].astype(int)
    out["28天点击"] = s28["clicks"].astype(int)
    out["28天订单"] = s28["orders"].astype(int)
    out["28天花费"] = np.round(s28["spend"], 2)
    out["28天销售额"] = np.round(s28["sales"], 2)
    out["28天ACoS"] = np.round(acos28, 3)
    out["置信度(Q优势)"] = np.round(q[np.arange(len(q)), act] - q[:, HOLD_ACTION], 4)
    out["标记"] = np.where(neg, f"{strategy.negative_clicks}+点击0单：考虑否定/暂停", "")
    out["动作"] = np.select([out["调整幅度"] > 0.001, out["调整幅度"] < -0.001], ["加价", "降价"], "保持")
    return out.sort_values("28天花费", ascending=False).reset_index(drop=True)


def to_bulk(recs, only_changes=True):
    """导出 SP 批量操作表（Bulk Operations）中更新竞价所需的列。

    需要报表中带有 Campaign ID / Ad Group ID / Keyword ID 或 Product Targeting ID。
    """
    r = recs[recs["动作"] != "保持"] if only_changes else recs
    r = r[r["keyword_id"].notna() | r["target_id"].notna()]
    is_kw = r["keyword_id"].notna()
    return pd.DataFrame({
        "Product": "Sponsored Products",
        "Entity": np.where(is_kw, "Keyword", "Product Targeting"),
        "Operation": "Update",
        "Campaign ID": r["campaign_id"].to_numpy(),
        "Ad Group ID": r["ad_group_id"].to_numpy(),
        "Keyword ID": r["keyword_id"].to_numpy(),
        "Product Targeting ID": r["target_id"].to_numpy(),
        "Bid": r["建议竞价"].to_numpy(),
    })


def summarize(recs):
    return {
        "关键词数": len(recs),
        "加价": int((recs["动作"] == "加价").sum()),
        "降价": int((recs["动作"] == "降价").sum()),
        "保持": int((recs["动作"] == "保持").sum()),
        "标记否定/暂停": int((recs["标记"] != "").sum()),
    }

