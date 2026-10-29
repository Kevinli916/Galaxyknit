"""生成模拟的针织服装 SP 广告账户报表（卖家后台英文表头格式），用于演示和测试。"""

import numpy as np
import pandas as pd

from .market import KNITWEAR_SEASON, MATCH_GROUPS, month_dow, simulate_day

TERMS = [
    "womens cardigan sweater", "chunky knit sweater", "cable knit sweater women", "oversized sweater women",
    "turtleneck sweater women", "mens wool sweater", "cashmere sweater women", "knit vest women",
    "crewneck sweater men", "mock neck sweater", "v neck sweater women", "fisherman sweater",
    "cropped cardigan", "long cardigan women", "sweater dress", "knit pullover", "merino wool sweater men",
    "christmas sweater", "half zip sweater men", "knitted beanie", "fall sweaters for women",
    "lightweight sweater", "mohair sweater", "striped sweater women",
]
CAMPAIGNS = ["SP-Cardigan", "SP-Pullover", "SP-Mens", "SP-Seasonal"]


def make_truth(n_keywords=120, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_keywords):
        mt = rng.choice(MATCH_GROUPS, p=[0.4, 0.2, 0.25, 0.15])
        term = TERMS[i % len(TERMS)] + ("" if i < len(TERMS) else f" {i // len(TERMS)}")
        targeting = term if mt != "OTHER" else f'asin="B0{rng.integers(10**7, 10**8)}"'
        ctr = {"EXACT": 0.006, "PHRASE": 0.004, "BROAD": 0.003, "OTHER": 0.002}[mt] * np.exp(rng.normal(0, 0.4))
        cvr = {"EXACT": 0.12, "PHRASE": 0.09, "BROAD": 0.07, "OTHER": 0.06}[mt] * np.exp(rng.normal(0, 0.45))
        rows.append({
            "campaign": CAMPAIGNS[i % len(CAMPAIGNS)], "ad_group": f"AG-{i % 12:02d}", "targeting": targeting,
            "match_type": mt if mt != "OTHER" else "-", "base_imp": float(np.exp(rng.normal(np.log(1500), 1.0))),
            "ref_bid": float(np.round(np.exp(rng.normal(np.log(1.1), 0.35)), 2)),
            "elasticity": float(rng.uniform(0.6, 1.8)), "ctr": float(np.clip(ctr, 5e-4, 0.05)),
            "cvr": float(np.clip(cvr, 0.01, 0.4)), "aov": float(np.exp(rng.normal(np.log(38), 0.25))),
            "cpc_ratio": float(rng.uniform(0.65, 0.92)),
            "campaign_id": str(10**11 + i % len(CAMPAIGNS)), "ad_group_id": str(2 * 10**11 + i % 12),
            "keyword_id": str(3 * 10**11 + i) if mt != "OTHER" else None,
            "target_id": str(4 * 10**11 + i) if mt == "OTHER" else None,
        })
    return pd.DataFrame(rows)


def generate_report(truth, start="2026-05-01", days=180, seed=11, include_bid=True):
    """按“当前竞价 + 每周随机微调”的运营方式生成历史日报表。"""
    rng = np.random.default_rng(seed)
    p = {c: truth[c].to_numpy(dtype=float) for c in
         ["base_imp", "ref_bid", "elasticity", "ctr", "cvr", "aov", "cpc_ratio"]}
    ref = truth["ref_bid"].to_numpy(dtype=float)
    bid = ref.copy()
    dow_idx = np.array([1.0, 1.0, 1.0, 1.0, 0.95, 1.05, 1.1])
    dates = np.datetime64(start) + np.arange(days).astype("timedelta64[D]")
    frames = []
    for d in dates:
        if (d - dates[0]).astype(int) % 7 == 0:  # 每周人工在常用竞价附近微调
            bid = np.round(np.clip(ref * np.exp(rng.normal(0, 0.2, len(bid))), 0.2, 5), 2)
        month, dow = month_dow(np.full(len(bid), d))
        out = simulate_day(rng, bid, p, month, dow, KNITWEAR_SEASON, dow_idx)
        f = pd.DataFrame({
            "Date": pd.Timestamp(d).strftime("%Y-%m-%d"), "Campaign Name": truth["campaign"],
            "Ad Group Name": truth["ad_group"], "Targeting": truth["targeting"], "Match Type": truth["match_type"],
            "Impressions": out["imp"].astype(int), "Clicks": out["clicks"].astype(int),
            "Spend": out["spend"].round(2), "7 Day Total Sales ": out["sales"].round(2),
            "7 Day Total Orders (#)": out["orders"].astype(int),
            "Campaign ID": truth["campaign_id"], "Ad Group ID": truth["ad_group_id"],
            "Keyword ID": truth["keyword_id"], "Product Targeting ID": truth["target_id"],
        })
        if include_bid:
            f["Bid"] = bid
        frames.append(f[f["Impressions"] > 0])
    return pd.concat(frames, ignore_index=True)
