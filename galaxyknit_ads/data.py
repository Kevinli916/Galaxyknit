"""读取并标准化亚马逊 SP 广告报表（卖家后台下载的 CSV/Excel，或 Ads API 的 JSON）。

输出统一的“关键词 × 日”面板，列为 CANONICAL 中的字段。
"""

import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

KEY_COLS = ["campaign", "ad_group", "targeting", "match_type"]
METRIC_COLS = ["impressions", "clicks", "spend", "orders", "sales"]
ID_COLS = ["campaign_id", "ad_group_id", "keyword_id", "target_id"]

# 标准字段 → 可能的报表列名（英文后台、中文后台、Ads API v3）
ALIASES = {
    "date": ["date", "日期", "start date", "开始日期"],
    "campaign": ["campaign name", "campaignname", "广告活动名称", "广告活动"],
    "ad_group": ["ad group name", "adgroupname", "广告组名称", "广告组"],
    "targeting": ["targeting", "keyword", "投放", "关键词", "投放目标"],
    "match_type": ["match type", "matchtype", "keywordtype", "匹配类型"],
    "impressions": ["impressions", "展示量", "曝光量", "展现量"],
    "clicks": ["clicks", "点击量", "点击次数"],
    "spend": ["spend", "cost", "花费", "支出"],
    "sales": ["7 day total sales", "sales7d", "sales", "7天总销售额", "7天总销售额(￥)", "销售额"],
    "orders": ["7 day total orders (#)", "purchases7d", "orders", "7天总订单数(#)", "7天总订单数", "订单数", "订单量"],
    "bid": ["bid", "keywordbid", "max bid", "竞价", "出价"],
    "campaign_id": ["campaign id", "campaignid", "广告活动编号", "广告活动id"],
    "ad_group_id": ["ad group id", "adgroupid", "广告组编号", "广告组id"],
    "keyword_id": ["keyword id", "keywordid", "关键词编号", "关键词id"],
    "target_id": ["product targeting id", "targetid", "商品投放编号"],
}


def _norm(name):
    s = str(name).strip().lower().replace("（", "(").replace("）", ")")
    return re.sub(r"[\s_\-()#$￥:：]", "", s)


_LOOKUP = {_norm(a): canon for canon, names in ALIASES.items() for a in names}


def _to_number(s):
    if s.dtype.kind in "iuf":
        return s.astype(float)
    cleaned = s.astype(str).str.replace(r"[^\d.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _read_any(path):
    path = Path(path)
    name = path.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    if name.endswith((".json", ".json.gz")):
        opener = gzip.open if name.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            return pd.DataFrame(json.load(f))
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法识别文件编码: {path}")


def normalize(raw):
    """把任意来源的报表列名和数值统一为标准格式。"""
    rename = {}
    for col in raw.columns:
        canon = _LOOKUP.get(_norm(col))
        if canon and canon not in rename.values():
            rename[col] = canon
    df = raw.rename(columns=rename)[list(rename.values())].copy()

    missing = {"date", "targeting", "impressions", "clicks", "spend", "sales", "orders"} - set(df.columns)
    if missing:
        raise ValueError(f"报表缺少必要列: {sorted(missing)}；现有列: {list(raw.columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])
    for c in METRIC_COLS + (["bid"] if "bid" in df.columns else []):
        df[c] = _to_number(df[c]).fillna(0.0)
    for c in ("campaign", "ad_group", "match_type"):
        if c not in df.columns:
            df[c] = ""
    for c in KEY_COLS:
        df[c] = df[c].fillna("").astype(str).str.strip()
    df["match_type"] = df["match_type"].str.upper()

    # Ads API 中关键词与商品投放共用 keywordId，按匹配类型拆分
    if "keyword_id" in df.columns and "target_id" not in df.columns:
        is_kw = df["match_type"].isin(["EXACT", "PHRASE", "BROAD"])
        df["target_id"] = df["keyword_id"].where(~is_kw)
        df["keyword_id"] = df["keyword_id"].where(is_kw)
    for c in ID_COLS:
        if c not in df.columns:
            df[c] = None
        df[c] = df[c].map(lambda v: None if pd.isna(v) or str(v).strip() in ("", "nan") else str(v).split(".")[0])
    return df


def aggregate_daily(df):
    """同一关键词同一天的多行（如不同广告位）合并为一行。"""
    agg = {c: "sum" for c in METRIC_COLS}
    agg.update({c: "first" for c in ID_COLS})
    if "bid" in df.columns:
        agg["bid"] = "max"
    out = df.groupby(KEY_COLS + ["date"], as_index=False, dropna=False).agg(agg)
    out["key"] = out[KEY_COLS].agg(" | ".join, axis=1)
    return out.sort_values(["key", "date"]).reset_index(drop=True)


def load_reports(paths):
    """读取一个或多个报表文件并返回标准化的日面板。"""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames = [normalize(_read_any(p)) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    if "bid" in df.columns:
        df.loc[df["bid"] <= 0, "bid"] = np.nan
    return aggregate_daily(df)


def complete_panel(df):
    """补齐每个关键词在其首末日期之间缺失的日期（报表通常不输出 0 展示的行）。"""
    rows = []
    for key, g in df.groupby("key", sort=False):
        idx = pd.date_range(g["date"].min(), df["date"].max(), freq="D")
        g = g.set_index("date").reindex(idx)
        g.index.name = "date"
        for c in METRIC_COLS:
            g[c] = g[c].fillna(0.0)
        fill_cols = KEY_COLS + ID_COLS + ["key"] + (["bid"] if "bid" in g.columns else [])
        g[fill_cols] = g[fill_cols].ffill().bfill()
        rows.append(g.reset_index())
    return pd.concat(rows, ignore_index=True)
