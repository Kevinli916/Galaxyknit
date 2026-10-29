"""Amazon Ads API v3 客户端：拉取 SP 投放日报表、批量更新关键词/商品投放竞价。

凭证从环境变量读取:
    AMAZON_ADS_CLIENT_ID, AMAZON_ADS_CLIENT_SECRET, AMAZON_ADS_REFRESH_TOKEN,
    AMAZON_ADS_PROFILE_ID, AMAZON_ADS_REGION (NA | EU | FE，默认 NA)
"""

import gzip
import json
import os
import time

import pandas as pd
import requests

REGIONS = {
    "NA": ("https://advertising-api.amazon.com", "https://api.amazon.com/auth/o2/token"),
    "EU": ("https://advertising-api-eu.amazon.com", "https://api.amazon.co.uk/auth/o2/token"),
    "FE": ("https://advertising-api-fe.amazon.com", "https://api.amazon.co.jp/auth/o2/token"),
}

SP_TARGETING_COLUMNS = [
    "date", "campaignId", "campaignName", "adGroupId", "adGroupName", "keywordId", "keyword", "targeting",
    "matchType", "keywordType", "keywordBid", "impressions", "clicks", "cost", "purchases7d", "sales7d",
]


def plan_bid_updates(recs):
    """把 policy.recommend 的输出转成 Ads API 请求体（只包含需要调价的投放）。"""
    changes = recs[recs["动作"] != "保持"]
    kw = [{"keywordId": k, "bid": float(b)} for k, b in
          zip(changes["keyword_id"], changes["建议竞价"]) if isinstance(k, str) and k]
    tg = [{"targetId": t, "bid": float(b)} for t, b in
          zip(changes["target_id"], changes["建议竞价"]) if isinstance(t, str) and t]
    return {"keywords": kw, "targetingClauses": tg}


class AdsClient:
    def __init__(self, client_id=None, client_secret=None, refresh_token=None, profile_id=None, region=None):
        env = os.environ.get
        self.client_id = client_id or env("AMAZON_ADS_CLIENT_ID")
        self.client_secret = client_secret or env("AMAZON_ADS_CLIENT_SECRET")
        self.refresh_token = refresh_token or env("AMAZON_ADS_REFRESH_TOKEN")
        self.profile_id = profile_id or env("AMAZON_ADS_PROFILE_ID")
        missing = [n for n, v in [("AMAZON_ADS_CLIENT_ID", self.client_id),
                                  ("AMAZON_ADS_CLIENT_SECRET", self.client_secret),
                                  ("AMAZON_ADS_REFRESH_TOKEN", self.refresh_token),
                                  ("AMAZON_ADS_PROFILE_ID", self.profile_id)] if not v]
        if missing:
            raise RuntimeError(f"缺少 Ads API 凭证: {', '.join(missing)}")
        self.base, self.token_url = REGIONS[(region or env("AMAZON_ADS_REGION") or "NA").upper()]
        self._token, self._expires = None, 0.0
        self.session = requests.Session()

    def _access_token(self):
        if not self._token or time.time() > self._expires - 60:
            r = self.session.post(self.token_url, data={
                "grant_type": "refresh_token", "refresh_token": self.refresh_token,
                "client_id": self.client_id, "client_secret": self.client_secret}, timeout=30)
            r.raise_for_status()
            body = r.json()
            self._token, self._expires = body["access_token"], time.time() + body.get("expires_in", 3600)
        return self._token

    def _request(self, method, path, media_type=None, **kw):
        headers = {
            "Amazon-Advertising-API-ClientId": self.client_id,
            "Amazon-Advertising-API-Scope": str(self.profile_id),
            "Authorization": f"Bearer {self._access_token()}",
        }
        if media_type:
            headers["Content-Type"] = headers["Accept"] = media_type
        for attempt in range(5):
            r = self.session.request(method, self.base + path, headers=headers, timeout=60, **kw)
            if r.status_code not in (429, 500, 502, 503):
                break
            time.sleep(2**attempt)
        r.raise_for_status()
        return r.json() if r.content else {}

    # ---------- 报表 ----------
    def sp_targeting_report(self, start_date, end_date, poll_seconds=15, timeout_minutes=30):
        """请求 SP 投放日报表（单次最长 31 天），完成后返回 DataFrame。"""
        body = {
            "name": f"galaxyknit-sp-targeting-{start_date}-{end_date}",
            "startDate": str(start_date), "endDate": str(end_date),
            "configuration": {
                "adProduct": "SPONSORED_PRODUCTS", "groupBy": ["targeting"], "columns": SP_TARGETING_COLUMNS,
                "reportTypeId": "spTargeting", "timeUnit": "DAILY", "format": "GZIP_JSON",
            },
        }
        rep = self._request("POST", "/reporting/reports", "application/vnd.createasyncreportrequest.v3+json",
                            data=json.dumps(body))
        report_id = rep["reportId"]
        deadline = time.time() + timeout_minutes * 60
        while True:
            st = self._request("GET", f"/reporting/reports/{report_id}")
            if st["status"] == "COMPLETED":
                raw = self.session.get(st["url"], timeout=120)
                raw.raise_for_status()
                return pd.DataFrame(json.loads(gzip.decompress(raw.content)))
            if st["status"] == "FAILED" or time.time() > deadline:
                raise RuntimeError(f"报表 {report_id} 失败或超时: {st}")
            time.sleep(poll_seconds)

    def pull_history(self, start_date, end_date):
        """按 31 天分段拉取更长时间范围。"""
        start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
        frames = []
        while start <= end:
            chunk_end = min(start + pd.Timedelta(days=30), end)
            frames.append(self.sp_targeting_report(start.date(), chunk_end.date()))
            start = chunk_end + pd.Timedelta(days=1)
        return pd.concat(frames, ignore_index=True)

    # ---------- 调价 ----------
    def update_bids(self, plan):
        """plan 为 plan_bid_updates 的输出。"""
        kw, tg = plan["keywords"], plan["targetingClauses"]
        results = []
        for i in range(0, len(kw), 1000):
            results.append(self._request("PUT", "/sp/keywords", "application/vnd.spKeyword.v3+json",
                                         data=json.dumps({"keywords": kw[i:i + 1000]})))
        for i in range(0, len(tg), 1000):
            results.append(self._request("PUT", "/sp/targets", "application/vnd.spTargetingClause.v3+json",
                                         data=json.dumps({"targetingClauses": tg[i:i + 1000]})))
        return results
