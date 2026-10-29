import numpy as np
import pandas as pd
import pytest

from galaxyknit_ads.ads_api import plan_bid_updates
from galaxyknit_ads.config import ACTIONS, get_strategy
from galaxyknit_ads.data import normalize, aggregate_daily
from galaxyknit_ads.dqn import MLP, DQNAgent
from galaxyknit_ads.env import OBS_DIM, BiddingEnv
from galaxyknit_ads.market import fit_market
from galaxyknit_ads.policy import recommend, to_bulk
from galaxyknit_ads.synth import generate_report, make_truth
from galaxyknit_ads.train import evaluate, train


@pytest.fixture(scope="module")
def report():
    truth = make_truth(30, seed=1)
    return truth, aggregate_daily(normalize(generate_report(truth, days=120, seed=2)))


def test_normalize_chinese_headers_and_currency():
    raw = pd.DataFrame({"日期": ["2026-10-01"], "广告活动名称": ["C"], "广告组名称": ["G"], "投放": ["knit sweater"],
                        "匹配类型": ["exact"], "展示量": ["1,200"], "点击量": [10], "花费": ["$12.50"],
                        "7天总销售额": ["￥80.00"], "7天总订单数(#)": [2]})
    df = normalize(raw)
    assert df.loc[0, "impressions"] == 1200 and df.loc[0, "spend"] == 12.5 and df.loc[0, "sales"] == 80
    assert df.loc[0, "match_type"] == "EXACT"


def test_normalize_ads_api_splits_keyword_and_target_ids():
    raw = pd.DataFrame({"date": ["2026-10-01"] * 2, "campaignName": ["C"] * 2, "adGroupName": ["G"] * 2,
                        "targeting": ["sweater", "asin=B0X"], "matchType": ["BROAD", "TARGETING_EXPRESSION"],
                        "keywordId": [111, 222], "keywordBid": [1.2, 0.9], "impressions": [100, 50],
                        "clicks": [3, 1], "cost": [2.4, 0.7], "sales7d": [40, 0], "purchases7d": [1, 0]})
    df = normalize(raw)
    assert df["keyword_id"].iloc[0] == "111" and pd.isna(df["keyword_id"].iloc[1])
    assert df["target_id"].tolist()[1] == "222"
    assert df["bid"].tolist() == [1.2, 0.9]


def test_fit_recovers_bid_and_elasticity(report):
    truth, df = report
    m = fit_market(df)
    j = m.params.merge(truth, on=["campaign", "ad_group", "targeting"], suffixes=("", "_t"))
    assert len(j) >= 25
    assert np.median(j["ref_bid"] / j["ref_bid_t"]) == pytest.approx(1.0, abs=0.1)
    assert np.corrcoef(j["elasticity"], j["elasticity_t"])[0, 1] > 0.5


def test_env_shapes_and_bid_limits(report):
    m = fit_market(report[1])
    st = get_strategy("growth")
    env = BiddingEnv(m, st, n_envs=64, seed=0)
    obs = env.reset()
    assert obs.shape == (64, OBS_DIM) and np.isfinite(obs).all()
    for _ in range(10):
        obs, r, done, info = env.step(np.full(64, len(ACTIONS) - 1))
    ref = env.p_obs["ref_bid"]
    assert (env.bid <= ref * st.rel_bid_cap + 0.01).all() and (env.bid <= st.max_bid).all()
    assert np.isfinite(r).all()


def test_mlp_gradient_matches_numeric():
    rng = np.random.default_rng(0)
    net = MLP([5, 8, 3], rng)
    x = rng.normal(size=(4, 5))
    out, acts = net.forward(x)
    grads = net.backward(acts, np.ones_like(out))
    W = net.W[0]
    i, j, h = 1, 2, 1e-6
    W[i, j] += h
    up = net.forward(x)[0].sum()
    W[i, j] -= 2 * h
    down = net.forward(x)[0].sum()
    W[i, j] += h
    assert grads[0][i, j] == pytest.approx((up - down) / (2 * h), rel=1e-4, abs=1e-8)


def test_training_beats_holding_bids(report):
    m = fit_market(report[1])
    st = get_strategy("growth")
    agent, _ = train(m, st, episodes=12, n_envs=128, log=lambda *_: None)
    ev = evaluate(m, st, agent, n_envs=256).set_index("策略")
    assert ev.loc["RL 智能体", "平均奖励"] > ev.loc["保持当前竞价", "平均奖励"]


def test_recommend_and_exports(report, tmp_path):
    m = fit_market(report[1])
    st = get_strategy("growth")
    agent = DQNAgent(OBS_DIM, len(ACTIONS))
    agent.save(tmp_path)
    agent = DQNAgent.load(tmp_path)
    recs = recommend(report[1], m, agent, st)
    assert len(recs) == len(m)
    assert (recs["建议竞价"] >= st.min_bid).all() and (recs["建议竞价"] <= st.max_bid).all()
    bulk = to_bulk(recs, only_changes=False)
    assert set(bulk["Entity"]) <= {"Keyword", "Product Targeting"} and len(bulk) == len(recs)
    plan = plan_bid_updates(recs)
    assert len(plan["keywords"]) + len(plan["targetingClauses"]) == (recs["动作"] != "保持").sum()
