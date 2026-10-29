"""命令行入口: python -m galaxyknit_ads <command>"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PRESETS, Strategy, get_strategy


def _print_eval(title, df):
    print(f"\n=== {title} ===")
    show = df.copy()
    for c in ("展示", "点击", "订单"):
        show[c] = show[c].map("{:,.0f}".format)
    for c in ("花费", "销售额", "净贡献(销售-花费/maxACoS)"):
        show[c] = show[c].map("{:,.0f}".format)
    for c in ("ACoS", "销售额vs保持", "订单vs保持"):
        show[c] = show[c].map("{:+.1%}".format if "vs" in c else "{:.1%}".format)
    show["平均奖励"] = show["平均奖励"].map("{:.2f}".format)
    print(show.to_string(index=False))


def cmd_synth(a):
    from .synth import generate_report, make_truth
    truth = make_truth(a.keywords, seed=a.seed)
    rep = generate_report(truth, start=a.start, days=a.days, seed=a.seed + 1, include_bid=not a.no_bid)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rep.to_csv(a.out, index=False)
    truth.to_csv(Path(a.out).with_suffix(".truth.csv"), index=False)
    print(f"已生成 {len(rep):,} 行模拟报表 → {a.out}")


def cmd_pull(a):
    from .ads_api import AdsClient
    df = AdsClient().pull_history(a.start, a.end)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"已从 Ads API 拉取 {len(df):,} 行 → {a.out}")


def cmd_train(a):
    from .data import load_reports
    from .market import fit_market
    from .train import evaluate, save_run, train
    st = get_strategy(a.strategy, max_acos=a.max_acos, target_acos=a.target_acos,
                      decision_interval=a.interval, max_bid=a.max_bid)
    df = load_reports(a.report)
    market = fit_market(df)
    print(f"数据: {df['date'].min().date()} ~ {df['date'].max().date()}，"
          f"{df['key'].nunique()} 个投放，拟合 {len(market)} 个（展示 ≥ 30）。"
          + ("  ⚠ 报表无竞价列，已用 CPC/0.8 估算竞价。" if market.globals["bid_estimated"] else ""))
    print(f"策略: {st.name}  边际ACoS上限={st.max_acos:.0%}  目标ACoS={st.target_acos:.0%}  "
          f"每 {st.decision_interval} 天决策一次")
    agent, hist = train(market, st, episodes=a.episodes, n_envs=a.envs, seed=a.seed)

    evals = {"全年": evaluate(market, st, agent, seed=a.seed + 100)}
    nxt = np.datetime64(market.globals["last_date"]) + np.timedelta64(1, "D")
    evals["未来28天"] = evaluate(market, st, agent, seed=a.seed + 200, start_dates=np.full(1024, nxt))
    _print_eval("模拟评估：全年随机起点 28 天", evals["全年"])
    _print_eval(f"模拟评估：从 {nxt} 起 28 天", evals["未来28天"])
    save_run(a.out, agent, market, st, hist, evals)
    print(f"\n模型已保存 → {a.out}")


def cmd_recommend(a):
    from .data import load_reports
    from .dqn import DQNAgent
    from .market import Market, fit_market
    from .policy import recommend, summarize, to_bulk
    model = Path(a.model)
    agent = DQNAgent.load(model)
    market = Market.load(model)
    st = Strategy.from_dict(json.loads((model / "strategy.json").read_text()))
    df = load_reports(a.report)
    current = fit_market(df)
    recs = recommend(df, market, agent, st, current_market=current)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    recs.to_csv(a.out, index=False, encoding="utf-8-sig")
    print(json.dumps(summarize(recs), ensure_ascii=False))
    print(recs[["targeting", "match_type", "当前竞价", "建议竞价", "28天点击", "28天订单", "28天ACoS", "标记"]]
          .head(15).to_string(index=False))
    print(f"\n调价建议 → {a.out}")
    if a.bulk:
        bulk = to_bulk(recs)
        if bulk.empty:
            print("报表缺少 Keyword ID / Product Targeting ID，无法生成批量上传表。")
        else:
            bulk.to_excel(a.bulk, index=False, sheet_name="Sponsored Products Campaigns")
            print(f"批量上传表（{len(bulk)} 行）→ {a.bulk}")


def cmd_push(a):
    from .ads_api import AdsClient, plan_bid_updates
    recs = pd.read_csv(a.recs, dtype={"keyword_id": str, "target_id": str})
    max_change = recs["调整幅度"].abs().max()
    if max_change > 0.5:
        raise SystemExit(f"存在超过 ±50% 的调价（{max_change:.0%}），请检查建议文件。")
    plan = plan_bid_updates(recs)
    print(f"将更新 {len(plan['keywords'])} 个关键词、{len(plan['targetingClauses'])} 个商品投放。")
    if not a.apply:
        print("[dry-run] 未发送任何请求。确认无误后加 --apply 执行。")
        return
    res = AdsClient().update_bids(plan)
    print(json.dumps(res, ensure_ascii=False, indent=2)[:4000])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="galaxyknit_ads", description="亚马逊 SP 广告竞价强化学习智能体")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="生成模拟针织类目报表")
    s.add_argument("--out", default="data/sample_report.csv")
    s.add_argument("--keywords", type=int, default=120)
    s.add_argument("--days", type=int, default=180)
    s.add_argument("--start", default="2026-05-01")
    s.add_argument("--no-bid", action="store_true", help="不输出竞价列（模拟后台下载的投放报告）")
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(fn=cmd_synth)

    s = sub.add_parser("pull", help="通过 Ads API 拉取 SP 投放日报表")
    s.add_argument("--start", required=True)
    s.add_argument("--end", required=True)
    s.add_argument("--out", default="data/api_report.csv")
    s.set_defaults(fn=cmd_pull)

    s = sub.add_parser("train", help="拟合市场模型 + 训练 RL 策略 + 模拟评估")
    s.add_argument("--report", nargs="+", required=True, help="一个或多个报表文件（CSV/XLSX/JSON）")
    s.add_argument("--strategy", choices=sorted(PRESETS), default="growth")
    s.add_argument("--max-acos", type=float)
    s.add_argument("--target-acos", type=float)
    s.add_argument("--max-bid", type=float)
    s.add_argument("--interval", type=int, help="每隔几天调一次价")
    s.add_argument("--episodes", type=int, default=120)
    s.add_argument("--envs", type=int, default=256)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="models/latest")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("recommend", help="对最新报表生成调价建议")
    s.add_argument("--report", nargs="+", required=True)
    s.add_argument("--model", default="models/latest")
    s.add_argument("--out", default="output/bid_recommendations.csv")
    s.add_argument("--bulk", help="同时导出批量上传 Excel，如 output/bulk_update.xlsx")
    s.set_defaults(fn=cmd_recommend)

    s = sub.add_parser("push", help="通过 Ads API 推送调价（默认 dry-run）")
    s.add_argument("--recs", default="output/bid_recommendations.csv")
    s.add_argument("--apply", action="store_true")
    s.set_defaults(fn=cmd_push)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
