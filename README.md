# Galaxyknit 亚马逊 SP 广告竞价智能体

这是一个用强化学习（Double DQN）做的关键词竞价智能体，针对针织服装类目，默认走「冲量」策略。只依赖 numpy 和 pandas，不需要 GPU。

```
报表 CSV/Excel ─┐
                ├─► 拟合市场模型 ─► 模拟器 ─► DQN 训练 ─► 调价建议 / 批量上传表
Ads API 拉数 ───┘   （每个关键词）                         └─► Ads API 推送（默认 dry-run）
```

## 快速开始

```bash
pip install -r requirements.txt

# 1. 准备数据：二选一
#   a) 卖家后台 → 广告 → 报告 → 商品推广 → 「投放」报告，时间单位选「每日」，导出至少 60 天，建议 90–180 天
#   b) 通过 Ads API 拉取（需要先设置环境变量，见下文）
python -m galaxyknit_ads pull --start 2026-05-01 --end 2026-10-28 --out data/api_report.csv
#   没有数据时可以先用模拟数据跑通流程
python -m galaxyknit_ads synth --out data/sample_report.csv

# 2. 训练（4 核 CPU 约 1 分钟），训练完会打印模拟评估
python -m galaxyknit_ads train --report data/sample_report.csv --strategy growth --out models/latest

# 3. 用最新报表生成调价建议
python -m galaxyknit_ads recommend --report data/latest.csv --model models/latest \
    --out output/bid_recommendations.csv --bulk output/bulk_update.xlsx

# 4. 执行：上传 bulk_update.xlsx 到后台批量操作，或者通过 API 推送
python -m galaxyknit_ads push --recs output/bid_recommendations.csv          # dry-run
python -m galaxyknit_ads push --recs output/bid_recommendations.csv --apply  # 真正执行
```

## 策略与奖励

每天的奖励是 **销售额 − 花费 / max_acos**。意思是：每多花 1 元，只要能多带来 1/max_acos 元的销售额就算值得。max_acos 控制的是边际 ACoS，所以账户整体的平均 ACoS 会比它低。

| 预设 | max_acos | target_acos | 适用场景 |
|---|---|---|---|
| `growth`（默认） | 50% | 35% | 新品、旺季、冲排名 |
| `balanced` | 35% | 25% | 平稳期 |
| `profit` | 25% | 18% | 清库存后控利润 |

可用的参数覆盖：`--max-acos 0.6`、`--target-acos`、`--max-bid`，以及 `--interval 3`（每 3 天调一次价，适合手动上传批量表的节奏）。

## 原理

1. **市场模型**（`market.py`）：给每个关键词拟合竞价→展示的弹性、CTR 和 CVR 的 Beta 后验（向同匹配类型的均值收缩）、客单价、CPC 与竞价的比值，再加上季节和星期系数。数据不足 3 个月时，季节曲线用针织类目的先验（10–12 月是旺季，并且旺季竞争更激烈）。
2. **模拟器**（`env.py`）：每个训练回合都会从后验里采样一组“真实”参数，所以训练出来的策略在 CVR/CTR 估计有误差时也能稳定工作。
3. **智能体**（`dqn.py`）：每个关键词每天的观测是 26 维，包括近 7 天和 28 天的表现、相对盈亏平衡 CPC 的出价水平、后验估计、季节、星期和匹配类型。可选动作是 7 档竞价倍数：×0.70 / 0.85 / 0.95 / 1.00 / 1.05 / 1.15 / 1.30。
4. **安全护栏**：竞价被限制在 [min_bid, max_bid] 和参考竞价的 [0.3×, 3×] 之内，单次调价最多 ±30%；推送时如果发现超过 ±50% 的调价会直接拒绝；28 天内点击 ≥ 20 但 0 单的投放会被标记为“考虑否定/暂停”。

## 模拟数据上的结果（120 个关键词，180 天历史，未来 28 天，冲量策略）

这组结果是在生成数据所用的**真实参数**下回测的，不是在拟合出来的模型上，所以包含了模型估计误差的影响：

| 策略 | 销售额 | 订单 | ACoS | 净贡献 |
|---|---|---|---|---|
| 保持当前竞价 | 1,815,526 | 49,032 | 22.4% | 1,002,809 |
| 规则调价（按 ACoS 每 3 天调整） | 2,662,192 | 69,186 | 25.1% | 1,328,098 |
| **RL 智能体** | **3,075,692** | **76,588** | 26.3% | **1,458,125** |

## 数据要求和注意事项

- 支持的格式：卖家后台英文或中文表头的 CSV/XLSX，以及 Ads API v3 的 `spTargeting` 报表。货币符号和千分位会自动清洗。
- **建议报表里带上竞价列**（API 拉取的数据自带 `keywordBid`）。后台下载的投放报告没有竞价列，这时会用 CPC/0.8 来估算，弹性的估计会差一些。
- 要导出批量上传表，报表里需要有 Campaign ID / Ad Group ID / Keyword ID 或 Product Targeting ID（API 数据自带）。
- 销售额采用 7 天归因，最近 7 天的数据还没完全回传，会让近期 ACoS 看起来偏高。建议用截至 2–3 天前的数据来出建议。
- 模拟评估的前提是市场模型足够准确。上线时建议先拿 20–30% 的关键词做 A/B，跑 2 周对比后再全量。
- 每周用最新数据重新训练一次（`train`），每天或每 3 天跑一次 `recommend`。

## Ads API 凭证

```bash
export AMAZON_ADS_CLIENT_ID=amzn1.application-oa2-client.xxx
export AMAZON_ADS_CLIENT_SECRET=xxx
export AMAZON_ADS_REFRESH_TOKEN=Atzr|xxx
export AMAZON_ADS_PROFILE_ID=1234567890
export AMAZON_ADS_REGION=NA   # NA / EU / FE
```

## 目录

```
galaxyknit_ads/
  config.py   策略预设、动作空间
  data.py     报表读取与标准化（中英文表头 / API）
  market.py   市场响应模型拟合 + 单日模拟
  env.py      向量化竞价环境 + 观测构造（训练和上线共用）
  dqn.py      numpy 实现的 Double DQN
  train.py    训练循环 + 基线对比评估
  policy.py   调价建议 + 批量上传表
  ads_api.py  Amazon Ads API v3（报表、调价）
  synth.py    针织类目模拟数据
tests/        python -m pytest
```
