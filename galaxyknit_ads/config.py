"""投放策略配置。

奖励函数: reward = 销售额 - 花费 / max_acos
即每多花 1 元，只要能多带来 1/max_acos 元销售就值得花。max_acos 是“边际 ACoS 上限”，
整体平均 ACoS 会低于它。冲量策略把它设得较高，以换取更多曝光和订单。
"""

from dataclasses import asdict, dataclass, replace

# 每次决策可选的竞价倍数（相对当前竞价）
ACTIONS = (0.70, 0.85, 0.95, 1.00, 1.05, 1.15, 1.30)
HOLD_ACTION = ACTIONS.index(1.00)


@dataclass(frozen=True)
class Strategy:
    name: str = "growth"
    max_acos: float = 0.50  # 边际 ACoS 上限（奖励函数核心参数）
    target_acos: float = 0.35  # 平均 ACoS 目标（规则基线 & 报告参考）
    min_bid: float = 0.15
    max_bid: float = 6.00
    rel_bid_floor: float = 0.30  # 竞价不低于参考竞价的倍数
    rel_bid_cap: float = 3.00  # 竞价不高于参考竞价的倍数
    change_penalty: float = 0.02  # 调价惩罚，抑制无意义的频繁抖动
    decision_interval: int = 1  # 每隔几天做一次调价决策
    negative_clicks: int = 20  # 28 天内点击 >= 该值且 0 单 → 标记否定/暂停

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


PRESETS = {
    # 冲量：新品/旺季抢排名，接受较高 ACoS
    "growth": Strategy(),
    "balanced": Strategy(name="balanced", max_acos=0.35, target_acos=0.25),
    "profit": Strategy(name="profit", max_acos=0.25, target_acos=0.18, change_penalty=0.03),
}


def get_strategy(name="growth", **overrides):
    s = PRESETS[name]
    overrides = {k: v for k, v in overrides.items() if v is not None}
    return replace(s, **overrides) if overrides else s
