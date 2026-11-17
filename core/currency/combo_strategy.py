"""货币战争 S2「连击阿格莱雅 / 那刻夏」专精经济、D牌、升级、站位与连胜连败决策模块。"""

from core.currency.equipment_db import (
    CARRY_UNITS,
    HARMONY_SUPPORT_UNITS,
    SUSTAIN_UNITS,
)
from core.currency.match_db import MatchDatabase, SlotRecord
from core.currency.s2_strategy import SPECIAL_NAMES, ShopOffer

COMBO_FRONT_ORDER = ("阿格莱雅", "那刻夏", "雪衣", "千冶·刃", "云璃", "瓦尔特", "镜流", "飞霄")
COMBO_BACK_ORDER = (
    "刻律德菈",
    "白露",
    "风堇",
    "寒鸦",
    "星期日",
    "花火",
    "知更鸟",
    "藿藿",
    "缇宝",
    "符玄",
    "砂金",
    "丹恒·腾荒",
)

HEALER_NAMES = frozenset({"白露", "风堇", "藿藿", "罗刹", "灵砂", "加拉赫", "娜塔莎", "玲可"})
BACK_PREFERRED = frozenset({"刻律德菈", "白露", "风堇", "藿藿", "寒鸦", "符玄", "砂金", "缇宝", "长夜月"})
FRONT_COMPETITORS_EXCLUDE = BACK_PREFERRED

# 核心追求星级与商店费用表（用于买卡、升星与高费卡留钱判断）
CORE_STAR_TARGETS: dict[str, int] = {
    "阿格莱雅": 2,
    "那刻夏": 3,
    "刻律德菈": 3,
    "星期日": 2,
    "花火": 2,
    "知更鸟": 2,
    "白露": 2,
    "风堇": 2,
    "寒鸦": 2,
    "缇宝": 2,
    "符玄": 2,
    "藿藿": 2,
    "雪衣": 2,
    "千冶·刃": 2,
}

UNIT_COSTS: dict[str, int] = {
    "那刻夏": 1,
    "刻律德菈": 1,
    "灵砂": 1,
    "瓦尔特": 1,
    "火花": 1,
    "星期日": 2,
    "花火": 2,
    "知更鸟": 2,
    "白露": 2,
    "云璃": 2,
    "飞霄": 3,
    "镜流": 3,
    "风堇": 4,
    "寒鸦": 4,
    "阿格莱雅": 5,
}

BOSS_STAGES = frozenset({"1-9", "2-7", "3-7"})
PRE_REWARD_STAGES = frozenset({"1-2", "1-3", "1-5", "1-7", "2-2", "2-3", "2-5", "3-2", "3-5"})
ALL_COMBO_NAMES = tuple(dict.fromkeys((*COMBO_FRONT_ORDER, *COMBO_BACK_ORDER)))


def streak_regime(db: MatchDatabase) -> str:
    """根据小队血量、关卡和连胜连败返回当前运营态势。

    Returns:
        ``"desperate"`` / ``"recovery"`` / ``"win_streak"`` / ``"normal"``。
    """
    if db.stage == "3-7" or (db.hp is not None and db.hp <= 30):
        return "desperate"
    if db.loss_streak >= 1 or (db.hp is not None and db.hp <= 60):
        return "recovery"
    if db.win_streak >= 2 and (db.hp is None or db.hp >= 70):
        return "win_streak"
    return "normal"


def _owned_records(db: MatchDatabase, name: str) -> list[SlotRecord]:
    return [r for r in db.slots.values() if r.occupied and r.name == name]


def _can_merge(records: list[SlotRecord], stars: int = 1) -> bool:
    return sum(1 for r in records if r.stars == stars) >= 2


def _exp_needed_for_level(db: MatchDatabase, xp_cost: int | None) -> int | None:
    if xp_cost is None or xp_cost <= 0 or db.level is None or db.level >= 9:
        return None
    if db.exp_cur is not None and db.exp_req is not None and db.exp_req > db.exp_cur:
        clicks = (db.exp_req - db.exp_cur + 3) // 4
        return clicks * xp_cost
    return None


def _deployable_bench_count(db: MatchDatabase) -> int:
    field_names = {r.name for r in db.field_records() if r.name}
    return sum(
        1
        for r in db.bench_records()
        if r.occupied and (not r.name or r.name not in field_names)
    )


def _best_bench_empty(db: MatchDatabase) -> int:
    return sum(1 for r in db.bench_records() if not r.occupied)


def _interest_floor(db: MatchDatabase) -> int:
    """计算本回合应严守的整十利息底线，兼顾奖励关前保息与连败止损。"""
    gold = db.gold or 0
    regime = streak_regime(db)
    if regime == "desperate" or db.stage in BOSS_STAGES:
        return 0
    if regime == "recovery":
        cap = 0 if (db.loss_streak >= 2 or (db.hp is not None and db.hp <= 45)) else 20
    elif db.plane == 1:
        cap = 40 if db.stage in PRE_REWARD_STAGES else 30
    else:
        cap = 50
    bracket = (gold // 10) * 10
    floor = min(cap, bracket)
    # 距下一个整十利息档仅差 1~2 金且非高压局时，锁住本回合余钱跨档吃息
    if db.stage in PRE_REWARD_STAGES and regime in ("win_streak", "normal") and gold >= 8 and (gold % 10) <= 2:
        floor = max(floor, bracket)
    return min(gold, floor)


def _offer_priority(db: MatchDatabase, offer: ShopOffer) -> tuple[int, bool]:
    """给出单张商店卡牌对「连击阿格莱雅/那刻夏」的战术分值与是否允许破利息购买。"""
    name = offer.name
    if not name or name in SPECIAL_NAMES:
        return -100, False

    regime = streak_regime(db)
    field_names = {r.name for r in db.field_records() if r.name}
    owned = _owned_records(db, name)
    best_star = max((r.stars for r in owned), default=0)
    will_merge = _can_merge(owned, stars=1)
    has_healer = bool(field_names & HEALER_NAMES)
    has_anaxa_2 = any(r.name == "那刻夏" and r.stars >= 2 for r in owned)

    # 1. 5 费阿格莱雅优先级最高：一旦能拿必须无视普通利息底线立刻锁定大核
    if name == "阿格莱雅":
        if best_star >= 3:
            return -10, False
        score = 165 if will_merge or name not in field_names else 135
        return score, True

    # 2. 1 费前期核心那刻夏（追到 3 星前必买）
    if name == "那刻夏":
        if best_star >= 3:
            return -10, False
        if not has_anaxa_2:
            return (150 if will_merge else 132), True
        if best_star < 3 and (db.plane <= 2 or will_merge):
            return (122 if will_merge else 95), will_merge

    # 3. 缺治疗保命位（连败或无奶必抓）
    if name in HEALER_NAMES and not has_healer:
        return (126 if regime == "recovery" else 112), True

    # 4. 体系关键挂件与核心追踪角标卡
    if name in CORE_STAR_TARGETS:
        target = CORE_STAR_TARGETS[name]
        if best_star >= target and owned:
            return -10, False
        rank_idx = ALL_COMBO_NAMES.index(name) if name in ALL_COMBO_NAMES else 12
        if will_merge:
            score = 128 - rank_idx
            break_interest = regime in ("recovery", "desperate") or name in field_names
            return score, break_interest
        if name not in field_names:
            score = 114 - rank_idx * 2
            # 1-2 费核心（刻律德菈/星期日/花火/知更鸟/白露）优先凑 6 连击羁绊
            break_interest = (
                name in ("刻律德菈", "星期日", "白露", "风堇", "寒鸦")
                or regime in ("recovery", "desperate")
            )
            return score, break_interest
        return 76 - rank_idx, False

    # 5. 其他攻略角标追踪卡（若与计划兼容且缺牌可买）
    if offer.column is not None and name in (HARMONY_SUPPORT_UNITS | SUSTAIN_UNITS):
        if name not in field_names and db.placed + _deployable_bench_count(db) < db.capacity:
            return 84 if name in SUSTAIN_UNITS else 70, regime == "recovery"

    # 6. 前期人数未满买低费保血过渡卡
    if db.placed + _deployable_bench_count(db) < db.capacity and offer.cost <= 2 and db.plane <= 2:
        is_sustain = name in SUSTAIN_UNITS or name in HEALER_NAMES
        return (62 if is_sustain else 46), db.placed == 0

    return -10, False


def decide_combo_shop(
    db: MatchDatabase,
    offers: list[ShopOffer],
    tracked_cols: tuple[int, ...],
    refresh_cost: int | None,
    xp_cost: int | None,
    refreshes: int,
    ops: int,
    *,
    max_ops: int = 10,
    target_level: int = 9,
) -> tuple[str, ShopOffer | None]:
    """针对「连击阿格莱雅/那刻夏」给出商店唯一最佳单步动作。

    Returns:
        ``("buy", offer)`` / ``("sell", None)`` / ``("exp", None)`` / ``("refresh", None)`` / ``("done", None)``。
    """
    gold = db.gold
    if gold is None or ops >= max_ops:
        return "done", None

    regime = streak_regime(db)
    floor = _interest_floor(db)
    bench_empty = _best_bench_empty(db)
    level = db.level or 3
    tracked_set = set(tracked_cols)
    need_exp_for_next = _exp_needed_for_level(db, xp_cost)

    # 评估商店每张可买卡的价值
    evaluated: list[tuple[int, int, ShopOffer, bool]] = []
    for offer in offers:
        if offer.cost > gold or offer.name in SPECIAL_NAMES:
            continue
        score, allow_break = _offer_priority(db, offer)
        if offer.column in tracked_set:
            if score < 0:
                # 攻略追踪额外认定的凑羁绊挂件
                field_names = {r.name for r in db.field_records()}
                if offer.name not in field_names and db.placed + _deployable_bench_count(db) < db.capacity:
                    score = 78
            else:
                score += 12
        if score <= 0:
            continue
        effective_budget = gold if allow_break else max(0, gold - floor)
        if offer.cost <= effective_budget:
            evaluated.append((score, -offer.cost, offer, allow_break))

    # 1. 有高价值可买卡：若会触发 3 张自动升星或备战有空位直接购买；满席时先卖冗余腾位
    if evaluated:
        evaluated.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best_score, _, best_offer, _ = evaluated[0]
        will_merge = _can_merge(_owned_records(db, best_offer.name))
        if bench_empty > 0 or will_merge:
            return "buy", best_offer
        if best_score >= 95 and plan_combo_sell(db) is not None:
            return "sell", None

    # 2. 按连胜/连败态势仲裁：买经验冲人口 vs D 牌找关键升星/阿格莱雅
    has_aglaea = any(r.name == "阿格莱雅" for r in db.slots.values() if r.occupied)
    anaxa_records = _owned_records(db, "那刻夏")
    anaxa_ready = any(r.stars >= 2 for r in anaxa_records)
    has_pending_field_unit = db.placed < db.capacity and _deployable_bench_count(db) > 0

    max_refreshes = {
        "desperate": 7,
        "recovery": 5,
        "win_streak": 1 if db.plane == 1 else (2 if level < 8 else 4),
        "normal": 2 if db.plane <= 2 else 4,
    }[regime]
    if db.stage in PRE_REWARD_STAGES and regime in ("win_streak", "normal") and anaxa_ready:
        max_refreshes = min(max_refreshes, 1)

    can_exp = xp_cost is not None and xp_cost > 0 and level < target_level and not has_pending_field_unit
    can_refresh = (
        db.stage != "1-1"
        and refresh_cost is not None
        and refreshes < max_refreshes
        and (bench_empty > 0 or plan_combo_sell(db) is not None)
    )

    # 濒死急救/连败低血优先 D 牌搜升星与治疗；高血连胜或已备好 2 星那刻夏优先按 F 拉人口找 5 费阿格莱雅
    prefer_roll_first = (
        regime == "desperate"
        or (regime == "recovery" and (not anaxa_ready or not any(r.name in HEALER_NAMES for r in db.field_records())))
        or (level >= 8 and not has_aglaea and db.plane == 3)
        or (level in (5, 6) and not anaxa_ready and db.plane >= 2)
    )

    if prefer_roll_first and can_refresh:
        roll_reserve = 0 if regime == "desperate" else floor
        if gold >= refresh_cost + roll_reserve + (0 if refresh_cost == 0 else 1):
            return "refresh", None

    if can_exp:
        exp_reserve = 0 if db.stage == "3-7" else floor
        if regime == "desperate" and db.stage != "3-7":
            # 濒死低血时除非本回合点 F 能立刻升级+1人口，否则绝不把救命钱换成半管经验
            if need_exp_for_next is not None and gold >= need_exp_for_next:
                return "exp", None
        else:
            # 优先花费不跌破整十利息的余钱购买经验；临近奖惩前仅当本手够升级或足够多 4 金整点时才按 F
            if gold - xp_cost >= exp_reserve:
                return "exp", None
            if need_exp_for_next is not None and gold - need_exp_for_next >= exp_reserve:
                return "exp", None

    if can_refresh:
        roll_reserve = 0 if regime == "desperate" else floor
        min_after = 2 if db.stage == "3-7" else max(roll_reserve, roll_reserve + 1)
        if refresh_cost == 0 or (
            gold - refresh_cost >= min_after
            and not (db.stage in PRE_REWARD_STAGES and regime == "win_streak" and anaxa_ready)
        ):
            return "refresh", None

    return "done", None


def plan_combo_sell(
    db: MatchDatabase,
    *,
    for_interest: bool = False,
) -> SlotRecord | None:
    """从备战席挑选最该变现卖出的冗余 1 星未追踪/非核心空装角色。

    Args:
        for_interest: True 时仅当卖出 1 张垃圾过渡卡即可跨过整十利息门槛才返回。
    """
    gold = db.gold or 0
    candidates: list[tuple[int, int, SlotRecord]] = []
    field_names = {r.name for r in db.field_records() if r.name}
    for rec in db.bench_records():
        if not rec.occupied or rec.equip_count > 0 or rec.worn_items or rec.has_halo:
            continue
        if rec.name in CARRY_UNITS:
            continue
        # 若备战这张 1 星卡加上另一张即将合并为高星，不轻易卖
        owned = _owned_records(db, rec.name) if rec.name else []
        if rec.name in CORE_STAR_TARGETS and len(owned) >= 2:
            continue
        if rec.stars >= 2 and rec.name in CORE_STAR_TARGETS:
            continue

        # 分值越小越先卖：未追踪杂卡(0) -> 非核心未上阵卡(10) -> 冗余 1 星低费挂件(20)
        if not rec.is_tracked and (not rec.name or rec.name not in CORE_STAR_TARGETS):
            pri = 0
        elif rec.name and rec.name not in field_names and rec.name in CORE_STAR_TARGETS:
            # 留给上阵位，除非实在缺格子
            pri = 60
        elif not rec.name or rec.name not in CORE_STAR_TARGETS:
            pri = 15
        else:
            pri = 45

        if pri >= 40 and for_interest:
            continue
        candidates.append((pri, -rec.index, rec))

    if not candidates:
        return None
    candidates.sort(key=lambda it: (it[0], it[1]))
    picked = candidates[0][2]

    if for_interest:
        if db.stage == "3-7" or streak_regime(db) == "desperate":
            return None
        sale_val = max(1, picked.cost or 1)
        cur_bracket = gold // 10
        new_bracket = min(5, (gold + sale_val) // 10)
        if new_bracket <= cur_bracket:
            return None

    return picked


def _zone_fit_score(rec: SlotRecord, zone: str, has_aglaea: bool) -> int:
    name = rec.name
    if zone == "front":
        if name == "阿格莱雅":
            return 300
        if name == "那刻夏":
            return 220 if has_aglaea else 280
        if name in COMBO_FRONT_ORDER:
            return 180 - COMBO_FRONT_ORDER.index(name) * 10 + rec.stars * 12
        if name in BACK_PREFERRED:
            return 10
        return 80 + rec.stars * 8 + rec.equip_count * 6
    if name in COMBO_BACK_ORDER:
        return 220 - COMBO_BACK_ORDER.index(name) * 10 + rec.stars * 12
    if name in SUSTAIN_UNITS:
        return 140 + rec.stars * 10
    if name in CARRY_UNITS:
        return 5
    return 60 + rec.stars * 8


def plan_combo_deployment(db: MatchDatabase) -> tuple[SlotRecord, str, int] | None:
    """结合记忆数据库给出一步前后台补位/换位建议：``(源格记录, 目标区, 目标格号)``。

    保证：
        - 前台优先放置 ``阿格莱雅`` / ``那刻夏``，不把后台奶/拐硬塞到前台；
        - 已穿装备或持光环的核心不被随便替换下场；
        - 有空余人口时优先将高价值连击/奇兽角色填满场上阵容。
    """
    field = db.field_records()
    field_names = {r.name for r in field if r.name}
    front = [r for r in db.iter_zone("front") if not r.blocked]
    back = [r for r in db.iter_zone("back") if not r.blocked]
    front_occupied = [r for r in front if r.occupied]
    has_aglaea = any(r.name == "阿格莱雅" for r in db.slots.values() if r.occupied)

    bench_candidates = [
        r for r in db.bench_records()
        if r.occupied and (not r.name or r.name not in field_names)
    ]
    if not bench_candidates:
        return None

    empty_front = [r for r in front if not r.occupied]
    empty_back = [r for r in back if not r.occupied]

    # 1. 人数尚未填满上限时优先从备战拖入空位
    if db.placed < db.capacity:
        best_fill: tuple[int, SlotRecord, str, int] | None = None
        for src in bench_candidates:
            choices: list[tuple[int, str, int]] = []
            if src.can_front and empty_front:
                f_score = _zone_fit_score(src, "front", has_aglaea)
                # 前台无人时优先填前台避免系统报错阻止开战
                if not front_occupied:
                    f_score += 200
                choices.append((f_score, "front", empty_front[0].index))
            if src.can_back and empty_back and front_occupied:
                b_score = _zone_fit_score(src, "back", has_aglaea)
                choices.append((b_score, "back", empty_back[0].index))
            if not choices:
                continue
            sc, z, idx = max(choices, key=lambda c: c[0])
            if best_fill is None or sc > best_fill[0]:
                best_fill = (sc, src, z, idx)
        if best_fill is not None:
            return best_fill[1], best_fill[2], best_fill[3]

    # 2. 满员时仅让明显更高顺位且带星级的真命核心换掉 0 装备无光环的非核心过渡卡
    best_swap: tuple[int, SlotRecord, str, int] | None = None
    for src in bench_candidates:
        if not src.name or src.name not in ALL_COMBO_NAMES:
            continue
        target_zones = (("front", front), ("back", back))
        for zone, row in target_zones:
            if (zone == "front" and not src.can_front) or (zone == "back" and not src.can_back):
                continue
            in_score = _zone_fit_score(src, zone, has_aglaea)
            for dst in row:
                if not dst.occupied or dst.has_halo or dst.equip_count > 0 or dst.worn_items:
                    continue
                if dst.name == "阿格莱雅":
                    continue
                if dst.name in CORE_STAR_TARGETS and src.stars <= dst.stars and src.name not in CARRY_UNITS:
                    continue
                if dst.name in HEALER_NAMES and src.name not in HEALER_NAMES:
                    if sum(1 for r in field if r.name in HEALER_NAMES) <= 1:
                        continue
                out_score = _zone_fit_score(dst, zone, has_aglaea)
                gain = in_score - out_score
                if gain >= 55 and (best_swap is None or gain > best_swap[0]):
                    best_swap = (gain, src, zone, dst.index)
    if best_swap is not None:
        return best_swap[1], best_swap[2], best_swap[3]

    return None
