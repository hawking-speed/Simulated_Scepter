"""货币战争 S2「入画奇航」纯决策与文本解析模块，不执行键鼠操作。"""

import re
from dataclasses import dataclass

import cv2
import numpy as np

# S2 通用主力阵容：连击那刻夏（低~中难度稳健成型）
ANAXA_FRONT = ("那刻夏", "星期日", "花火", "知更鸟", "雪衣")
ANAXA_BACK = ("刻律德菈", "白露", "寒鸦", "千冶·刃", "缇宝", "符玄", "藿藿")

# S2 通用治疗与生存护航角色（缺人保底时优先购入）
SUSTAIN_NAMES = frozenset({
    "白露", "藿藿", "风堇", "罗刹", "灵砂", "加拉赫", "娜塔莎", "玲可",
    "符玄", "砂金", "杰帕德", "丹恒·腾荒", "真珠", "长夜月",
})

# 具备特殊站位或商店规则的角色，自动买牌时避开以免拖拽卡死
SPECIAL_NAMES = frozenset({
    "白厄", "丹恒·饮月", "星神★阿哈", "星神☆阿哈", "佩佩", "姵姵", "叽米",
})

S2_ENV_PRIOR = (
    "连击概念股", "人才下沉", "青梅牛马", "银·金·彩", "头彩",
    "长线利好", "增发货币", "人身意外险",
)

BOND_PLANS = {
    "s2_anaxa": ANAXA_FRONT + ANAXA_BACK,
    "s2_dot": (
        "海瑟音", "卡芙卡", "黑天鹅", "椒丘", "希露瓦", "桂乃芬",
        "娜塔莎", "桑博", "刻律德菈", "白露", "寒鸦", "星期日", "藿藿",
    ),
    "s2_followup": (
        "飞霄", "托帕", "三月七", "知更鸟", "砂金", "缇宝",
        "不死途", "花火", "星期日", "白露",
    ),
}

FATAL_INVESTMENT_WORDS = (
    "生命值变为", "随机移除", "移除所有", "轮回不止", "无法合成",
    "只有3个备战", "节省工位", "长线空头", "奋斗协议", "钻石商人",
)
RISK_INVESTMENT_WORDS = ("失去", "扣除", "损失", "敌人难度", "出售", "替换为")
BENEFIT_INVESTMENT_WORDS = (
    "免费刷新", "额外金币", "获得金币", "简易装备", "进阶装备", "武装箱",
    "伤害提高", "伤害增幅", "回复生命值", "经验", "能量", "前景", "速度增幅",
)

HALO_RISK_WORDS = (
    "熄火", "只能造成1", "献祭", "离场", "自伤", "退回备战席",
    "同名角色不可上阵", "乱披风拳法", "优先施放普攻",
)
HALO_BENEFIT_WORDS = (
    "前台强度", "前/后台强度", "伤害增幅", "施放战技", "暴击",
    "恢复能量", "速度增幅", "额外伤害", "真实伤害", "获得金币",
)

_BASE_NAME_RE = r"[\u3400-\u9fffA-Za-z&＆·.★☆]{2,18}(?:999)?"
_UNIT_NAME_FIXES = {
    "干冶·刃": "千冶·刃",
    "刻律德": "刻律德菈",
    "刻律德拉": "刻律德菈",
    "知更鸟·晴歌": "知更鸟",
    "交光": "爻光",
    "霍霍": "藿藿",
    "霞蝶": "遐蝶",
    "托帕": "托帕&账账",
    "盾丹": "丹恒·腾荒",
    "银狼999": "银狼",
}


@dataclass(frozen=True, slots=True)
class ShopOffer:
    """商店单格商品解析结果。"""

    column: int
    name: str
    cost: int


def clean_unit_name(raw: str) -> str:
    """规范化 OCR 识别出的角色名称，修正已知固定误识别或尾字截断。"""
    text = re.sub(r"[\s「」『』【】]", "", raw or "").replace("•", "·")
    return _UNIT_NAME_FIXES.get(text, text)


def parse_shop_offers(slot_texts: list[tuple[int, str, int | None]]) -> list[ShopOffer]:
    """将商店 5 个卡槽的 OCR 文字解析为可读商品列表。

    Args:
        slot_texts: 每项为 (列序号 0~4, 该列底部名称费用拼接文本, 底栏色阶兜底费用或 None)。

    Returns:
        合法的 ShopOffer 列表（排除空槽与无法确认费用的条目）。
    """
    offers: list[ShopOffer] = []
    for column, raw_text, fallback_cost in slot_texts:
        text = clean_unit_name(raw_text)
        if not text:
            continue
        match = re.fullmatch(fr"({_BASE_NAME_RE}?)(1[0-5]|[1-9])", text)
        if match:
            name = clean_unit_name(match.group(1))
            cost = int(match.group(2))
        elif re.fullmatch(_BASE_NAME_RE, text) and fallback_cost is not None:
            name = text
            cost = fallback_cost
        else:
            continue
        if 1 <= cost <= 15:
            offers.append(ShopOffer(column=column, name=name, cost=cost))
    return offers


def decide_shop_action(
    gold: int | None,
    level: int | None,
    placed: int,
    capacity: int,
    bench_empty: int,
    offers: list[ShopOffer],
    target_names: tuple[str, ...],
    refresh_cost: int | None,
    xp_cost: int | None,
    refreshes: int,
    ops: int,
    *,
    max_refreshes: int = 2,
    max_ops: int = 8,
    target_level: int = 8,
    tracked_cols: tuple[int, ...] = (),
    guide_active: bool = False,
    bench_chars: int = 0,
) -> tuple[str, ShopOffer | None]:
    """根据当前经济与阵容空位给出一步商店动作。

    Returns:
        ("done", None) / ("sell", None) / ("buy", ShopOffer) / ("exp", None) / ("refresh", None)。
    """
    if ops >= max_ops or gold is None:
        return "done", None

    affordable = [o for o in offers if o.cost <= gold and o.name not in SPECIAL_NAMES]
    target_rank = {name: idx for idx, name in enumerate(target_names)}
    tracked_set = set(tracked_cols)

    # 1. 开启攻略追踪时按真实追踪角标购买；未开启则按计划名单购买
    target_offers = [
        o for o in affordable
        if o.column in tracked_set or (not guide_active and o.name in target_rank)
    ]
    if target_offers:
        best = min(
            target_offers,
            key=lambda o: (
                0 if o.column in tracked_set else 1,
                target_rank.get(o.name, len(target_rank)),
                o.cost,
                o.column,
            ),
        )
        if bench_empty > 0:
            return "buy", best
        return "sell", None

    # 2. 场上与备战区合计仍不足上限时才买低费/保命卡补人口，避免备战已有人还买垃圾卡
    if placed + bench_chars < capacity and bench_empty > 0 and affordable:
        filler = min(
            affordable,
            key=lambda o: (o.name not in SUSTAIN_NAMES, o.cost, o.column),
        )
        return "buy", filler

    # 3. 人口未到目标且资金允许（或当前已满员急需扩人口）时购买经验
    if xp_cost is not None and level is not None and level < target_level:
        reserve = 0 if placed >= capacity else 4
        if gold >= xp_cost + reserve:
            return "exp", None

    # 4. 本小关刷新额度未满且钱充裕时刷新寻找目标卡
    if refreshes < max_refreshes and refresh_cost is not None and gold >= refresh_cost + 3:
        return "refresh", None

    return "done", None


def score_investment_text(
    text: str,
    has_icon: bool,
    prefer_names: tuple[str, ...],
    *,
    catalog_first: bool = True,
) -> int:
    """为单列投资环境或投资策略打分，兼顾未收录图鉴优先与风险规避。"""
    cleaned = clean_unit_name(text)
    if not cleaned:
        return -500
    if any(word in cleaned for word in FATAL_INVESTMENT_WORDS):
        return -1000

    score = 0
    if catalog_first and has_icon:
        score += 500
    for idx, name in enumerate(prefer_names):
        if name in cleaned:
            score += max(20, 100 - idx * 5)
            break
    score += 12 * sum(word in cleaned for word in BENEFIT_INVESTMENT_WORDS)
    score -= 40 * sum(word in cleaned for word in RISK_INVESTMENT_WORDS)
    return score


def score_halo_text(text: str) -> int:
    """评估主角光环选项安全性与收益，得分越高越优先。"""
    cleaned = clean_unit_name(text)
    if not cleaned or any(word in cleaned for word in HALO_RISK_WORDS):
        return -1000
    score = 15 * sum(word in cleaned for word in HALO_BENEFIT_WORDS)
    if "敌人难度" in cleaned:
        match = re.search(r"敌人难度\+(\d+)", cleaned)
        score -= 25 * int(match.group(1)) if match else 50
    if "占用" in cleaned and re.search(r"占用[2-9二两三四]", cleaned):
        score -= 40
    return score


def pick_encounter_index(column_texts: list[str]) -> int:
    """从遭遇节点多列文本中选择最低难度选项索引（默认最左列 0）。"""
    digits = "一二三四五六七八九"
    best_idx = 0
    best_rank = 999
    for idx, raw in enumerate(column_texts):
        raw_str = str(raw)
        cleaned = clean_unit_name(raw_str)
        match = re.search(r"遭遇其([一二三四五六七八九1-9])", cleaned)
        if match:
            ch = match.group(1)
            rank = digits.index(ch) + 1 if ch in digits else int(ch)
        elif (
            "遭遇其-" in raw_str
            or re.search(r"遭遇其[-—－_]", raw_str)
            or ("遭遇其" in cleaned and not re.search(r"遭遇其[二三四五六七八九2-9]", cleaned))
        ):
            # RapidOCR 常把“一”识别为短横“-”或省略，无更高等级字样时按 1 阶处理
            rank = 1
        elif any(k in cleaned for k in ("普通遭遇", "普通挑战", "稳妥迎战", "离开", "跳过")):
            rank = 0
        else:
            rank = 50 + idx
        if rank < best_rank:
            best_rank = rank
            best_idx = idx
    return best_idx


def is_button_lit(patch: np.ndarray, *, luma_floor: int = 175, min_ratio: float = 0.30) -> bool:
    """判断按钮区域是否呈高亮可点击状态，防止点击灰暗不可用按钮。"""
    if patch.size == 0:
        return False
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    return float((gray >= luma_floor).mean()) >= min_ratio


ORB_TRAY_BOX = (1270, 105, 1665, 445)
EQUIPMENT_RAIL_SLOTS = (
    (1852, 245),
    (1852, 325),
    (1852, 405),
    (1852, 485),
)
PLAN_BOND_KEYWORDS = {
    "s2_anaxa": ("连击", "奇兽花园", "星间旅人"),
    "s2_dot": ("持续伤害", "奇兽花园", "仙舟舰队"),
    "s2_followup": ("追击", "仙舟舰队", "千星商会", "连击"),
    "s2_xianzhou": ("仙舟舰队", "召唤", "云上五骁", "连击"),
}
ITEM_COMPONENT_PREFS = (
    ("折刀", 45),
    ("幸运星", 40),
    ("和平手枪", 35),
    ("轮滑鞋", 30),
    ("光能电池", 25),
)


def detect_orb_points(frame: np.ndarray, *, limit: int = 12) -> list[tuple[int, int]]:
    """检测右上角战利品晶球托盘内可见的蓝/银/金色晶球中心坐标列表（1920x1080）。"""
    if frame is None or frame.ndim != 3 or frame.shape[0] < ORB_TRAY_BOX[3] or frame.shape[1] < ORB_TRAY_BOX[2]:
        return []
    x1, y1, x2, y2 = ORB_TRAY_BOX
    tray = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(tray, cv2.COLOR_BGR2HSV)
    b = tray[:, :, 0].astype(np.int16)
    g = tray[:, :, 1].astype(np.int16)
    blue = ((b >= 210) & (g >= 100)).astype(np.uint8)
    # 银白阈值抬高并压低饱和度，避免装备追踪速合悬浮窗的灰白「✔」圆钮误报为晶球
    silver = ((hsv[:, :, 2] >= 195) & (hsv[:, :, 1] <= 55)).astype(np.uint8)
    # 同步识别黄金高阶晶球（橙黄高亮球体）
    gold = ((hsv[:, :, 0] >= 10) & (hsv[:, :, 0] <= 34) & (hsv[:, :, 1] >= 110) & (hsv[:, :, 2] >= 165)).astype(np.uint8)
    mask = ((blue | silver | gold) * 255).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8))
    if not np.any(mask):
        return []
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    points: list[tuple[int, int]] = []
    work = dist.copy()
    while len(points) < limit:
        _, max_val, _, max_pt = cv2.minMaxLoc(work)
        if max_val < 5.0:
            break
        points.append((int(max_pt[0] + x1), int(max_pt[1] + y1)))
        cv2.circle(work, max_pt, 48, 0, -1)
    return points


def has_guide_badge(patch: np.ndarray, *, min_ratio: float = 0.08) -> bool:
    """判断商店/备战卡片左上角是否带有一键攻略追踪的橙金色角标。"""
    if patch is None or patch.size == 0 or patch.ndim != 3:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 0] >= 10) & (hsv[:, :, 0] <= 28) & (hsv[:, :, 1] >= 120) & (hsv[:, :, 2] >= 150)
    return float(mask.mean()) >= min_ratio


def detect_shop_guide_badges(frame: np.ndarray) -> tuple[bool, ...]:
    """检测商店 5 个槽位左上角是否有推荐攻略追踪角标。"""
    if frame is None or frame.ndim != 3 or frame.shape[0] < 115 or frame.shape[1] < 1700:
        return (False,) * 5
    return tuple(
        has_guide_badge(frame[70:115, 367 + 269 * col + 4:367 + 269 * col + 48])
        for col in range(5)
    )


def detect_bench_guide_badges(frame: np.ndarray) -> tuple[bool, ...]:
    """检测备战栏 9 个槽位左上角是否有推荐攻略追踪角标。"""
    if frame is None or frame.ndim != 3 or frame.shape[0] < 875 or frame.shape[1] < 1500:
        return (False,) * 9
    return tuple(
        has_guide_badge(frame[845:875, 383 + 125 * idx + 2:383 + 125 * idx + 32])
        for idx in range(9)
    )


def detect_loose_equipment_slots(frame: np.ndarray) -> list[tuple[int, int]]:
    """检测右侧装备栏（跳过顶部扳手/熔炉工具行）中仍待穿戴/合成的装备格坐标。"""
    if frame is None or frame.ndim != 3 or frame.shape[0] < 520 or frame.shape[1] < 1890:
        return []
    occupied: list[tuple[int, int]] = []
    for cx, cy in EQUIPMENT_RAIL_SLOTS:
        patch = frame[cy - 25:cy + 25, cx - 27:cx + 28]
        if patch.size == 0:
            continue
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        if int(patch.max()) >= 165 and float(gray.std()) >= 24.0:
            occupied.append((cx, cy))
        elif not occupied:
            # 装备格自上而下紧缩排列，首格为空说明暂无可穿戴装备
            break
    return occupied


def score_item_choice(raw_text: str, plan_id: str) -> int:
    """为武装箱/星徽秘典的备选物品打分，优先匹配当前阵容羁绊星徽与高泛用散件。"""
    cleaned = clean_unit_name(raw_text)
    if not cleaned:
        return -100
    score = 0
    if "星徽" in cleaned:
        bonds = PLAN_BOND_KEYWORDS.get(plan_id, ())
        score += 200 if any(b in cleaned for b in bonds) else 60
    for keyword, bonus in ITEM_COMPONENT_PREFS:
        if keyword in cleaned:
            score += bonus
            break
    return score
