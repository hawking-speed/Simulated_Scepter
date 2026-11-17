"""货币战争 S2「入画奇航」全量装备、合成路径与配装拆装决策库。"""

from dataclasses import dataclass

# S2 8 件简易装备散件及保留定位：keep_core=True 时单件尽量留作主C毕业大件合成。
BASIC_ITEMS: dict[str, dict[str, object]] = {
    "轮滑鞋": {"tags": ("speed", "carry"), "keep_core": True},
    "折叠小刀": {"tags": ("damage", "carry"), "keep_core": True},
    "和平手枪": {"tags": ("support", "carry"), "keep_core": False},
    "幸运星": {"tags": ("crit", "carry"), "keep_core": False},
    "生命之花": {"tags": ("sustain", "hp"), "keep_core": False},
    "量产型装甲": {"tags": ("defense", "sustain"), "keep_core": False},
    "以太钻头": {"tags": ("pierce", "carry"), "keep_core": True},
    "光能电池": {"tags": ("energy", "support"), "keep_core": True},
}

WRENCH_NAMES = ("精密拆装扳手", "拆装扳手")
FURNACE_NAME = "重铸熔炉"

# 主 C 不可盲塞纯防御肉装的标签集合
PURE_DEFENSE_TAGS = frozenset({"defense"})


@dataclass(frozen=True, slots=True)
class EquipmentSpec:
    """进阶成装或特殊装备的属性与定位权重。"""

    name: str
    role: str  # carry_speed / carry_dmg / support / sustain / emblem / tool
    carry_score: int
    support_score: int
    sustain_score: int
    unique_unit: bool = False
    is_emblem: bool = False


def _spec(
    name: str,
    role: str,
    carry: int,
    support: int,
    sustain: int,
    *,
    unique: bool = False,
    emblem: bool = False,
) -> EquipmentSpec:
    return EquipmentSpec(
        name=name,
        role=role,
        carry_score=carry,
        support_score=support,
        sustain_score=sustain,
        unique_unit=unique,
        is_emblem=emblem,
    )


# 完整 36 条两两合成路径（8×9/2，按 S2「入画奇航」正式名称，不分先后）。
SYNTHESIS_TABLE: dict[tuple[str, str], EquipmentSpec] = {
    # 轮滑鞋系（8）
    ("轮滑鞋", "轮滑鞋"): _spec("反重力皮靴", "carry_speed", 130, 55, 45),
    ("轮滑鞋", "以太钻头"): _spec("光速螺旋桨", "carry_speed", 128, 60, 45),
    ("轮滑鞋", "折叠小刀"): _spec("火力风暴潮", "carry_dmg", 124, 40, 25),
    ("轮滑鞋", "光能电池"): _spec("星航履", "carry_speed", 118, 92, 60),
    ("轮滑鞋", "生命之花"): _spec("反螺旋舞步", "carry_dmg", 116, 60, 75),
    ("轮滑鞋", "和平手枪"): _spec("电磁弹射器", "support", 72, 114, 72, unique=True),
    ("轮滑鞋", "幸运星"): _spec("追逐星辰", "carry_dmg", 108, 62, 35),
    ("轮滑鞋", "量产型装甲"): _spec("流星飞翼", "sustain", 30, 62, 92),
    # 折叠小刀系（7）
    ("折叠小刀", "幸运星"): _spec("高周波电锯", "carry_dmg", 118, 30, 20),
    ("折叠小刀", "折叠小刀"): _spec("碎星斩舰刀", "carry_dmg", 112, 25, 15),
    ("折叠小刀", "和平手枪"): _spec("武器大师", "carry_dmg", 96, 82, 40),
    ("折叠小刀", "以太钻头"): _spec("动能激发剑", "carry_dmg", 104, 35, 25),
    ("折叠小刀", "光能电池"): _spec("战场进化手册", "support", 82, 92, 55),
    ("折叠小刀", "生命之花"): _spec("杀红眼", "carry_dmg", 90, 25, 45),
    ("折叠小刀", "量产型装甲"): _spec("信心注入器", "support", 35, 86, 84, unique=True),
    # 和平手枪系（6）
    ("和平手枪", "光能电池"): _spec("蓄能帆", "support", 95, 118, 82, unique=True),
    ("和平手枪", "幸运星"): _spec("反卫星狙击枪", "carry_dmg", 106, 40, 25),
    ("和平手枪", "和平手枪"): _spec("天基轨道炮", "carry_dmg", 98, 45, 20),
    ("和平手枪", "以太钻头"): _spec("斩首行动", "carry_dmg", 100, 35, 20),
    ("和平手枪", "生命之花"): _spec("掩体生成枪", "sustain", 20, 68, 108, unique=True),
    ("和平手枪", "量产型装甲"): _spec("自适应外骨骼", "sustain", 15, 58, 98),
    # 幸运星系（5）
    ("幸运星", "以太钻头"): _spec("胜利之旗", "support", 76, 98, 72, unique=True),
    ("幸运星", "光能电池"): _spec("冷笑话引擎", "support", 68, 90, 50),
    ("幸运星", "生命之花"): _spec("热血沸腾拳", "carry_dmg", 86, 42, 65),
    ("幸运星", "幸运星"): _spec("随便骰子", "support", 60, 75, 65),
    ("幸运星", "量产型装甲"): _spec("以牙还牙甲", "sustain", 15, 45, 94),
    # 生命之花系（4）
    ("生命之花", "光能电池"): _spec("绝对热量", "sustain", 50, 104, 128, unique=True),
    ("生命之花", "生命之花"): _spec("生命之环", "sustain", 15, 70, 110, unique=True),
    ("生命之花", "量产型装甲"): _spec("痛觉阻断芯片", "sustain", 10, 66, 112, unique=True),
    ("生命之花", "以太钻头"): _spec("物质分解液", "support", 45, 78, 82),
    # 量产型装甲系（3）
    ("量产型装甲", "量产型装甲"): _spec("很硬的甲", "sustain", 10, 55, 118, unique=True),
    ("量产型装甲", "以太钻头"): _spec("闪光手榴弹", "sustain", 15, 72, 106, unique=True),
    ("量产型装甲", "光能电池"): _spec("光能盾牌", "sustain", 25, 74, 102),
    # 以太钻头系（2）
    ("以太钻头", "光能电池"): _spec("行星钻地弹", "support", 72, 96, 58, unique=True),
    ("以太钻头", "以太钻头"): _spec("虫洞掘进钻头", "carry_dmg", 92, 48, 35),
    # 光能电池系（1）
    ("光能电池", "光能电池"): _spec("永动机", "support", 88, 126, 86, unique=True),
}

SYNTHESIS_RECIPES: dict[frozenset[str], EquipmentSpec] = {
    frozenset(pair): spec for pair, spec in SYNTHESIS_TABLE.items()
}

# 按散件名建立无序查表（含同件双拼，避免 frozenset 丢掉计数语义）
_RECIPES_BY_PAIR: dict[tuple[str, str], EquipmentSpec] = {
    tuple(sorted(pair)): spec for pair, spec in SYNTHESIS_TABLE.items()
}

SPECIAL_ITEMS: dict[str, EquipmentSpec] = {
    "连击星徽": _spec("连击星徽", "emblem", 40, 102, 95, emblem=True),
    "奇兽花园星徽": _spec("奇兽花园星徽", "emblem", 40, 105, 95, emblem=True),
    "昼之半神星徽": _spec("昼之半神星徽", "emblem", 35, 88, 80, emblem=True),
    "仙舟星徽": _spec("仙舟星徽", "emblem", 25, 72, 72, emblem=True),
    "能量星徽": _spec("能量星徽", "emblem", 35, 86, 78, emblem=True),
    "欢愉星徽": _spec("欢愉星徽", "emblem", 20, 68, 68, emblem=True),
    "战技点星徽": _spec("战技点星徽", "emblem", 30, 82, 74, emblem=True),
    "群攻星徽": _spec("群攻星徽", "emblem", 20, 68, 68, emblem=True),
    "减益星徽": _spec("减益星徽", "emblem", 20, 66, 68, emblem=True),
    "持续伤害星徽": _spec("持续伤害星徽", "emblem", 15, 62, 62, emblem=True),
    "击破星徽": _spec("击破星徽", "emblem", 15, 62, 62, emblem=True),
    "燃血星徽": _spec("燃血星徽", "emblem", 15, 62, 62, emblem=True),
    "夜之半神星徽": _spec("夜之半神星徽", "emblem", 20, 66, 66, emblem=True),
    "盛会之星星徽": _spec("盛会之星星徽", "emblem", 20, 66, 66, emblem=True),
    "狼狩星徽": _spec("狼狩星徽", "emblem", 20, 64, 64, emblem=True),
    "贝洛伯格星徽": _spec("贝洛伯格星徽", "emblem", 20, 64, 64, emblem=True),
    "星间旅人星徽": _spec("星间旅人星徽", "emblem", 30, 84, 78, emblem=True),
    "银河学者星徽": _spec("银河学者星徽", "emblem", 25, 74, 72, emblem=True),
    "财富宝钻": _spec("财富宝钻", "support", 50, 96, 85),
    "红钻": _spec("红钻", "support", 40, 76, 70),
    "蓝钻": _spec("蓝钻", "support", 40, 76, 70),
    "白昼·光速螺旋桨": _spec("白昼·光速螺旋桨", "carry_speed", 132, 60, 45),
}

ITEM_SPECS_BY_NAME: dict[str, EquipmentSpec] = {
    **{spec.name: spec for spec in SYNTHESIS_TABLE.values()},
    **SPECIAL_ITEMS,
}

ALL_KNOWN_ITEM_NAMES: tuple[str, ...] = tuple(
    dict.fromkeys(
        list(BASIC_ITEMS)
        + list(ITEM_SPECS_BY_NAME)
        + list(WRENCH_NAMES)
        + [FURNACE_NAME]
    )
)

# 连击阿格莱雅/那刻夏阵容角色定位与出装角色映射
CARRY_UNITS = frozenset({"阿格莱雅", "那刻夏"})
HARMONY_SUPPORT_UNITS = frozenset({"刻律德菈", "星期日", "花火", "知更鸟", "寒鸦", "缇宝"})
SUSTAIN_UNITS = frozenset({"风堇", "白露", "藿藿", "符玄", "砂金", "丹恒·腾荒", "灵砂", "罗刹", "加拉赫", "娜塔莎", "玲可", "杰帕德"})
BRUISER_FRONT_UNITS = frozenset({"雪衣", "千冶·刃", "云璃", "瓦尔特", "镜流", "飞霄"})

# 需要通过拆装扳手回收转移给 5 费阿格莱雅的主C核心大件
CORE_CARRY_GEAR = frozenset({
    "反重力皮靴",
    "光速螺旋桨",
    "白昼·光速螺旋桨",
    "火力风暴潮",
    "星航履",
    "反螺旋舞步",
    "高周波电锯",
    "碎星斩舰刀",
    "反卫星狙击枪",
})


@dataclass(frozen=True, slots=True)
class UnitEquipSnapshot:
    """单个在场单位参与装备规划的轻量快照。"""

    zone: str
    index: int
    name: str
    stars: int
    worn: tuple[str, ...]
    equip_count: int
    has_halo: bool = False


@dataclass(frozen=True, slots=True)
class EquipAction:
    """一步可执行或待合成的装备分配方案。"""

    score: int
    zone: str
    index: int
    unit_name: str
    item_name: str
    kind: str  # panel_craft / panel_wear / direct_drag / direct_craft
    ingredients: tuple[str, ...] = ()
    rail_pos: tuple[int, int] | None = None
    second_rail_pos: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class WrenchPlan:
    """拆装扳手拆卸建议。"""

    source_zone: str
    source_index: int
    source_name: str
    target_name: str
    reason: str


def lookup_recipe(part_a: str, part_b: str) -> EquipmentSpec | None:
    """查询两件简易装备的正式合成产物。"""
    key = tuple(sorted((part_a.strip(), part_b.strip())))
    return _RECIPES_BY_PAIR.get(key)  # type: ignore[arg-type]


def all_recipes_using(part: str) -> list[tuple[str, EquipmentSpec]]:
    """列出某个散件可参与合成的全部成装，按主C与辅助价值降序排列。"""
    results: list[tuple[str, EquipmentSpec]] = []
    for (a, b), spec in _RECIPES_BY_PAIR.items():
        if a == part:
            results.append((b, spec))
        elif b == part:
            results.append((a, spec))
    results.sort(key=lambda item: max(item[1].carry_score, item[1].support_score, item[1].sustain_score), reverse=True)
    return results


def pairable_recipes(inventory_names: list[str]) -> list[tuple[str, str, EquipmentSpec]]:
    """从当前右栏装备列表找出所有两两可合成的进阶大件。"""
    counts: dict[str, int] = {}
    for name in inventory_names:
        if name in BASIC_ITEMS:
            counts[name] = counts.get(name, 0) + 1
    found: list[tuple[str, str, EquipmentSpec]] = []
    seen: set[tuple[str, str]] = set()
    for (a, b), spec in _RECIPES_BY_PAIR.items():
        if (a, b) in seen:
            continue
        if a == b and counts.get(a, 0) >= 2:
            seen.add((a, b))
            found.append((a, b, spec))
        elif a != b and counts.get(a, 0) >= 1 and counts.get(b, 0) >= 1:
            seen.add((a, b))
            found.append((a, b, spec))
    found.sort(key=lambda it: max(it[2].carry_score, it[2].support_score, it[2].sustain_score), reverse=True)
    return found


def classify_unit_role(name: str, zone: str) -> str:
    """判定角色在连击阿格莱雅/那刻夏体系中的配装职责。"""
    if name in CARRY_UNITS:
        return "carry"
    if name in HARMONY_SUPPORT_UNITS:
        return "support"
    if name in SUSTAIN_UNITS:
        return "sustain"
    if zone == "front" or name in BRUISER_FRONT_UNITS:
        return "bruiser"
    return "support"


def item_spec(name: str) -> EquipmentSpec | None:
    """获取已知成装/星徽规格；8 件散件与未知名称返回 None。"""
    return ITEM_SPECS_BY_NAME.get(name)


def score_item_for_unit(
    unit_name: str,
    zone: str,
    stars: int,
    worn: tuple[str, ...],
    equip_count: int,
    item_name: str,
    *,
    hp: int | None = None,
    is_final: bool = False,
    has_aglaea: bool = False,
    tracked_prefs: tuple[str, ...] = (),
) -> int:
    """评估某件装备给指定格位角色的收益分。

    约束：
        - 已穿满 3 件装备（``equip_count >= 3`` 或 ``len(worn) >= 3``）必返回 ``-999``。
        - 纯防御肉装严禁给主 C（阿格莱雅/那刻夏）穿戴，除非终局低血无治疗空位。
    """
    used_slots = max(equip_count, len(worn))
    if used_slots >= 3 or not item_name:
        return -999

    role = classify_unit_role(unit_name, zone)
    low_hp = hp is not None and hp <= 55

    # 单件简易散件本身不直接乱穿；主 C 散件尽量等合成，绝不塞防御散件给主 C
    if item_name in BASIC_ITEMS:
        if role == "carry":
            if item_name in ("量产型装甲", "生命之花"):
                return 0
            return 28 if is_final else 10
        if role == "sustain":
            return 35 if is_final and item_name in ("量产型装甲", "生命之花", "光能电池") else 5
        return 22 if is_final else 4

    spec = item_spec(item_name)
    if spec is None:
        return 25 if zone == "front" else 20

    if spec.unique_unit and item_name in worn:
        # 唯一光环或叠穿收益归零的大件不重复堆给同一名角色
        return 5

    # 护盾/防御装严禁配给阿格莱雅/那刻夏主 C
    if role == "carry" and spec.role == "sustain":
        if not (is_final and hp is not None and hp <= 40 and used_slots == 2):
            return 0

    if spec.is_emblem:
        # 星徽给带空槽的非毕业主 C 辅助或副 C 凑 6 连击/3 奇兽，避免占用主 C 毕业槽
        if unit_name == "阿格莱雅" and used_slots < 2:
            return 24
        return spec.support_score + (18 if "连击" in item_name or "奇兽花园" in item_name else 0)

    if unit_name == "阿格莱雅":
        base = spec.carry_score + (35 if has_aglaea else 25)
    elif unit_name == "那刻夏":
        # 阿格莱雅尚未成型前，那刻夏吃全部主 C 毕业装；阿格莱雅已在场则适当让出第一优先给阿格莱雅
        base = spec.carry_score + (8 if has_aglaea else 28)
    elif role == "support":
        base = spec.support_score + (10 if zone == "back" else 6)
    elif role == "sustain":
        base = spec.sustain_score + (22 if low_hp and spec.role == "sustain" else 8)
    else:
        base = max(spec.sustain_score, spec.carry_score // 2 + 25)

    # 星级越高的核心角色优先持有关键成装，避免塞给 1 星过渡卡
    star_bonus = max(0, stars - 1) * 10
    pref_bonus = 24 if item_name in tracked_prefs else 0
    return base + star_bonus + pref_bonus


def _preserve_penalty(
    part_a: str,
    part_b: str | None,
    inventory_names: list[str],
    *,
    is_final: bool,
    has_carry_speed: bool,
) -> int:
    """计算拆散核心散件的机会成本：保护阿格莱雅/那刻夏反重力皮靴与永动机材料。"""
    if is_final:
        return 0
    penalty = 0
    for part in (part_a, part_b):
        if not part:
            continue
        if part == "轮滑鞋" and not has_carry_speed:
            # 只有 1 只轮滑鞋而合非毕业速度装时扣分，等第 2 只拼反重力皮靴
            if part_a != part_b and inventory_names.count("轮滑鞋") == 1:
                spec = lookup_recipe(part_a, part_b or "")
                if spec is None or spec.name not in ("反重力皮靴", "光速螺旋桨", "火力风暴潮", "星航履"):
                    penalty += 38
        elif part == "光能电池" and part_a != part_b and inventory_names.count("光能电池") == 1:
            spec = lookup_recipe(part_a, part_b or "")
            if spec is None or spec.name not in ("绝对热量", "永动机", "蓄能帆", "星航履"):
                penalty += 26
    return penalty


def plan_best_equip_action(
    units: list[UnitEquipSnapshot],
    rail_items: list[tuple[tuple[int, int], str | None]],
    *,
    hp: int | None = None,
    is_final: bool = False,
    tracked_prefs: dict[str, tuple[str, ...]] | None = None,
) -> EquipAction | None:
    """全局评估全队前后排已穿槽位与右栏材料，给出唯一最佳合成/直穿动作。

    若当前只有低价值单件散件且尚未到终局，返回 ``None`` 保留散件等下一波配对。
    """
    eligible = [u for u in units if u.equip_count < 3 and len(u.worn) < 3 and u.name]
    if not eligible or not rail_items:
        return None

    tracked_map = tracked_prefs or {}
    has_aglaea = any(u.name == "阿格莱雅" for u in units)
    has_carry_speed = any(
        u.name in CARRY_UNITS and bool(set(u.worn) & {"反重力皮靴", "光速螺旋桨", "白昼·光速螺旋桨"})
        for u in units
    )
    known_names = [name for _, name in rail_items if name]
    candidates: list[EquipAction] = []

    # 1. 同右栏两两直接拖拽合成最佳进阶装并直接给最匹配目标
    indexed_known = [(pos, name) for pos, name in rail_items if name in BASIC_ITEMS]
    for i in range(len(indexed_known)):
        for j in range(i + 1, len(indexed_known)):
            pos_a, name_a = indexed_known[i]
            pos_b, name_b = indexed_known[j]
            spec = lookup_recipe(name_a, name_b)
            if spec is None:
                continue
            cost_penalty = _preserve_penalty(
                name_a, name_b, known_names, is_final=is_final, has_carry_speed=has_carry_speed
            )
            for unit in eligible:
                sc = score_item_for_unit(
                    unit.name, unit.zone, unit.stars, unit.worn, unit.equip_count,
                    spec.name, hp=hp, is_final=is_final, has_aglaea=has_aglaea,
                    tracked_prefs=tracked_map.get(unit.name, ()),
                ) - cost_penalty
                if sc >= 45:
                    candidates.append(
                        EquipAction(
                            score=sc + 12,
                            zone=unit.zone,
                            index=unit.index,
                            unit_name=unit.name,
                            item_name=spec.name,
                            kind="direct_craft",
                            ingredients=(name_a, name_b),
                            rail_pos=pos_a,
                            second_rail_pos=pos_b,
                        )
                    )

    # 2. 右栏已有成品进阶装 / 星徽 / 宝钻（或终局实在没得合的散件）直拖给最适配未满格角色
    for pos, name in rail_items:
        if not name:
            continue
        is_basic = name in BASIC_ITEMS
        if is_basic and not is_final:
            # 非终局绝不把未合成散件盲拖到身上占掉格子或浪费配方
            continue
        for unit in eligible:
            sc = score_item_for_unit(
                unit.name, unit.zone, unit.stars, unit.worn, unit.equip_count,
                name, hp=hp, is_final=is_final, has_aglaea=has_aglaea,
                tracked_prefs=tracked_map.get(unit.name, ()),
            )
            min_gate = 20 if is_final else 45
            if sc >= min_gate:
                candidates.append(
                    EquipAction(
                        score=sc + (6 if not is_basic else 0),
                        zone=unit.zone,
                        index=unit.index,
                        unit_name=unit.name,
                        item_name=name,
                        kind="direct_drag",
                        rail_pos=pos,
                    )
                )

    if not candidates:
        return None
    return max(candidates, key=lambda act: (act.score, act.zone == "front"))


def plan_wrench_transfer(
    units: list[UnitEquipSnapshot],
    *,
    has_wrench: bool,
    has_risky_emblem_reset: bool = False,
) -> WrenchPlan | None:
    """判断是否应使用拆装扳手回收毕业装：前期打工 -> 5费阿格莱雅，或低星同卡让装。"""
    if not has_wrench or not units:
        return None

    aglaea = next((u for u in units if u.name == "阿格莱雅" and u.equip_count < 3 and len(u.worn) < 3), None)
    if aglaea is None:
        return None

    # 从非阿格莱雅在场单位身上查找持有核心输出毕业装者（阿格莱雅到来前由那刻夏/打工C代穿）
    for src in units:
        if src.name == "阿格莱雅" or src.equip_count <= 0:
            continue
        worn_set = set(src.worn)
        # 若环境可能改变星徽属性且仅穿星徽，保守不拆星徽
        if has_risky_emblem_reset and worn_set and all("星徽" in w for w in worn_set):
            continue
        if worn_set & CORE_CARRY_GEAR:
            # 对 2 星以上那刻夏仅在其拿有多件或阿格莱雅尚缺启动鞋时拆转，保证双核战力平滑
            if src.name == "那刻夏" and src.stars >= 3 and aglaea.equip_count >= 2:
                continue
            return WrenchPlan(
                source_zone=src.zone,
                source_index=src.index,
                source_name=src.name,
                target_name="阿格莱雅",
                reason="移交主C核心装备给阿格莱雅",
            )
    return None
