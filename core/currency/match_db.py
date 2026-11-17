"""货币战争 S2 动态对局记忆数据库：随对局发展实时记录棋盘、仓库、人口、经济、连胜连败与装备状态。"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from core.currency.equipment_db import UnitEquipSnapshot
from tool.storage import config_path, write_data

DEFAULT_MATCH_DB_PATH = config_path(__file__).with_name("currency_match_db.json")


def _slot_key(zone: str, index: int) -> str:
    return f"{zone}:{index}"


_TEMPLATE_NAME_ALIAS = {
    "刻律德拉": "刻律德菈",
    "霍霍": "藿藿",
    "霞蝶": "遐蝶",
    "托帕": "托帕&账账",
    "盾丹": "丹恒·腾荒",
    "银狼999": "银狼",
}


def _infer_template_name(tpl: str) -> str:
    if not tpl or not tpl.startswith("character_"):
        return ""
    suffix = tpl.removeprefix("character_")
    if any("\u4e00" <= ch <= "\u9fff" for ch in suffix):
        return _TEMPLATE_NAME_ALIAS.get(suffix, suffix)
    return ""


@dataclass
class SlotRecord:
    """单个棋盘格或备战格的完整实时状态。"""

    zone: str  # front / back / bench
    index: int
    occupied: bool = False
    blocked: bool = False
    template: str = ""
    name: str = ""
    stars: int = 1
    cost: int | None = None
    can_front: bool = True
    can_back: bool = True
    is_tracked: bool = False
    has_halo: bool = False
    equip_count: int = 0
    worn_items: list[str] = field(default_factory=list)
    detail_checked_stage: str = ""

    def clear_unit(self) -> None:
        """清空该格上的角色与穿戴记录，仅保留站位位置信息。"""
        self.occupied = False
        self.template = ""
        self.name = ""
        self.stars = 1
        self.cost = None
        self.can_front = True
        self.can_back = True
        self.is_tracked = False
        self.has_halo = False
        self.equip_count = 0
        self.worn_items = []
        self.detail_checked_stage = ""

    def copy_unit_from(self, other: "SlotRecord") -> None:
        """将另一格的角色身份、星级、光环与穿戴装备迁移到当前格。"""
        self.occupied = True
        self.template = other.template
        self.name = other.name
        self.stars = other.stars
        self.cost = other.cost
        self.can_front = other.can_front
        self.can_back = other.can_back
        self.is_tracked = other.is_tracked
        self.has_halo = other.has_halo
        self.equip_count = other.equip_count
        self.worn_items = list(other.worn_items)
        self.detail_checked_stage = other.detail_checked_stage


@dataclass
class InventoryEntry:
    """右侧装备栏单格识别记录。"""

    x: int
    y: int
    name: str | None = None


class MatchDatabase:
    """单局动态状态数据库（支持内存高频读取与原子落盘）。"""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.path = Path(db_path) if db_path is not None else Path(DEFAULT_MATCH_DB_PATH)
        self.rank_text: str = ""
        self.stage: str = "1-1"
        self.plane: int = 1
        self.node: int = 1
        self.gold: int | None = None
        self.level: int | None = None
        self.exp_cur: int | None = None
        self.exp_req: int | None = None
        self.placed: int = 0
        self.capacity: int = 3
        self.hp: int | None = None
        self.last_combat_hp: int | None = None
        self.win_streak: int = 0
        self.loss_streak: int = 0
        self.guide_applied: bool = False
        self.risky_emblem_reset: bool = False
        self.slots: dict[str, SlotRecord] = {}
        self.inventory: list[InventoryEntry] = []
        self.wrench_pos: tuple[int, int] | None = None
        self.furnace_pos: tuple[int, int] | None = None
        self.pending_purchases: list[dict[str, object]] = []
        self.equip_tracking_prefs: dict[str, list[str]] = {}
        self.template_names: dict[str, str] = {}
        self._init_empty_slots()

    def _init_empty_slots(self) -> None:
        self.slots = {}
        for zone, count in (("front", 4), ("back", 9), ("bench", 9)):
            for idx in range(count):
                self.slots[_slot_key(zone, idx)] = SlotRecord(zone=zone, index=idx)

    def reset_run(self, rank_text: str = "") -> None:
        """开启新对局时重置整盘记忆数据库，保留已积累的模板到真名映射缓存。"""
        saved_names = dict(self.template_names)
        self.rank_text = rank_text
        self.stage = "1-1"
        self.plane = 1
        self.node = 1
        self.gold = None
        self.level = 3
        self.exp_cur = 0
        self.exp_req = 4
        self.placed = 0
        self.capacity = 3
        self.hp = 100
        self.last_combat_hp = 100
        self.win_streak = 0
        self.loss_streak = 0
        self.guide_applied = False
        self.risky_emblem_reset = False
        self.inventory = []
        self.wrench_pos = None
        self.furnace_pos = None
        self.pending_purchases = []
        self.equip_tracking_prefs = {}
        self._init_empty_slots()
        self.template_names = saved_names
        self.save()

    def slot(self, zone: str, index: int) -> SlotRecord:
        key = _slot_key(zone, index)
        rec = self.slots.get(key)
        if rec is None:
            rec = SlotRecord(zone=zone, index=index)
            self.slots[key] = rec
        return rec

    def iter_zone(self, zone: str) -> list[SlotRecord]:
        return sorted(
            (r for r in self.slots.values() if r.zone == zone),
            key=lambda r: r.index,
        )

    def field_records(self) -> list[SlotRecord]:
        return [r for r in self.slots.values() if r.zone in ("front", "back") and r.occupied and not r.blocked]

    def bench_records(self) -> list[SlotRecord]:
        return [r for r in self.iter_zone("bench") if not r.blocked]

    def sync_economy(
        self,
        *,
        stage: str | None = None,
        gold: int | None = None,
        level: int | None = None,
        exp: tuple[int, int] | None = None,
        counts: tuple[int, int] | None = None,
        hp: int | None = None,
    ) -> None:
        """同步游戏屏幕读到的金钱、等级、经验条、人口与小队生命。"""
        if stage and stage != "0-0":
            self.stage = stage
            parts = stage.split("-", 1)
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                self.plane, self.node = int(parts[0]), int(parts[1])
        if gold is not None:
            self.gold = gold
        if level is not None:
            self.level = level
        if exp is not None:
            self.exp_cur, self.exp_req = exp
        if counts is not None:
            placed, cap = counts
            if cap > 1 or (placed == 0 and cap == 1):
                self.placed, self.capacity = placed, cap
        if hp is not None and 0 <= hp <= 200:
            self.hp = hp

    def sync_bench(
        self,
        bench_slots,
        *,
        badges: tuple[bool, ...] | None = None,
        stars_map: dict[int, int | None] | None = None,
        equip_map: dict[int, int] | None = None,
    ) -> None:
        """按当前备战区扫描结果动态增量更新：新进角色自动继承刚购买记录。"""
        seen_indices: set[int] = set()
        for slot in bench_slots:
            idx = slot.index
            seen_indices.add(idx)
            rec = self.slot("bench", idx)
            was_occupied = rec.occupied
            prev_tpl = rec.template

            if slot.kind != "character":
                rec.clear_unit()
                continue

            rec.occupied = True
            tpl = getattr(slot, "template", "") or ""
            rec.template = tpl
            areas = tuple(getattr(slot, "areas", ("front", "back")))
            rec.can_front = "front" in areas
            rec.can_back = "back" in areas
            if badges and idx < len(badges):
                rec.is_tracked = bool(badges[idx])
            if stars_map and stars_map.get(idx) is not None:
                rec.stars = int(stars_map[idx])  # type: ignore[arg-type]
            if equip_map and idx in equip_map:
                rec.equip_count = max(0, min(3, int(equip_map[idx])))

            # 同一张已知模板持续占用此格时不覆盖已学到的真名与装备
            if tpl and tpl in self.template_names and not rec.name:
                rec.name = self.template_names[tpl]
            elif tpl and not rec.name:
                inferred = _infer_template_name(tpl)
                if inferred:
                    rec.name = inferred
                    self.template_names[tpl] = inferred

            if not was_occupied or (prev_tpl and tpl and prev_tpl != tpl):
                rec.worn_items = []
                rec.has_halo = False
                if self.pending_purchases:
                    purchase = self.pending_purchases.pop(0)
                    name = str(purchase.get("name", ""))
                    if name:
                        rec.name = name
                        if tpl:
                            self.template_names[tpl] = name
                    if purchase.get("cost") is not None:
                        rec.cost = int(purchase["cost"])  # type: ignore[arg-type]
                    rec.stars = int(purchase.get("stars", 1))
                    if purchase.get("is_tracked"):
                        rec.is_tracked = True
                    rec.can_front = bool(purchase.get("can_front", rec.can_front))
                    rec.can_back = bool(purchase.get("can_back", rec.can_back))
                elif tpl and tpl in self.template_names:
                    rec.name = self.template_names[tpl]

        for idx in range(9):
            if idx not in seen_indices:
                self.slot("bench", idx).clear_unit()

    def sync_field(
        self,
        vacancies: dict[str, list[int]],
        members: dict[str, list[str | None]] | None = None,
        *,
        active_back_slots: int = 6,
        stars_map: dict[tuple[str, int], int | None] | None = None,
        equip_map: dict[tuple[str, int], int] | None = None,
        halo_slot: tuple[str, int] | None = None,
    ) -> None:
        """同步前后台所有格位占用状态、模板名、星级、装备数与光环宿主。"""
        zone_limits = {"front": 4, "back": max(6, min(9, active_back_slots))}
        for zone, limit in zone_limits.items():
            vac = set(vacancies.get(zone, ()))
            raw_members = members.get(zone, []) if isinstance(members, dict) else []
            zone_members = raw_members if isinstance(raw_members, (list, tuple)) else []
            for idx in range(9):
                rec = self.slot(zone, idx)
                if idx >= limit:
                    rec.clear_unit()
                    rec.blocked = True
                    continue
                rec.blocked = False
                if idx in vac:
                    rec.clear_unit()
                    continue
                rec.occupied = True
                tpl = zone_members[idx] if idx < len(zone_members) and isinstance(zone_members[idx], str) else ""
                if tpl:
                    rec.template = tpl
                    if tpl in self.template_names and not rec.name:
                        rec.name = self.template_names[tpl]
                    elif not rec.name:
                        inferred = _infer_template_name(tpl)
                        if inferred:
                            rec.name = inferred
                            self.template_names[tpl] = inferred
                if stars_map and stars_map.get((zone, idx)) is not None:
                    rec.stars = int(stars_map[(zone, idx)])  # type: ignore[arg-type]
                if equip_map and (zone, idx) in equip_map:
                    rec.equip_count = max(0, min(3, int(equip_map[(zone, idx)])))
                if halo_slot is not None:
                    rec.has_halo = (zone, idx) == halo_slot

    def sync_inventory(
        self,
        rail_items: list[tuple[tuple[int, int], str | None]],
        *,
        wrench_pos: tuple[int, int] | None = None,
        furnace_pos: tuple[int, int] | None = None,
    ) -> None:
        """同步右侧装备栏当前物品坐标/名称与顶部拆装扳手、重铸熔炉位置。"""
        self.inventory = [InventoryEntry(x=pos[0], y=pos[1], name=name) for pos, name in rail_items]
        self.wrench_pos = wrench_pos
        self.furnace_pos = furnace_pos

    def record_purchase(
        self,
        name: str,
        cost: int,
        *,
        stars: int = 1,
        is_tracked: bool = False,
        can_front: bool = True,
        can_back: bool = True,
    ) -> None:
        """记录商店购买意图，供下一步备战空位认领或同名自动升星消费。"""
        if not name:
            return
        # 若备战/场上已有两张同星同卡，买第 3 张会直接触发游戏自动升星
        owned = [
            r for r in self.slots.values()
            if r.occupied and r.name == name and r.stars == stars
        ]
        if len(owned) >= 2:
            # 优先由在场卡吸收合并并升星
            host = next((r for r in owned if r.zone != "bench"), owned[0])
            host.stars = min(3, host.stars + 1)
            for other in owned:
                if other is not host:
                    if not other.worn_items and other.equip_count == 0:
                        other.clear_unit()
            return
        self.pending_purchases.append({
            "name": name,
            "cost": cost,
            "stars": stars,
            "is_tracked": is_tracked,
            "can_front": can_front,
            "can_back": can_back,
        })

    def record_move(self, from_zone: str, from_idx: int, to_zone: str, to_idx: int) -> None:
        """备战拖放换位成功后同步更新两个格位的角色与装备记忆。"""
        src = self.slot(from_zone, from_idx)
        dst = self.slot(to_zone, to_idx)
        if not src.occupied:
            return
        snapshot = SlotRecord(zone=dst.zone, index=dst.index)
        snapshot.copy_unit_from(src)
        if dst.occupied:
            src.copy_unit_from(dst)
        else:
            src.clear_unit()
        dst.copy_unit_from(snapshot)

    def record_detail(
        self,
        zone: str,
        index: int,
        name: str,
        worn_items: list[str],
        *,
        stars: int | None = None,
        cost: int | None = None,
    ) -> None:
        """打开角色详情弹层后写入准确的角色名与 0～3 件已穿戴装备。"""
        rec = self.slot(zone, index)
        rec.occupied = True
        if name:
            rec.name = name
            if rec.template:
                self.template_names[rec.template] = name
        cleaned_items = [item for item in worn_items if item][:3]
        rec.worn_items = cleaned_items
        rec.equip_count = max(rec.equip_count, len(cleaned_items))
        if stars is not None:
            rec.stars = stars
        if cost is not None:
            rec.cost = cost
        rec.detail_checked_stage = self.stage

    def record_equip(self, zone: str, index: int, item_name: str) -> None:
        """成功穿戴/合成一件装备后更新对应角色的装备记录。"""
        rec = self.slot(zone, index)
        rec.occupied = True
        if item_name and len(rec.worn_items) < 3:
            rec.worn_items.append(item_name)
        rec.equip_count = min(3, max(rec.equip_count + 1, len(rec.worn_items)))

    def record_wrench(self, zone: str, index: int) -> list[str]:
        """用拆装扳手卸下某角色的全部已穿装备后清空其身上记录并返回列表。"""
        rec = self.slot(zone, index)
        removed = list(rec.worn_items)
        rec.worn_items = []
        rec.equip_count = 0
        return removed

    def record_combat_outcome(self, won: bool | None, current_hp: int | None = None) -> None:
        """战斗或节点结算后依据胜负结果与血量变化更新连胜 / 连败计数。"""
        if current_hp is not None and 0 <= current_hp <= 200:
            if won is None and self.last_combat_hp is not None:
                won = current_hp >= self.last_combat_hp
            self.hp = current_hp
            self.last_combat_hp = current_hp
        if won is True:
            self.win_streak += 1
            self.loss_streak = 0
        elif won is False:
            self.loss_streak += 1
            self.win_streak = 0

    def as_equip_snapshots(self) -> list[UnitEquipSnapshot]:
        """导出供装备/扳手决策模块消费的在场单位列表。"""
        snapshots: list[UnitEquipSnapshot] = []
        for rec in self.field_records():
            snapshots.append(
                UnitEquipSnapshot(
                    zone=rec.zone,
                    index=rec.index,
                    name=rec.name,
                    stars=rec.stars,
                    worn=tuple(rec.worn_items),
                    equip_count=rec.equip_count,
                    has_halo=rec.has_halo,
                )
            )
        return snapshots

    def to_dict(self) -> dict:
        """序列化为可持久化 JSON 字典。"""
        return {
            "rank_text": self.rank_text,
            "stage": self.stage,
            "plane": self.plane,
            "node": self.node,
            "gold": self.gold,
            "level": self.level,
            "exp_cur": self.exp_cur,
            "exp_req": self.exp_req,
            "placed": self.placed,
            "capacity": self.capacity,
            "hp": self.hp,
            "last_combat_hp": self.last_combat_hp,
            "win_streak": self.win_streak,
            "loss_streak": self.loss_streak,
            "guide_applied": self.guide_applied,
            "risky_emblem_reset": self.risky_emblem_reset,
            "wrench_pos": list(self.wrench_pos) if self.wrench_pos else None,
            "furnace_pos": list(self.furnace_pos) if self.furnace_pos else None,
            "equip_tracking_prefs": self.equip_tracking_prefs,
            "template_names": self.template_names,
            "slots": {k: asdict(v) for k, v in self.slots.items() if v.occupied or v.blocked},
            "inventory": [asdict(item) for item in self.inventory],
        }

    def save(self) -> None:
        """原子保存对局数据库快照到磁盘，防止崩溃或异常停机写坏文件。"""
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        write_data(self.path, payload)

    def load(self) -> bool:
        """从磁盘恢复上次对局数据库；若文件不存在或损坏则返回 False。"""
        if not self.path.exists():
            return False
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if not isinstance(data, dict):
            return False
        self.reset_from_dict(data)
        return True

    def reset_from_dict(self, data: dict) -> None:
        """用序列化字典恢复内存状态。"""
        self._init_empty_slots()
        self.rank_text = str(data.get("rank_text", ""))
        self.stage = str(data.get("stage", "1-1"))
        self.plane = int(data.get("plane", 1))
        self.node = int(data.get("node", 1))
        self.gold = data.get("gold")
        self.level = data.get("level")
        self.exp_cur = data.get("exp_cur")
        self.exp_req = data.get("exp_req")
        self.placed = int(data.get("placed", 0))
        self.capacity = int(data.get("capacity", 3))
        self.hp = data.get("hp")
        self.last_combat_hp = data.get("last_combat_hp")
        self.win_streak = int(data.get("win_streak", 0))
        self.loss_streak = int(data.get("loss_streak", 0))
        self.guide_applied = bool(data.get("guide_applied", False))
        self.risky_emblem_reset = bool(data.get("risky_emblem_reset", False))
        wp = data.get("wrench_pos")
        self.wrench_pos = (int(wp[0]), int(wp[1])) if isinstance(wp, list) and len(wp) == 2 else None
        fp = data.get("furnace_pos")
        self.furnace_pos = (int(fp[0]), int(fp[1])) if isinstance(fp, list) and len(fp) == 2 else None
        prefs = data.get("equip_tracking_prefs", {})
        if isinstance(prefs, dict):
            self.equip_tracking_prefs = {str(k): list(v) for k, v in prefs.items() if isinstance(v, list)}
        tnames = data.get("template_names", {})
        if isinstance(tnames, dict):
            self.template_names = {str(k): str(v) for k, v in tnames.items()}
        slots_data = data.get("slots", {})
        if isinstance(slots_data, dict):
            for key, raw in slots_data.items():
                if not isinstance(raw, dict):
                    continue
                rec = self.slot(str(raw.get("zone", "bench")), int(raw.get("index", 0)))
                rec.occupied = bool(raw.get("occupied", False))
                rec.blocked = bool(raw.get("blocked", False))
                rec.template = str(raw.get("template", ""))
                rec.name = str(raw.get("name", ""))
                rec.stars = int(raw.get("stars", 1))
                rec.cost = raw.get("cost")
                rec.can_front = bool(raw.get("can_front", True))
                rec.can_back = bool(raw.get("can_back", True))
                rec.is_tracked = bool(raw.get("is_tracked", False))
                rec.has_halo = bool(raw.get("has_halo", False))
                rec.equip_count = int(raw.get("equip_count", 0))
                rec.worn_items = list(raw.get("worn_items", []))
                rec.detail_checked_stage = str(raw.get("detail_checked_stage", ""))
        inv_data = data.get("inventory", [])
        if isinstance(inv_data, list):
            self.inventory = [
                InventoryEntry(x=int(item["x"]), y=int(item["y"]), name=item.get("name"))
                for item in inv_data
                if isinstance(item, dict) and "x" in item and "y" in item
            ]
