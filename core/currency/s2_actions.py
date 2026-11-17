"""货币战争 S2「入画奇航」自动化动作混入模块。"""

import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np

from core.currency.combo_strategy import (
    ALL_COMBO_NAMES,
    COMBO_BACK_ORDER,
    COMBO_FRONT_ORDER,
    decide_combo_shop,
    plan_combo_deployment,
    plan_combo_sell,
)
from core.currency.equipment_db import (
    ALL_KNOWN_ITEM_NAMES,
    WRENCH_NAMES,
    lookup_recipe,
    plan_best_equip_action,
    plan_wrench_transfer,
    score_item_for_unit,
)
from core.currency.match_db import MatchDatabase
from core.currency.preparation import (
    AREA_BOXES,
    BENCH_BOXES,
    ItemIconCatalog,
    detect_card_equip_count,
    detect_card_stars,
    detect_inventory_rail_positions,
    detect_tool_positions,
    is_vivid_item_patch,
)
from core.currency.s2_strategy import (
    ANAXA_BACK,
    ANAXA_FRONT,
    BOND_PLANS,
    S2_ENV_PRIOR,
    clean_unit_name,
    decide_shop_action,
    detect_bench_guide_badges,
    detect_orb_points,
    detect_shop_guide_badges,
    is_button_lit,
    parse_shop_offers,
    pick_encounter_index,
    score_halo_text,
    score_investment_text,
    score_item_choice,
)
from core.currency.settings import load_s2_settings
from tool.GLOBAL import key_mouse_manager
from tool.log import CUS_LOGGER
from tool.public_ocr import clean_text, merge_text, sort_text
from tool.storage import config_path

ZZZ_LOCK_PATH = Path(
    os.environ.get("SIMS_ZZZ_LOCK_PATH")
    or (Path(__file__).resolve().parents[3] / "tools" / "ZZZ_RUNNING.lock")
)

# S2 界面关键区域（1920x1080 游戏客户区坐标，参考实际 S2 布局并留实机微调余量）
ADVANCE_BUTTON_BOX = [1690, 704, 1918, 800]
ADVANCE_KEYWORDS = (
    "出战", "跳过", "选择主角", "开启宝库",
    "幕后茶歇", "开始茶歇", "前往茶歇", "前往补给",
    "前往", "挑战", "进入",
)
SHOP_TOGGLE_POS = (1620, 980)
SHOP_REFRESH_POS = (1620, 505)
INVESTMENT_CENTERS = (460, 960, 1460)
SUPPLY_FIVE_X = (250, 605, 960, 1315, 1670)
SUPPLY_FOUR_X = (428, 783, 1137, 1492)
COST_BAR_PALETTE = np.array(
    ((128, 131, 146), (52, 139, 119), (90, 107, 220), (143, 109, 249), (210, 165, 65)),
    dtype=np.float32,
)


class CurrencyS2Mixin:
    """为 SimulatedCurrency 提供 S2「入画奇航」全流程操作能力。"""

    def init_s2(self):
        """初始化 S2 运行参数、轮换计划与防卡死状态。"""
        self.s2_cfg = load_s2_settings()
        self._s2_run_count = 0
        self._shop_done_stage = None
        self._equip_done_stage = None
        self._item_done_stage = None
        self._item_open_tries = {}
        self._shop_ops = {}
        self._shop_refreshes = {}
        self._equip_attempts = {}
        self._match_result = None
        self._rank_steps = 0
        self._tea_steps = 0
        self._unlock_steps = 0
        self._last_team_counts = None
        self._halo_drag_idx = 0
        self._halo_replace_tries = 0
        self._halo_force_next_drag = False
        self._special_shop_tries = 0
        self._supply_click_idx = 0
        self._master_tries = 0
        self._generic_choice_tries = 0
        self._orbs_done_stage = None
        self._orb_overflow_stage = None
        self._orb_tries = {}
        self._guide_applied_run = False
        self._guide_tries = 0
        self._shop_open_tries = {}
        self._prep_stall_tries = {}
        self._equip_checked_slots = {}
        self._equip_drag_tries = {}
        self._proactive_sells = {}
        self._pending_equip_slot = None
        self._wrench_done_slots = set()
        self._learned_rail_coords = set()
        self.match_db = MatchDatabase()
        self._item_catalog = ItemIconCatalog()
        self._reset_run_stage_flags()
        if self.match_db.load():
            self._guide_applied_run = self.match_db.guide_applied

    def _reset_run_stage_flags(self):
        self._shop_done_stage = None
        self._equip_done_stage = None
        self._item_done_stage = None
        self._item_open_tries.clear()
        self._shop_ops.clear()
        self._shop_refreshes.clear()
        self._equip_attempts.clear()
        self._match_result = None
        self._rank_steps = 0
        self._tea_steps = 0
        self._last_team_counts = None
        self._halo_drag_idx = 0
        self._halo_replace_tries = 0
        self._halo_force_next_drag = False
        self._special_shop_tries = 0
        self._supply_click_idx = 0
        self._master_tries = 0
        self._generic_choice_tries = 0
        self._orbs_done_stage = None
        self._orb_overflow_stage = None
        self._orb_tries = {}
        self._guide_applied_run = False
        self._guide_tries = 0
        self._shop_open_tries.clear()
        self._prep_stall_tries.clear()
        self._equip_checked_slots.clear()
        self._equip_drag_tries.clear()
        self._proactive_sells.clear()
        self._pending_equip_slot = None
        self._wrench_done_slots.clear()
        self._learned_rail_coords.clear()

    def _s2_state_path(self) -> Path:
        return config_path(__file__).with_name("currency_s2_state.json")

    def _load_s2_state(self) -> dict:
        try:
            return json.loads(self._s2_state_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"done_plans": [], "history": []}

    def _save_s2_state(self, state: dict) -> None:
        try:
            self._s2_state_path().parent.mkdir(parents=True, exist_ok=True)
            self._s2_state_path().write_text(
                json.dumps(state, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as error:
            CUS_LOGGER.warning("货币战争 S2 状态保存失败：%s", error)

    def _current_plan(self) -> tuple[str, tuple[str, ...]]:
        """返回本局 (计划ID, 目标角色元组)，支持按轮换列表推进未完成羁绊。"""
        done = set(self._load_s2_state().get("done_plans", []))
        for item in self.s2_cfg.get("rotation", []):
            plan_id = item.get("id")
            names = item.get("targets")
            if plan_id and plan_id not in done and isinstance(names, list) and names:
                return plan_id, tuple(str(n) for n in names)
        strategy = self.s2_cfg.get("strategy", "s2_anaxa_aglaea")
        if strategy == "s2_anaxa_aglaea":
            return strategy, ALL_COMBO_NAMES
        return strategy, BOND_PLANS.get(strategy, ANAXA_FRONT + ANAXA_BACK)

    def _ocr_box_text(self, box, *, redundancy=20, strip_ascii=False) -> str:
        try:
            rows = list(self.ts.find_with_box(box, redundancy=redundancy, forward=0))
            if strip_ascii:
                return merge_text(rows)
            return "".join(str(r.get("raw_text", "")).strip() for r in sort_text(rows))
        except Exception:
            return ""

    def _ocr_box_rows(self, box, *, redundancy=20) -> list[dict]:
        try:
            return list(self.ts.find_with_box(box, redundancy=redundancy, forward=0))
        except Exception:
            return []

    def _read_int(self, box, *, upper=999) -> int | None:
        raw = ""
        try:
            raw = self.ts.ocr_one_row(self.screen, box)
        except Exception:
            raw = ""
        digits = re.findall(r"\d+", raw)
        if not digits:
            digits = re.findall(r"\d+", self._ocr_box_text(box, redundancy=15))
        if len(digits) == 1:
            val = int(digits[0])
            if 0 <= val <= upper:
                return val
        return None

    def _read_stage(self) -> str:
        for box, red in (([226, 25, 558, 122], 15), ([240, 80, 360, 130], 25)):
            text = self._ocr_box_text(box, redundancy=red)
            match = re.search(r"([1-3]-[1-9]\d?)", text)
            if match:
                return match.group(1)
        return "0-0"

    def _read_board_gold(self) -> int | None:
        # 优先仅裁切金币数字区（避开左侧六角金币图标干扰单数字 0/3/5 识别）
        for box in ([1634, 898, 1696, 944], [1612, 890, 1686, 946], [1580, 890, 1690, 950]):
            val = self._read_int(box, upper=999)
            if val is not None:
                return val
        # 0 金币偶发被 OCR 识作字母 O/o
        raw = self._ocr_box_text([1634, 898, 1696, 944], redundancy=15).strip()
        if raw in {"O", "o", "D", "Q"}:
            return 0
        raw_wide = self._ocr_box_text([1612, 890, 1686, 946], redundancy=15).strip()
        return 0 if raw_wide in {"O", "o", "D", "Q"} else None

    def _read_board_level(self) -> int | None:
        text = self._ocr_box_text([235, 880, 360, 935], redundancy=15)
        match = re.search(r"(?:Lv\.?|L|V)?\s*([1-9]\d?)", text, re.IGNORECASE)
        if match:
            val = int(match.group(1))
            if 1 <= val <= 15:
                return val
        return None

    def _read_xp_cost(self) -> int | None:
        if "购买经验" not in self._ocr_box_text([225, 830, 365, 905], redundancy=15):
            return None
        return self._read_int([295, 965, 355, 1015], upper=9) or 4

    def _read_refresh_cost(self) -> int | None:
        text = self._ocr_box_text([1535, 450, 1710, 565], redundancy=15)
        if "免费刷新" in text:
            return 0
        if "刷新" not in text:
            return None
        return self._read_int([1590, 500, 1680, 560], upper=9) or 2

    def _read_board_hp(self) -> int | None:
        return self._read_int([1418, 55, 1495, 102], upper=200)

    def _read_board_exp(self) -> tuple[int, int] | None:
        text = self._ocr_box_text([232, 925, 358, 970], redundancy=15)
        match = re.search(r"(\d+)\s*[/／]\s*(\d+)", text)
        if not match:
            return None
        cur, req = int(match.group(1)), int(match.group(2))
        return (cur, req) if 0 <= cur <= 200 and req > 0 else None

    def _box_lit(self, box) -> bool:
        x1, y1, x2, y2 = box
        patch = self.screen[y1:y2, x1:x2]
        return is_button_lit(patch)

    def _click_lit_confirm(self, box, labels=("确认", "确认选择", "选定", "选择")) -> bool:
        """在目标区域查找确认按钮文字并在亮起后点击。"""
        rows = self._ocr_box_rows(box, redundancy=25)
        for row in rows:
            txt = row["raw_text"].strip()
            if any(k in txt for k in labels):
                rb = row["box"]
                if self._box_lit([max(0, rb[0] - 25), max(0, rb[1] - 8), min(1920, rb[2] + 25), min(1080, rb[3] + 8)]):
                    cx = (rb[0] + rb[2]) // 2
                    cy = (rb[1] + rb[3]) // 2
                    key_mouse_manager.click_point(cx, cy, box=rb)
                    key_mouse_manager.wait()
                    return True
        return False

    def _click_text_fast(self, labels, box, fallback=None, *, aliases=()) -> bool:
        """优先用全屏 OCR 缓存命中目标文字并点击，失败再退回局部重识别或保底坐标。"""
        targets = (labels,) if isinstance(labels, str) else tuple(labels)
        all_keys = targets + tuple(aliases)
        for row in self._ocr_box_rows(box, redundancy=25):
            cleaned = clean_text(row["raw_text"], 1)
            if any(k in cleaned for k in all_keys):
                rb = row["box"]
                key_mouse_manager.click_point((rb[0] + rb[2]) // 2, (rb[1] + rb[3]) // 2, box=rb)
                key_mouse_manager.wait()
                return True
        query = targets[0] if len(targets) == 1 else list(targets)
        if self.click_text(query, box=box, click=True, allow_fail=True):
            return True
        if fallback is not None:
            key_mouse_manager.click_point(*fallback)
            key_mouse_manager.wait()
            return True
        return False

    def _offer_border_cost(self, x: int) -> int | None:
        patch = self.screen[293:316, x + 152:x + 169]
        if patch.size == 0:
            return None
        color = np.median(patch.reshape(-1, 3), axis=0)
        # screen 为 BGR，把基准调色板按 BGR 比较前先转 RGB
        rgb = color[::-1]
        distance = np.abs(COST_BAR_PALETTE - rgb).sum(axis=1)
        hits = np.flatnonzero(distance < 55)
        return int(hits[0]) + 1 if len(hits) == 1 else None

    # ---------- 局外入口、模式、难度 ----------

    def s2_handle_reconnect(self):
        if self._click_lit_confirm([1020, 620, 1340, 730], ("确认", "重试")):
            return 1
        return 0

    def s2_handle_pause_modal(self):
        if self.state == "exit":
            self._click_text_fast("放弃并结算", [660, 700, 900, 810])
            return 1
        CUS_LOGGER.info("关闭意外弹出的「是否暂时离开」菜单")
        key_mouse_manager.press("esc")
        key_mouse_manager.wait()
        return 1

    def s2_handle_weekly_reward(self):
        rows = self._ocr_box_rows([1440, 680, 1850, 770], redundancy=20)
        if any("领取" in r["raw_text"] for r in rows) and self._box_lit([1520, 700, 1760, 755]):
            key_mouse_manager.click_point(1640, 728)
            key_mouse_manager.wait()
            return 1
        key_mouse_manager.press("esc")
        key_mouse_manager.wait()
        return 1

    def s2_open_guide_tab(self):
        self._click_text_fast("旷宇纷争", [450, 160, 800, 260], fallback=(593, 212))
        return 1

    def s2_handle_guide_tab(self):
        """在星际和平指南·旷宇纷争页切到货币战争并点击前往参与。"""
        self.ts.forward(self.get_screen())
        if "入画奇航" not in self._ocr_box_text([1400, 330, 1750, 460], redundancy=20):
            self._click_text_fast("货币战争", [250, 280, 500, 420], fallback=(365, 344))
            return 1
        if "未解锁" in self._ocr_box_text([780, 520, 1020, 620], redundancy=20):
            CUS_LOGGER.warning("当前指南显示入画奇航未解锁，尝试点击前往参与查看引导")
        self._click_text_fast("前往参与", [1350, 820, 1650, 930], fallback=(1491, 878))
        self.update_state("start_currencywar")
        return 1

    def _step_unlock_guard(self, label: str) -> bool:
        self._unlock_steps += 1
        if self._unlock_steps > 8:
            CUS_LOGGER.error("新手引导连续执行 %d 次（%s）仍未进入入画奇航主页，停止运行防止死循环", self._unlock_steps, label)
            self.stop()
            return False
        return True

    def s2_handle_unlock_quest(self):
        """在任务面板点击「前往」或「开始追踪」，自动推进货币战争·入画奇航解锁/试用任务。"""
        if not self._step_unlock_guard("任务页"):
            return 0
        self.ts.forward(self.get_screen())
        for label in ("前往", "开始追踪"):
            if self._click_text_fast(label, [1480, 940, 1830, 1025]):
                CUS_LOGGER.info("新手引导任务页点击「%s」", label)
                return 1
        return 0

    def s2_handle_unlock_nav(self):
        """在导航地图页点击「传送」，前往匹诺康尼·朝露公馆货币战争终端。"""
        if not self._step_unlock_guard("导航页"):
            return 0
        self.ts.forward(self.get_screen())
        if self._click_text_fast("传送", [1640, 930, 1890, 1020], aliases=("传诺",)):
            CUS_LOGGER.info("新手引导导航页点击「传送」前往朝露公馆")
            return 1
        return 0

    def s2_handle_home(self):
        """入画奇航首页：检查停机让位/局数上限，优先清理新手激活引导后开始货币战争。"""
        self._unlock_steps = 0
        if ZZZ_LOCK_PATH.exists():
            CUS_LOGGER.warning("检测到 ZZZ_RUNNING.lock，停止货币战争让位绝区零")
            self.stop()
            return 0
        if self._s2_run_count >= self.s2_cfg["max_runs"]:
            CUS_LOGGER.info("已达到 S2 运行局数上限 %d，停止运行", self.s2_cfg["max_runs"])
            self.stop()
            return 0
        self.ts.forward(self.get_screen())
        if "记得激活" in self._ocr_box_text([280, 310, 860, 420], redundancy=25):
            CUS_LOGGER.info("检测到主页「优势布局」激活提示，先点击优势布局清理引导")
            self._click_text_fast("优势布局", [50, 320, 260, 410], fallback=(154, 364))
            return 1
        plan_id, _ = self._current_plan()
        CUS_LOGGER.info("准备开局入画奇航（第 %d 局，当前计划：%s）", self._s2_run_count + 1, plan_id)
        self._click_text_fast("开始货币战争", [1300, 900, 1895, 1030], fallback=(1586, 973))
        self.update_state("start_currencywar")
        return 1

    def s2_handle_advantage(self):
        """在优势布局界面激活一次可点布局后返回入画奇航主页。"""
        self.ts.forward(self.get_screen())
        if "已布局" not in self._ocr_box_text([1600, 895, 1860, 980], redundancy=20):
            if self._click_lit_confirm([1620, 895, 1860, 980], ("布局",)):
                CUS_LOGGER.info("在优势布局页面激活一条局外常驻优势")
                return 1
        key_mouse_manager.press("esc")
        key_mouse_manager.wait()
        return 1

    def s2_handle_modes(self):
        """模式选择页：有未完成对局时按配置续局，否则切到指定模式并进入。"""
        self.ts.forward(self.get_screen())
        bottom = self._ocr_box_text([1200, 880, 1910, 1040], redundancy=25)
        if self.s2_cfg["resume_unfinished"] and any(k in bottom for k in ("继续进度", "继续对局", "继续博弈")):
            if self._click_text_fast(("继续进度", "继续对局", "继续博弈"), [1200, 880, 1910, 1040]):
                CUS_LOGGER.info("检测到未完成对局，接管继续进度并载入对局数据库")
                self._reset_run_stage_flags()
                self.match_db.load()
                self._guide_applied_run = self.match_db.guide_applied
                self.update_state("currency_main")
                return 1

        target_mode = self.s2_cfg["mode"]
        target_entry = "进入超频博弈" if target_mode == "overclock" else "进入标准博弈"
        if target_entry in bottom:
            self._click_text_fast(target_entry, [1200, 880, 1910, 1040])
            self.update_state("difficulty_select")
            return 1

        # 切换到目标博弈模式卡片
        card_text = "超频博弈" if target_mode == "overclock" else "标准博弈"
        y = 418 if target_mode == "overclock" else 260
        self._click_text_fast(card_text, [50, 174, 510, 550], fallback=(153, y))
        return 1

    def s2_handle_rank(self):
        """职级难度选择页：rank='lowest' 时降到 A0 黑铁；默认/highest 通过「返回最高职级」切回最高已解锁职级后开局。"""
        self.ts.forward(self.get_screen())
        cur_rank = self._ocr_box_text([70, 290, 350, 355], redundancy=20)
        if self.s2_cfg["rank"] == "lowest":
            is_black_iron = "黑铁" in self._ocr_box_text([50, 290, 350, 560], redundancy=20)
            rank_num = self._read_int([490, 920, 600, 995], upper=999)
            if not is_black_iron and rank_num != 1:
                if rank_num is not None and rank_num > 1 and self._rank_steps < 25:
                    self._rank_steps += 1
                    clicks = min(3, rank_num - 1)
                    CUS_LOGGER.info("当前敌人难度 %d > 1，点击左侧低职级 %d 次", rank_num, clicks)
                    for _ in range(clicks):
                        key_mouse_manager.click_point(270, 780)
                        key_mouse_manager.sleep(0.25)
                    key_mouse_manager.wait()
                    return 1
                if rank_num is None and not self.is_one() and self._rank_steps < 15:
                    self._rank_steps += 1
                    key_mouse_manager.click_point(270, 780)
                    key_mouse_manager.wait()
                    return 1
        else:
            back_btn = self._ocr_box_text([1240, 910, 1545, 1020], redundancy=20)
            if "返回最高职级" in back_btn and self._rank_steps < 12:
                self._rank_steps += 1
                CUS_LOGGER.info("当前职级非最高（%s），点击「返回最高职级」第 %d 次", cur_rank or "未知", self._rank_steps)
                self._click_text_fast("返回最高职级", [1240, 910, 1545, 1020], fallback=(1392, 965))
                key_mouse_manager.sleep(0.55)
                key_mouse_manager.wait()
                return 1
            if cur_rank:
                CUS_LOGGER.info("确认最高已解锁职级开局：%s", cur_rank)

        self._reset_run_stage_flags()
        self.match_db.reset_run(rank_text=cur_rank)
        self.investment_tracker.reset()
        self.preparation.reset_members()
        self.run_history.start_run()
        self.update_state("difficulty_select")
        self._click_text_fast("开始对局", [1550, 910, 1850, 1020], fallback=(1692, 965))
        return 1

    # ---------- 投资环境与策略（图鉴 + 安全优先） ----------

    def _s2_pick_three_columns(self, title_label: str, boxes: list[list[int]], text_boxes: list[list[int]], prefer: tuple[str, ...]):
        """通用三栏环境/策略评分与确认逻辑。"""
        self.ts.forward(self.get_screen())
        texts = [self._ocr_box_text(tb, redundancy=25) for tb in text_boxes]
        has_icons = []
        for x in INVESTMENT_CENTERS:
            # S2 图鉴新角标位于卡片右上角约 28x28 区域；待实机截图后再校准偏移
            roi = self.screen[218:246, x + 157:x + 185]
            flag, _ = self.detect_has_icon(roi) if roi.size else (False, 0.0)
            has_icons.append(bool(flag))

        catalog_first = self.s2_cfg.get("catalog_first", True)
        scores = [
            score_investment_text(texts[i], has_icons[i], prefer, catalog_first=catalog_first)
            for i in range(3)
        ]
        CUS_LOGGER.info("%s评分：%s (图鉴=%s)", title_label, scores, has_icons)
        best = max(range(3), key=lambda i: scores[i])
        cx = (boxes[best][0] + boxes[best][2]) // 2
        cy = (boxes[best][1] + boxes[best][3]) // 2
        key_mouse_manager.click_point(cx, cy, box=boxes[best])
        key_mouse_manager.wait()
        key_mouse_manager.sleep(0.25)
        self.ts.forward(self.get_screen())
        if not self._click_lit_confirm([820, 935, 1340, 1030], ("确认",)):
            key_mouse_manager.click_point(960, 985)
            key_mouse_manager.wait()
        return best, texts[best], has_icons[best]

    def s2_select_environment(self):
        text_boxes = [[200, 350, 700, 840], [700, 350, 1220, 840], [1220, 350, 1740, 840]]
        if not self.has_fresh:
            self.ts.forward(self.get_screen())
            texts = [self._ocr_box_text(tb, redundancy=25) for tb in text_boxes]
            has_any_icon = any(
                self.detect_has_icon(self.screen[218:246, x + 157:x + 185])[0]
                for x in INVESTMENT_CENTERS
            )
            has_pref = any(p in t for p in S2_ENV_PRIOR for t in texts)
            if not has_any_icon and not has_pref and "刷新" in self._ocr_box_text([560, 940, 780, 1020]):
                self.has_fresh = True
                key_mouse_manager.click_point(672, 984)
                key_mouse_manager.wait()
                return 1
        self.has_fresh = False
        _, chosen_env, _ = self._s2_pick_three_columns("投资环境", self.ENVIR_BOXES, text_boxes, S2_ENV_PRIOR)
        if any(k in chosen_env for k in ("重置", "随机", "改变星徽")):
            self.match_db.risky_emblem_reset = True
        self.investment_tracker.reset()
        self.update_state("1-1")
        return 1

    def s2_select_strategy(self):
        selection = self.investment_tracker.begin_selection()
        text_boxes = [[200, 440, 700, 840], [700, 440, 1220, 840], [1220, 440, 1740, 840]]
        _, chosen_text, has_icon = self._s2_pick_three_columns(
            "投资策略", self.BLESS_BOXES, text_boxes, ()
        )
        if any(k in chosen_text for k in ("重置", "随机", "改变星徽")):
            self.match_db.risky_emblem_reset = True
        self.investment_tracker.complete_selection(selection, grants_bonus=False)
        if has_icon and chosen_text:
            title = chosen_text[:16]
            self.run_history.record_unlocked_investment(title)
        if not self.s2_cfg.get("complete_run", True) and self.investment_tracker.should_exit(self.exit_plane):
            CUS_LOGGER.info("达到指定退出位面，请求退出结算")
            self.exit_requested = True
            self.update_state("exit")
        return 1

    # ---------- 商店运营、备战、装备推荐 ----------

    def s2_manage_shop(self):
        """处理商店开启状态：记忆同期状态、连击阿格莱雅/那刻夏精准买牌/吃息升级/D牌并收起。"""
        if self._stop or self.exit_requested:
            return 0
        self._special_shop_tries = 0
        self.ts.forward(self.get_screen())
        stage = self._read_stage()
        self._shop_open_tries[stage] = 0
        ops = self._shop_ops.get(stage, 0)
        refreshes = self._shop_refreshes.get(stage, 0)
        gold = self._read_board_gold()
        level = self._read_board_level()
        exp = self._read_board_exp()
        # 商店抽屉会遮挡场上 5/6 人数区域，优先复用备战画面测得的真实人数
        counts = self._last_team_counts or self.read_team_count() or (0, 3)
        xp_cost = self._read_xp_cost()
        refresh_cost = self._read_refresh_cost()
        slots = self.preparation.scan_bench(self.screen)
        bench_empty = sum(1 for s in slots if s.kind == "empty")
        bench_chars = sum(1 for s in slots if s.kind == "character")

        slot_texts = []
        for col in range(5):
            x = 367 + 269 * col
            rows = self._ocr_box_rows([x, 280, x + 245, 338], redundancy=15)
            rows.sort(key=lambda r: r["box"][0])
            joined = "".join(r["raw_text"] for r in rows)
            slot_texts.append((col, joined, self._offer_border_cost(x)))

        offers = parse_shop_offers(slot_texts)
        badges = detect_shop_guide_badges(self.screen)
        tracked_cols = tuple(idx for idx, flag in enumerate(badges) if flag)
        plan_id, targets = self._current_plan()

        self.match_db.sync_economy(stage=stage, gold=gold, level=level, exp=exp, counts=counts)
        self.match_db.sync_bench(slots)

        sell_slot_override = None
        if plan_id == "s2_anaxa_aglaea":
            if gold is None:
                action, offer = "done", None
            else:
                action, offer = decide_combo_shop(
                    self.match_db,
                    offers=offers,
                    tracked_cols=tracked_cols,
                    refresh_cost=refresh_cost,
                    xp_cost=xp_cost,
                    refreshes=refreshes,
                    ops=ops,
                    max_ops=self.s2_cfg["max_shop_ops"],
                    target_level=self.s2_cfg["target_level"],
                )
            if action == "sell":
                sell_slot_override = plan_combo_sell(self.match_db)
        else:
            action, offer = decide_shop_action(
                gold=gold,
                level=level,
                placed=counts[0],
                capacity=counts[1],
                bench_empty=bench_empty,
                offers=offers,
                target_names=targets,
                refresh_cost=refresh_cost,
                xp_cost=xp_cost,
                refreshes=refreshes,
                ops=ops,
                max_refreshes=self.s2_cfg["max_shop_refreshes"],
                max_ops=self.s2_cfg["max_shop_ops"],
                target_level=self.s2_cfg["target_level"],
                tracked_cols=tracked_cols,
                guide_active=self._guide_applied_run,
                bench_chars=bench_chars,
            )
        CUS_LOGGER.debug(
            "S2商店[%s] 金币=%s 等级=%s 人数=%s 连胜=%d/连败=%d 空位=%d 追踪=%s 商品=%s → %s",
            stage, gold, level, counts, self.match_db.win_streak, self.match_db.loss_streak,
            bench_empty, tracked_cols, offers, action,
        )

        if action == "buy" and offer is not None:
            self._shop_ops[stage] = ops + 1
            if self.match_db.gold is not None:
                self.match_db.gold = max(0, self.match_db.gold - offer.cost)
            self.match_db.record_purchase(
                offer.name,
                offer.cost,
                is_tracked=(offer.column in tracked_cols),
            )
            self.match_db.save()
            px = 367 + 269 * offer.column + 106
            CUS_LOGGER.info("购买商店 %d 号位：%s（%d 费）", offer.column + 1, offer.name, offer.cost)
            key_mouse_manager.click_point(px, 194)
            key_mouse_manager.wait()
            return 1
        if action == "exp":
            self._shop_ops[stage] = ops + 1
            if self.match_db.gold is not None:
                self.match_db.gold = max(0, self.match_db.gold - (xp_cost if xp_cost is not None else 4))
                self.match_db.save()
            CUS_LOGGER.info("按 F 购买经验升级人口")
            key_mouse_manager.press("f")
            key_mouse_manager.wait()
            return 1
        if action == "refresh":
            self._shop_ops[stage] = ops + 1
            self._shop_refreshes[stage] = refreshes + 1
            if self.match_db.gold is not None:
                self.match_db.gold = max(0, self.match_db.gold - (refresh_cost if refresh_cost is not None else 2))
                self.match_db.save()
            CUS_LOGGER.info("按 D 刷新商店")
            key_mouse_manager.press("d")
            key_mouse_manager.wait()
            return 1
        if action == "sell":
            self._shop_ops[stage] = ops + 1
            key_mouse_manager.click_point(*SHOP_TOGGLE_POS)
            key_mouse_manager.wait()
            sold = (
                self._sell_bench_slot(sell_slot_override.index, reason="数据库优选腾位")
                if sell_slot_override is not None
                else self._sell_one_non_target(targets)
            )
            if not sold:
                self._shop_done_stage = stage
            return 1

        self._shop_done_stage = stage
        self.match_db.save()
        key_mouse_manager.click_point(*SHOP_TOGGLE_POS)
        key_mouse_manager.wait()
        return 1

    def _sell_bench_slot(self, slot_index: int, *, reason: str = "腾出空位") -> bool:
        """点选并出售指定备战格角色，同时从记忆数据库清除该格记录。"""
        bx = (BENCH_BOXES[slot_index][0] + BENCH_BOXES[slot_index][2]) // 2
        by = (BENCH_BOXES[slot_index][1] + BENCH_BOXES[slot_index][3]) // 2
        key_mouse_manager.click_point(bx, by)
        key_mouse_manager.wait()
        key_mouse_manager.sleep(0.2)
        self.ts.forward(self.get_screen())
        if self._click_text_fast("出售", [1680, 875, 1915, 965]):
            unit_name = self.match_db.slot("bench", slot_index).name or "未知"
            self.match_db.slot("bench", slot_index).clear_unit()
            self.match_db.save()
            CUS_LOGGER.info("出售备战格 %d「%s」（%s）", slot_index + 1, unit_name, reason)
            return True
        self._close_detail_sidebar()
        return False

    def _sell_one_non_target(
        self,
        target_names: tuple[str, ...],
        *,
        allow_fallback: bool = False,
        require_untracked: bool = False,
    ) -> bool:
        """从右到左找一张不在目标列表中的备战角色出售腾位；攻略追踪时优先直接卖出未追踪角色。"""
        self.ts.forward(self.get_screen())
        if "出售" in self._ocr_box_text([1680, 875, 1915, 965], redundancy=15):
            self._close_detail_sidebar()
            self.ts.forward(self.get_screen())
        slots = self.preparation.scan_bench(self.screen)
        if self._guide_applied_run:
            badges = detect_bench_guide_badges(self.screen)
            for slot in reversed(slots):
                if slot.kind != "character" or badges[slot.index]:
                    continue
                bx = (BENCH_BOXES[slot.index][0] + BENCH_BOXES[slot.index][2]) // 2
                by = (BENCH_BOXES[slot.index][1] + BENCH_BOXES[slot.index][3]) // 2
                key_mouse_manager.click_point(bx, by)
                key_mouse_manager.wait()
                key_mouse_manager.sleep(0.2)
                self.ts.forward(self.get_screen())
                if self._click_text_fast("出售", [1680, 875, 1915, 965]):
                    self.match_db.slot("bench", slot.index).clear_unit()
                    CUS_LOGGER.info("出售备战格 %d 未追踪冗余角色变现腾位", slot.index + 1)
                    return True
                self._close_detail_sidebar()
            if require_untracked:
                return False

        candidates: list[tuple[int, int, int, int, str]] = []
        for slot in reversed(slots):
            if slot.kind != "character":
                continue
            bx = (BENCH_BOXES[slot.index][0] + BENCH_BOXES[slot.index][2]) // 2
            by = (BENCH_BOXES[slot.index][1] + BENCH_BOXES[slot.index][3]) // 2
            key_mouse_manager.click_point(bx, by)
            key_mouse_manager.wait()
            key_mouse_manager.sleep(0.2)
            self.ts.forward(self.get_screen())
            raw_unit = self._ocr_box_text([1488, 200, 1745, 255], redundancy=15)
            name = re.sub(r"^(?:试用)+|\d+$", "", clean_unit_name(raw_unit))
            matched_idx = next(
                (
                    i
                    for i, t in enumerate(target_names)
                    if name and (t in name or (len(name) >= 3 and t.startswith(name[:3])))
                ),
                None,
            )
            if matched_idx is not None:
                candidates.append((matched_idx, slot.index, bx, by, name))
                self._close_detail_sidebar()
                continue
            if self._click_text_fast("出售", [1680, 875, 1915, 965]):
                CUS_LOGGER.info("出售备战格 %d 非核心角色「%s」腾出空间", slot.index + 1, name or "未知")
                return True
            self._close_detail_sidebar()

        if allow_fallback and candidates:
            _, s_idx, bx, by, name = max(candidates, key=lambda item: (item[0], item[1]))
            key_mouse_manager.click_point(bx, by)
            key_mouse_manager.wait()
            key_mouse_manager.sleep(0.2)
            self.ts.forward(self.get_screen())
            if self._click_text_fast("出售", [1680, 875, 1915, 965]):
                CUS_LOGGER.warning("备战席全为核心无法腾空，兜底出售备战格 %d「%s」解卡死", s_idx + 1, name)
                return True
            self._close_detail_sidebar()
        return False

    def _click_text_fast(self, labels, box, fallback=None, *, aliases=()) -> bool:
        """优先用全屏 OCR 缓存命中目标文字并点击，失败再退回局部重识别或保底坐标。"""
        targets = (labels,) if isinstance(labels, str) else tuple(labels)
        all_keys = targets + tuple(aliases)
        for row in self._ocr_box_rows(box, redundancy=25):
            cleaned = clean_text(row["raw_text"], 1)
            if any(k in cleaned for k in all_keys):
                rb = row["box"]
                key_mouse_manager.click_point((rb[0] + rb[2]) // 2, (rb[1] + rb[3]) // 2, box=rb)
                key_mouse_manager.wait()
                return True
        query = targets[0] if len(targets) == 1 else list(targets)
        if self.click_text(query, box=box, click=True, allow_fail=True):
            return True
        if fallback is not None:
            key_mouse_manager.click_point(*fallback)
            key_mouse_manager.wait()
            return True
        return False

    def s2_handle_bench_overflow(self):
        _, targets = self._current_plan()
        if self._sell_one_non_target(targets, allow_fallback=True):
            return 1
        key_mouse_manager.press("f")
        key_mouse_manager.wait()
        return 1

    def s2_handle_empty_front(self):
        self.click_text("确认", box=[755, 605, 1270, 840], click=True, allow_fail=True)
        slots = self.preparation.scan_bench(self.get_screen())
        chars = [s for s in slots if s.kind == "character"]
        ordered = [s for s in chars if "front" in s.areas] or chars
        for slot in ordered:
            sx = (BENCH_BOXES[slot.index][0] + BENCH_BOXES[slot.index][2]) // 2
            sy = (BENCH_BOXES[slot.index][1] + BENCH_BOXES[slot.index][3]) // 2
            tx = (AREA_BOXES["front"][0][0] + AREA_BOXES["front"][0][2]) // 2
            ty = (AREA_BOXES["front"][0][1] + AREA_BOXES["front"][0][3]) // 2
            key_mouse_manager.drag_point(sx, sy, tx, ty)
            key_mouse_manager.wait()
            return 1
        return 0

    def s2_prepare_and_advance(self):
        """备战阶段主调度：开箱 → 逛商店 → 拖上阵 → 推荐穿装 → 点击出战/推进。"""
        if self._stop or self.exit_requested:
            return 0
        self._special_shop_tries = 0
        self._tea_steps = 0
        self._halo_drag_idx = 0
        self._halo_replace_tries = 0
        self._halo_force_next_drag = False
        self._master_tries = 0
        self._generic_choice_tries = 0
        self.ts.forward(self.get_screen())
        # 若从补给/茶歇/事件界面切回查看备战，右上角会有「返回补给阶段」等按钮，商店此时禁用，需先返回
        for row in self._ocr_box_rows([1620, 20, 1918, 110], redundancy=20):
            txt = row["raw_text"].strip()
            if "返回" in txt and "备战" not in txt:
                rb = row["box"]
                CUS_LOGGER.info("检测到右上角「%s」，返回原节点阶段", txt)
                key_mouse_manager.click_point((rb[0] + rb[2]) // 2, (rb[1] + rb[3]) // 2, box=rb)
                key_mouse_manager.wait()
                return 1
        # 若右上装备追踪被点开成了「✔ / × 取消」速合抽屉，先点「取消」关闭防止遮挡晶球区
        if "取消" in self._ocr_box_text([1620, 315, 1900, 415], redundancy=15):
            CUS_LOGGER.info("关闭右侧装备追踪速合浮层「取消」")
            self._click_text_fast("取消", [1620, 315, 1900, 415], fallback=(1800, 350))
            return 1
        stage = self._read_stage()
        if self.state == "1-1":
            self.preparation.reset_members()
            self.update_state("currency_main")

        # 先记录无遮挡备战面板上的人数，防止开箱提前 return 后商店首帧读取陈旧值
        board_counts = self.read_team_count()
        if board_counts is not None:
            self._last_team_counts = board_counts

        # 0. 先同步对局记忆数据库（金钱、等级、经验、血量、前后台占格/装备槽/星级、右栏与扳手）
        self._sync_match_db(stage)

        # -1. 本局尚未开启右上角攻略一键追踪时先应用推荐攻略（需关店状态露出右上图标）
        if self._ensure_guide_tracked():
            return 1

        # 0. 先拾取右上角战利品晶球（必须在商店收起时，且开出金币/角色/装备可马上投入本轮备战）
        if self._collect_orbs(stage):
            return 1

        # 1. 有未打开的道具/武装箱先开启（星铁可拖拽卡需悬停按住约 0.15s 才能稳定触发「开启」）
        item_tries = self._item_open_tries.get(stage, 0)
        if self._item_done_stage != stage and item_tries < 2:
            slots = self.preparation.scan_bench(self.screen)
            item_slot = next((s for s in slots if s.kind == "item"), None)
            if item_slot is not None:
                self._item_open_tries[stage] = item_tries + 1
                self._equip_done_stage = None
                self._equip_checked_slots.pop(stage, None)
                bx = (BENCH_BOXES[item_slot.index][0] + BENCH_BOXES[item_slot.index][2]) // 2
                by = (BENCH_BOXES[item_slot.index][1] + BENCH_BOXES[item_slot.index][3]) // 2
                CUS_LOGGER.info("悬停按住打开备战格 %d 道具/武装箱", item_slot.index + 1)
                self._click_open_bench_item(bx, by)
                return 1
            self._item_done_stage = stage

        # 2. 本小关尚未逛商店时进入商店买牌/升级（1-1 开局锁定禁用商店，直接跳过）
        if stage == "1-1":
            self._shop_done_stage = stage
        if self._shop_done_stage != stage:
            open_tries = self._shop_open_tries.get(stage, 0)
            gold = self._read_board_gold()
            if (gold is None or gold >= 1) and open_tries < 3:
                self._shop_open_tries[stage] = open_tries + 1
                key_mouse_manager.click_point(*SHOP_TOGGLE_POS)
                key_mouse_manager.wait()
                return 1
            if open_tries >= 3:
                CUS_LOGGER.warning("阶段 %s 连续 %d 次点击商店仍未打开，跳过商店继续推进", stage, open_tries)
            self._shop_done_stage = stage

        # 3. 优先按「连击阿格莱雅/那刻夏」前后台站位规则调度；满员后不再由旧模板引擎逆序替换毁阵
        plan_id, _ = self._current_plan()
        if plan_id == "s2_anaxa_aglaea":
            if self._apply_combo_move():
                return 1
            front_has_unit = any(r.occupied for r in self.match_db.iter_zone("front"))
            if self.match_db.placed >= self.match_db.capacity and front_has_unit:
                ready = True
            else:
                ready = self.prepare_battle()
        else:
            ready = self.prepare_battle()
        if not ready:
            stall = self._prep_stall_tries.get(stage, 0) + 1
            self._prep_stall_tries[stage] = stall
            CUS_LOGGER.warning("阶段 %s 备战第 %d 次未就绪（人数未识别或前排未站稳）", stage, stall)
            if stall == 2:
                self._close_detail_sidebar()
                return 1
            if stall in (3, 4):
                if self.s2_handle_empty_front():
                    return 1
                gold = self._read_board_gold()
                if stage != "1-1" and (gold is None or gold >= 1):
                    self._shop_done_stage = None
            elif stall == 5:
                self._self_diagnose_and_repair(2, stage=stage, reason="备战连续未就绪")
            elif stall >= 6:
                CUS_LOGGER.error("阶段 %s 备战连续 %d 次卡住未能就绪，停机防止死循环", stage, stall)
                self.stop()
            return 0

        # 3.5 备战清理 + 奖励关前卡整十利息卖冗余过渡卡
        if self._cleanup_untracked_bench(stage):
            return 1
        if plan_id == "s2_anaxa_aglaea":
            interest_pick = plan_combo_sell(self.match_db, for_interest=True)
            if interest_pick is not None and self._sell_bench_slot(interest_pick.index, reason="跨档吃息"):
                return 1

        # 4. 数据库驱动配装（采集未知角色名/已穿槽 -> 拆装扳手转核心装 -> 全局合成与定向分配）
        if self._equip_done_stage != stage and self._run_smart_equip(stage):
            return 1

        # 5. 点击右侧推进按钮（出战 / 跳过 / 选择主角 / 幕后茶歇 / 前往补给等）
        rows = self._ocr_box_rows(ADVANCE_BUTTON_BOX, redundancy=25)
        for row in rows:
            txt = row["raw_text"].strip()
            if any(k in txt for k in ADVANCE_KEYWORDS):
                rb = row["box"]
                cx, cy = (rb[0] + rb[2]) // 2, (rb[1] + rb[3]) // 2
                CUS_LOGGER.info("备战完成，点击推进按钮：「%s」", txt)
                self._prep_stall_tries[stage] = 0
                key_mouse_manager.click_point(cx, cy, box=rb)
                key_mouse_manager.wait()
                if any(k in txt for k in ("出战", "跳过")):
                    self.update_state("startbattle")
                else:
                    self.update_state("currency_main")
                return 1
        stall = self._prep_stall_tries.get(stage, 0) + 1
        self._prep_stall_tries[stage] = stall
        CUS_LOGGER.warning("阶段 %s 未识别到推进按钮（第 %d 次）", stage, stall)
        if stall == 3:
            key_mouse_manager.click_point(1816, 749)
            key_mouse_manager.wait()
            self.update_state("startbattle")
            return 1
        if stall == 5:
            self._self_diagnose_and_repair(2, stage=stage, reason="连续未找到推进按钮")
        if stall >= 6:
            CUS_LOGGER.error("阶段 %s 连续 %d 次找不到推进按钮，停机防止死循环", stage, stall)
            self.stop()
        return 0

    def _ensure_guide_tracked(self) -> bool:
        """关店备战画面检查右上角「装备追踪中」状态，未激活则点攻略图标自动应用热门配对追踪。"""
        if self._guide_applied_run:
            return False
        if any("追踪" in str(r.get("raw_text", "")) for r in self._ocr_box_rows([1680, 105, 1915, 155], redundancy=15)):
            self._guide_applied_run = True
            return False
        shop_btn = self._ocr_box_rows([1540, 940, 1700, 1015], redundancy=15)
        if not any("商店" in str(r.get("raw_text", "")) for r in shop_btn):
            return False
        CUS_LOGGER.info("右上角尚未开启推荐配对追踪，点击攻略图标自动应用热门攻略")
        key_mouse_manager.click_point(1610, 58)
        key_mouse_manager.wait()
        return True

    def _cleanup_untracked_bench(self, stage: str) -> bool:
        """攻略追踪生效且阵容就绪后，主动卖掉备战席未追踪的冗余角色换金币并留足开球空位。"""
        if self._proactive_sells.get(stage, 0) >= 2:
            return False
        plan_id, targets = self._current_plan()
        if plan_id == "s2_anaxa_aglaea" and self.match_db.placed >= self.match_db.capacity:
            filler = plan_combo_sell(self.match_db)
            if filler is not None and not filler.is_tracked and (not filler.name or filler.name not in ALL_COMBO_NAMES):
                if self._sell_bench_slot(filler.index, reason="清理非连击冗余占位卡"):
                    self._proactive_sells[stage] = self._proactive_sells.get(stage, 0) + 1
                    return True
        if not self._guide_applied_run:
            return False
        slots = self.preparation.scan_bench(self.screen)
        badges = detect_bench_guide_badges(self.screen)
        untracked = [s for s in slots if s.kind == "character" and not badges[s.index]]
        empty_count = sum(1 for s in slots if s.kind == "empty")
        if not untracked or (len(untracked) < 2 and empty_count >= 3):
            return False
        if self._sell_one_non_target(targets, allow_fallback=False, require_untracked=True):
            self._proactive_sells[stage] = self._proactive_sells.get(stage, 0) + 1
            return True
        return False

    def _collect_orbs(self, stage: str) -> bool:
        """关店状态下连续清理右上角战利品晶球直至清空、满位或达到本关扫盘上限。"""
        if self._orbs_done_stage == stage or self._orb_overflow_stage == stage:
            return False
        if "收起" in self._ocr_box_text([1540, 940, 1700, 1015], redundancy=15):
            return False
        orbs = detect_orb_points(self.screen, limit=12)
        if not orbs:
            self._orbs_done_stage = stage
            return False

        tries = self._orb_tries.get(stage, 0) + 1
        self._orb_tries[stage] = tries

        slots = self.preparation.scan_bench(self.screen)
        if not any(s.kind == "empty" for s in slots):
            _, targets = self._current_plan()
            if self._sell_one_non_target(targets):
                CUS_LOGGER.info("领取右上角晶球前备战已满，先出售 1 名冗余角色腾位")
                if tries >= 2:
                    self._orbs_done_stage = stage
                return True
            if tries >= 2:
                self._orbs_done_stage = stage
                return False

        for sub_pass in range(2):
            if not orbs:
                break
            CUS_LOGGER.info("检测到右上角 %d 颗晶球（第 %d 轮扫盘），逐个点击领取", len(orbs), sub_pass + 1)
            for ox, oy in orbs:
                key_mouse_manager.click_point(ox, oy)
                key_mouse_manager.sleep(0.15)
            key_mouse_manager.wait()
            key_mouse_manager.sleep(0.35)
            self.ts.forward(self.get_screen())
            warn_text = self._ocr_box_text([420, 260, 1510, 570], redundancy=20)
            if "售卖备战区" in warn_text or "备战席已满" in warn_text:
                _, targets = self._current_plan()
                if not self._sell_one_non_target(targets):
                    CUS_LOGGER.warning("阶段 %s 晶球提示备战已满且全为核心，暂停本关续领", stage)
                    self._orb_overflow_stage = stage
                elif tries >= 2:
                    self._orbs_done_stage = stage
                return True
            orbs = detect_orb_points(self.screen, limit=12)

        # 开球后可能新掉道具箱、武装或装备，清空完成标记以便立刻处理；若连点 2 轮仍残留则跳过防死循环
        self._item_done_stage = None
        self._equip_done_stage = None
        self._equip_checked_slots.pop(stage, None)
        if not orbs or tries >= 2:
            if orbs:
                CUS_LOGGER.warning("阶段 %s 晶球已尝试扫盘 %d 轮仍有残留误判点，放弃续领继续备战", stage, tries)
            self._orbs_done_stage = stage
        return True

    def _find_cyan_spotlight_target(self, rows: list[dict]) -> tuple[int, int] | None:
        """存在「提示：」教学浮窗时，剥离气泡本体并定位屏幕上的青框高亮目标中心。"""
        screen = getattr(self, "screen", None)
        if not isinstance(screen, np.ndarray) or screen.shape[:2] != (1080, 1920):
            return None
        hint_rows = [r for r in rows if "提示" in str(r.get("raw_text", "")) and r.get("box")]
        if not hint_rows:
            return None
        b = screen[:, :, 0].astype(np.int16)
        g = screen[:, :, 1].astype(np.int16)
        r = screen[:, :, 2].astype(np.int16)
        mask = ((g > 175) & (b > 175) & (r < 120)).astype(np.uint8)
        for row in hint_rows:
            rb = row["box"]
            x1 = max(0, int(rb[0]) - 25)
            y1 = max(0, int(rb[1]) - 45)
            x2 = min(1920, int(rb[2]) + 125)
            y2 = min(1080, int(rb[3]) + 65)
            mask[y1:y2, x1:x2] = 0
        dilated = cv2.dilate(mask, np.ones((7, 7), dtype=np.uint8))
        _, _, stats, _ = cv2.connectedComponentsWithStats(dilated)
        best: tuple[int, int, int] | None = None
        for stat in stats[1:]:
            sx, sy, sw, sh, area = (int(v) for v in stat)
            if 50 <= sw <= 650 and 50 <= sh <= 450 and area >= 300:
                if best is None or area > best[0]:
                    best = (area, sx + sw // 2, sy + sh // 2)
        return (best[1], best[2]) if best is not None else None

    def _click_blocking_overlay_buttons(self, rows: list[dict]) -> bool:
        """自检全屏文本中常见的阻塞弹窗、教学高亮或推进确认钮，有把握时直接点击脱困。"""
        full_text = " ".join(clean_text(str(r.get("raw_text", "")), 1) for r in rows)
        if "主角光环与奇遇详情" in full_text:
            self.s2_handle_halo_detail()
            return True
        if "重选主角光" in full_text:
            self.s2_handle_halo_tutorial()
            return True
        if "前往重选" in full_text or "可合成列表" in full_text:
            CUS_LOGGER.info("自检点击右上角关闭教学/合成预览弹窗 @(1863,65)")
            key_mouse_manager.click_point(1863, 65)
            key_mouse_manager.wait()
            return True
        if "数据统计" in full_text:
            CUS_LOGGER.info("自检点击右上角关闭「数据统计」弹窗 @(1495,265)")
            key_mouse_manager.click_point(1495, 265)
            key_mouse_manager.wait()
            return True
        modal_keys = (
            "点击空白处继续", "点击空白处关闭", "返回货币战争", "前往结算",
            "继续挑战", "下一步", "跳过", "暂时离开",
        )
        lit_keys = ("确认选择", "确定", "确认")
        for row in rows:
            txt = clean_text(str(row.get("raw_text", "")), 1)
            rb = row.get("box")
            if not rb or not txt:
                continue
            if any(k in txt for k in modal_keys):
                cx, cy = (rb[0] + rb[2]) // 2, (rb[1] + rb[3]) // 2
                CUS_LOGGER.info("自检命中阻塞/继续按钮「%s」@(%d,%d)", txt, cx, cy)
                key_mouse_manager.click_point(cx, cy, box=rb)
                key_mouse_manager.wait()
                return True
            if any(k in txt for k in lit_keys):
                cx, cy = (rb[0] + rb[2]) // 2, (rb[1] + rb[3]) // 2
                lit_box = [max(0, rb[0] - 25), max(0, rb[1] - 8), min(1920, rb[2] + 25), min(1080, rb[3] + 8)]
                if self._box_lit(lit_box):
                    CUS_LOGGER.info("自检点击亮起确认按钮「%s」@(%d,%d)", txt, cx, cy)
                    key_mouse_manager.click_point(cx, cy, box=rb)
                    key_mouse_manager.wait()
                    return True
        spotlight = self._find_cyan_spotlight_target(rows)
        if spotlight is not None:
            CUS_LOGGER.info("自检点击教学「提示：」青框高亮目标 @(%d,%d)", spotlight[0], spotlight[1])
            key_mouse_manager.click_point(*spotlight)
            key_mouse_manager.wait()
            return True
        return False

    def _self_diagnose_and_repair(self, tier: int, *, stage: str = "", reason: str = "") -> bool:
        """画面静止/卡死分级自检修复（1级清遮挡与弹窗 → 2级解锁子步骤并推关 → 3级切焦点+Esc复位态）。"""
        self.ts.forward(self.get_screen())
        rows = self._ocr_box_rows([0, 0, 1920, 1080], redundancy=25)
        preview = " ".join(str(r.get("raw_text", "")).strip() for r in rows[:10] if str(r.get("raw_text", "")).strip())
        CUS_LOGGER.warning(
            "触发 %d 级画面自检修复（原因：%s，阶段：%s）：可见文本=[%s]",
            tier, reason or "未知", stage or self.state, preview[:80],
        )

        # 第 1 级：关闭右栏速合抽屉/暂停遮罩，处理全屏可点继续/结算/教学高亮/亮起确认钮
        if "取消" in self._ocr_box_text([1620, 315, 1900, 415], redundancy=15):
            self._click_text_fast("取消", [1620, 315, 1900, 415], fallback=(1800, 350))
            return True
        if self._click_blocking_overlay_buttons(rows):
            return True
        self._close_detail_sidebar()
        if tier <= 1:
            return False

        # 第 2 级：收起卡住商店（若遮罩导致未实际收起则继续向下尝试推关）、释放子锁并推进「出战」
        if stage and stage != "0-0":
            self._orbs_done_stage = stage
            self._equip_done_stage = stage
            self._shop_done_stage = stage
            self._item_done_stage = stage
        if "收起" in self._ocr_box_text([1540, 940, 1700, 1015], redundancy=15):
            key_mouse_manager.click_point(*SHOP_TOGGLE_POS)
            key_mouse_manager.wait()
            key_mouse_manager.sleep(0.25)
            self.ts.forward(self.get_screen())
            if "收起" not in self._ocr_box_text([1540, 940, 1700, 1015], redundancy=15):
                return True
        for row in self._ocr_box_rows(ADVANCE_BUTTON_BOX, redundancy=25):
            txt = str(row.get("raw_text", "")).strip()
            if any(k in txt for k in ADVANCE_KEYWORDS):
                rb = row["box"]
                key_mouse_manager.click_point((rb[0] + rb[2]) // 2, (rb[1] + rb[3]) // 2, box=rb)
                key_mouse_manager.wait()
                return True
        if any("备战阶段" in str(r.get("raw_text", "")) for r in rows):
            key_mouse_manager.click_point(1816, 749)
            key_mouse_manager.wait()
            return True
        if tier <= 2:
            key_mouse_manager.click_point(960, 540)
            key_mouse_manager.wait()
            return False

        # 第 3 级：重置状态与局部重试计数（实际置前窗口与按 Esc 由外层调度器统一执行）
        if self.state in ("startbattle", "black"):
            self.update_state("currency_main")
        if stage:
            self._prep_stall_tries.pop(stage, None)
            self._equip_attempts.pop(stage, None)
        self._master_tries = 0
        self._special_shop_tries = 0
        self._generic_choice_tries = 0
        return True

    def _recover_from_stagnation(self, stage: str) -> None:
        """兼容入口：执行 1~2 级卡死自检并跳过卡住的子步骤。"""
        self._self_diagnose_and_repair(2, stage=stage, reason="同阶段停滞")

    def _click_open_bench_item(self, bx: int, by: int) -> None:
        """打开备战栏道具/武装箱：先悬停按住 0.15s 点击真实「开启」位置。"""
        rows = self._ocr_box_rows([bx - 80, 915, bx + 80, 995], redundancy=15)
        open_row = next((r for r in rows if "开启" in str(r.get("raw_text", ""))), None)
        if open_row is None:
            key_mouse_manager.click_point(bx, by)
            key_mouse_manager.wait()
            key_mouse_manager.sleep(0.2)
            self.ts.forward(self.get_screen())
            rows = self._ocr_box_rows([bx - 80, 915, bx + 80, 995], redundancy=15)
            open_row = next((r for r in rows if "开启" in str(r.get("raw_text", ""))), None)
        cy = (open_row["box"][1] + open_row["box"][3]) // 2 if open_row else 954
        key_mouse_manager.drag_point(bx, cy, bx, min(1070, cy + 2), duration=0.15)
        key_mouse_manager.wait()

    def _identify_crop(self, x: int, y: int) -> str | None:
        """提取屏幕 (x, y) 中心 44×44 图标并查表识别装备/工具名。"""
        h, w = self.screen.shape[:2]
        if x < 22 or y < 22 or x + 22 > w or y + 22 > h:
            return None
        patch = self.screen[y - 22:y + 22, x - 22:x + 22]
        return self._item_catalog.identify(patch)

    def _close_detail_sidebar(self) -> None:
        """点击备战棋盘空白处安全关闭右侧角色栏或浮窗，严禁按 Esc 触发「是否暂时离开」。"""
        key_mouse_manager.click_point(500, 150)
        key_mouse_manager.sleep(0.15)
        key_mouse_manager.wait()

    def _learn_item_at(self, stage: str, x: int, y: int) -> str | None:
        """右栏开启「装备追踪」时单击物品格会弹出速合抽屉而非悬停提示，仅安全跳过并防误弹。"""
        coord_key = (stage, x, y)
        if coord_key in self._learned_rail_coords or self._guide_applied_run:
            return None
        h, w = self.screen.shape[:2]
        if x < 22 or y < 22 or x + 22 > w or y + 22 > h:
            return None
        crop = self.screen[y - 22:y + 22, x - 22:x + 22].copy()
        if not is_vivid_item_patch(crop):
            return None
        self._learned_rail_coords.add(coord_key)
        key_mouse_manager.click_point(x, y)
        key_mouse_manager.wait()
        key_mouse_manager.sleep(0.22)
        self.ts.forward(self.get_screen())
        tip_box = [1050, max(30, y - 200), 1800, min(800, y + 220)]
        tip_text = clean_unit_name(self._ocr_box_text(tip_box, redundancy=20))
        if "取消" in self._ocr_box_text([1620, 315, 1900, 415], redundancy=15):
            self._click_text_fast("取消", [1620, 315, 1900, 415], fallback=(1800, 350))
        else:
            self._close_detail_sidebar()
        for known in sorted(ALL_KNOWN_ITEM_NAMES, key=len, reverse=True):
            if known in tip_text:
                self._item_catalog.save_item(known, crop)
                CUS_LOGGER.debug("自学习录入 S2 装备图标：%s @(%d,%d)", known, x, y)
                return known
        CUS_LOGGER.debug("右栏悬停浮窗 @(%d,%d) 未匹配已知装备名：%s", x, y, tip_text)
        return None

    def _sync_match_db(self, stage: str) -> None:
        """从当前无遮挡备战画面刷新对局记忆数据库与模板站位优先级。"""
        if "收起" in self._ocr_box_text([1540, 940, 1700, 1015], redundancy=15):
            return
        gold = self._read_board_gold()
        level = self._read_board_level()
        exp = self._read_board_exp()
        hp = self._read_board_hp()
        counts = self._last_team_counts or self.read_team_count()
        self.match_db.sync_economy(stage=stage, gold=gold, level=level, exp=exp, counts=counts, hp=hp)
        if (
            self.match_db.last_combat_hp is not None
            and hp is not None
            and hp < self.match_db.last_combat_hp
        ):
            self.match_db.record_combat_outcome(False, hp)

        slots = self.preparation.scan_bench(self.screen)
        vacancies = self.preparation.read_areas(self.screen)
        bench_badges = detect_bench_guide_badges(self.screen)
        bench_stars = {
            s.index: detect_card_stars(self.screen, BENCH_BOXES[s.index], "bench")
            for s in slots
            if s.kind == "character"
        }
        bench_equips = {
            s.index: detect_card_equip_count(self.screen, BENCH_BOXES[s.index])
            for s in slots
            if s.kind == "character"
        }
        self.match_db.sync_bench(slots, badges=bench_badges, stars_map=bench_stars, equip_map=bench_equips)

        field_stars: dict[tuple[str, int], int | None] = {}
        field_equips: dict[tuple[str, int], int] = {}
        for area, boxes in AREA_BOXES.items():
            vac = set(vacancies.get(area, ()))
            for idx, box in enumerate(boxes):
                if idx in vac:
                    continue
                field_equips[(area, idx)] = detect_card_equip_count(self.screen, box)
                field_stars[(area, idx)] = detect_card_stars(self.screen, box, area)
        members = getattr(self.preparation, "members", {"front": [], "back": []})
        self.match_db.sync_field(
            vacancies,
            members,
            stars_map=field_stars,
            equip_map=field_equips,
        )

        # 把已知真名反向同步回 preparation 前后台优先级表，协助 BattlePreparation 自动站位
        name_to_tpl = {name: tpl for tpl, name in self.match_db.template_names.items() if name and tpl}
        if name_to_tpl and hasattr(self, "front_priority") and hasattr(self, "back_priority"):
            self.front_priority = [name_to_tpl[n] for n in COMBO_FRONT_ORDER if n in name_to_tpl]
            self.back_priority = [name_to_tpl[n] for n in COMBO_BACK_ORDER if n in name_to_tpl]

        tracked_bar = self._guide_applied_run or any(
            "追踪" in str(r.get("raw_text", "")) for r in self._ocr_box_rows([1680, 105, 1915, 155], redundancy=15)
        )
        rail_coords = detect_inventory_rail_positions(self.screen, tracked_bar=tracked_bar)
        rail_items = [(pos, self._identify_crop(pos[0], pos[1])) for pos in rail_coords]
        tool_coords = detect_tool_positions(self.screen, tracked_bar=tracked_bar)
        wrench_pos = None
        furnace_pos = None
        for tpos in tool_coords:
            tname = self._identify_crop(tpos[0], tpos[1])
            if tname in WRENCH_NAMES:
                wrench_pos = tpos
            elif tname == "重铸熔炉":
                furnace_pos = tpos
        self.match_db.sync_inventory(rail_items, wrench_pos=wrench_pos, furnace_pos=furnace_pos)

    def _apply_combo_move(self) -> bool:
        """执行一步基于记忆数据库的「连击阿格莱雅/那刻夏」精准落位或替补换位。"""
        counts = self._last_team_counts or self.read_team_count()
        if counts is None:
            return False
        move_plan = plan_combo_deployment(self.match_db)
        if move_plan is None:
            return False
        src_rec, dst_zone, dst_idx = move_plan
        src_box = BENCH_BOXES[src_rec.index]
        dst_box = AREA_BOXES[dst_zone][dst_idx]
        CUS_LOGGER.debug(
            "连击库调度拖拽：%s(格%d) -> %s第%d格",
            src_rec.name or "角色", src_rec.index + 1, "前台" if dst_zone == "front" else "后台", dst_idx + 1,
        )
        sx, sy = (src_box[0] + src_box[2]) // 2, (src_box[1] + src_box[3]) // 2
        tx, ty = (dst_box[0] + dst_box[2]) // 2, (dst_box[1] + dst_box[3]) // 2
        key_mouse_manager.drag_point(sx, sy, tx, ty)
        key_mouse_manager.sleep(0.45)
        key_mouse_manager.wait()
        self.match_db.record_move("bench", src_rec.index, dst_zone, dst_idx)
        self.match_db.save()
        return True

    def _learn_field_identity_if_unknown(self, stage: str) -> bool:
        """点选一名尚无真名的在场角色，从详情侧栏录入真名与已穿戴装备到数据库。"""
        unknown = next(
            (r for r in self.match_db.field_records() if not r.name and r.detail_checked_stage != stage),
            None,
        )
        if unknown is None:
            return False
        box = AREA_BOXES[unknown.zone][unknown.index]
        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        key_mouse_manager.click_point(cx, cy)
        key_mouse_manager.wait()
        key_mouse_manager.sleep(0.25)
        self.ts.forward(self.get_screen())
        raw_name = self._ocr_box_text([1488, 200, 1745, 255], redundancy=15)
        unit_name = re.sub(r"^(?:试用)+|\d+$", "", clean_unit_name(raw_name))
        worn = []
        for wx in (1646, 1731, 1819):
            wpatch = self.screen[817 - 22:817 + 22, wx - 22:wx + 22]
            if not is_vivid_item_patch(wpatch, min_std=32.0, min_peak=110, min_ratio=0.02):
                continue
            wname = self._identify_crop(wx, 817)
            if wname:
                worn.append(wname)
        self.match_db.record_detail(unknown.zone, unknown.index, unit_name, worn)
        self.match_db.save()
        self._close_detail_sidebar()
        return True

    def _open_slot_equipment_panel(self, zone: str, index: int, *, target_item: str = "") -> bool:
        """点选指定前/后台单位并尝试打开其「推荐装备」弹层。"""
        box = AREA_BOXES[zone][index]
        fx, fy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        self._pending_equip_slot = (zone, index, target_item)
        key_mouse_manager.click_point(fx, fy)
        key_mouse_manager.wait()
        key_mouse_manager.sleep(0.25)
        self.ts.forward(self.get_screen())
        if self._click_text_fast(("装备推荐", "推荐"), [1360, 765, 1610, 900]):
            return True
        self._pending_equip_slot = None
        if "出售" in self._ocr_box_text([1680, 875, 1915, 965], redundancy=15):
            self._close_detail_sidebar()
        return False

    def _run_smart_equip(self, stage: str) -> bool:
        """执行数据库驱动的拆装扳手转移、推荐合成与角色定位定向配装。"""
        if not self.match_db.inventory:
            self._equip_done_stage = stage
            self.match_db.save()
            return False

        # 1. 有尚不认识图标的右栏/工具物品时先读一次悬停详情入库（每个坐标每关至多学一次）
        if any("出战" in str(r.get("raw_text", "")) or "商店" in str(r.get("raw_text", "")) for r in self._ocr_box_rows([1540, 700, 1918, 1020], redundancy=20)):
            tracked_bar = self._guide_applied_run
            for pos in detect_tool_positions(self.screen, tracked_bar=tracked_bar):
                if self._identify_crop(pos[0], pos[1]) is None and self._learn_item_at(stage, pos[0], pos[1]):
                    return True
            for entry in self.match_db.inventory:
                if entry.name is None and self._learn_item_at(stage, entry.x, entry.y):
                    return True

        # 2. 补充采集尚不知道真名的场上角色名与已穿装备
        if self._learn_field_identity_if_unknown(stage):
            return True

        snapshots = self.match_db.as_equip_snapshots()

        # 3. 拆装扳手：优先把前期那刻夏/打工位穿的主C毕业装无损转交给后期阿格莱雅
        wrench_plan = plan_wrench_transfer(
            snapshots,
            has_wrench=(self.match_db.wrench_pos is not None),
            has_risky_emblem_reset=self.match_db.risky_emblem_reset,
        )
        if wrench_plan is not None:
            wkey = (stage, wrench_plan.source_zone, wrench_plan.source_index)
            if wkey not in self._wrench_done_slots and self.match_db.wrench_pos is not None:
                self._wrench_done_slots.add(wkey)
                sbox = AREA_BOXES[wrench_plan.source_zone][wrench_plan.source_index]
                tx, ty = (sbox[0] + sbox[2]) // 2, (sbox[1] + sbox[3]) // 2
                wx, wy = self.match_db.wrench_pos
                CUS_LOGGER.info(
                    "使用拆装扳手 (%d,%d) 从 %s 卸下核心装备转交给 %s",
                    wx, wy, wrench_plan.source_name, wrench_plan.target_name,
                )
                key_mouse_manager.drag_point(wx, wy, tx, ty, duration=0.35)
                key_mouse_manager.sleep(0.3)
                key_mouse_manager.wait()
                self.match_db.record_wrench(wrench_plan.source_zone, wrench_plan.source_index)
                self._equip_checked_slots.pop(stage, None)
                self.match_db.save()
                return True

        # 4. 全局规划所有未满 3 槽的前后台角色（禁止碰已满 3 格角色及主C戴肉装）
        rail_pairs = [((e.x, e.y), e.name) for e in self.match_db.inventory]
        prefs = {k: tuple(v) for k, v in self.match_db.equip_tracking_prefs.items()}
        best_act = plan_best_equip_action(
            snapshots,
            rail_pairs,
            hp=self.match_db.hp,
            is_final=(stage == "3-7"),
            tracked_prefs=prefs,
        )
        checked = self._equip_checked_slots.setdefault(stage, set())
        if best_act is not None:
            slot_key = (best_act.zone, best_act.index)
            target_box = AREA_BOXES[best_act.zone][best_act.index]
            tx, ty = (target_box[0] + target_box[2]) // 2, (target_box[1] + target_box[3]) // 2
            if best_act.kind == "direct_craft":
                if slot_key not in checked:
                    checked.add(slot_key)
                    CUS_LOGGER.info(
                        "打开 %s(%s第%d格) 装备推荐合成目标大件：%s",
                        best_act.unit_name, "前台" if best_act.zone == "front" else "后台",
                        best_act.index + 1, best_act.item_name,
                    )
                    self._open_slot_equipment_panel(best_act.zone, best_act.index, target_item=best_act.item_name)
                    return True
                if best_act.rail_pos is not None and best_act.second_rail_pos is not None:
                    # 先拖右栏更靠下的散件（y 较大）再到更靠上的散件，避免前一件移走导致后一件上移错位
                    first_pos, second_pos = sorted(
                        (best_act.rail_pos, best_act.second_rail_pos),
                        key=lambda p: p[1],
                        reverse=True,
                    )
                    CUS_LOGGER.info(
                        "两件直拖合成：%s (%d,%d)+(%d,%d) → %s(%s第%d格)",
                        best_act.item_name, first_pos[0], first_pos[1], second_pos[0], second_pos[1],
                        best_act.unit_name, "前台" if best_act.zone == "front" else "后台", best_act.index + 1,
                    )
                    key_mouse_manager.drag_point(first_pos[0], first_pos[1], tx, ty, duration=0.35)
                    key_mouse_manager.wait()
                    key_mouse_manager.sleep(0.25)
                    key_mouse_manager.drag_point(second_pos[0], second_pos[1], tx, ty, duration=0.35)
                    key_mouse_manager.wait()
                    key_mouse_manager.sleep(0.25)
                    self.match_db.record_equip(best_act.zone, best_act.index, best_act.item_name)
                    self.match_db.save()
                    return True
            elif best_act.rail_pos is not None:
                ex, ey = best_act.rail_pos
                CUS_LOGGER.info(
                    "精准配装：%s (%d,%d) → %s(%s第%d格)",
                    best_act.item_name, ex, ey, best_act.unit_name,
                    "前台" if best_act.zone == "front" else "后台", best_act.index + 1,
                )
                key_mouse_manager.drag_point(ex, ey, tx, ty, duration=0.35)
                key_mouse_manager.wait()
                key_mouse_manager.sleep(0.25)
                self.match_db.record_equip(best_act.zone, best_act.index, best_act.item_name)
                self.match_db.save()
                return True

        # 5. 若仍有未知图标（如首次见面且仅靠推荐面板可见），仅检查未满 3 格且未查过的在场角色
        if any(e.name is None for e in self.match_db.inventory):
            eligible = [
                (r.zone, r.index)
                for r in sorted(
                    self.match_db.field_records(),
                    key=lambda rec: (rec.zone != "front", -rec.stars, rec.index),
                )
                if r.equip_count < 3 and len(r.worn_items) < 3 and (r.zone, r.index) not in checked
            ]
            if eligible:
                area, idx = eligible[0]
                checked.add((area, idx))
                self._open_slot_equipment_panel(area, idx)
                return True

            # 兜底直拖仅允许在有未满 3 格角色且图标未知时最多试 1 次，不乱拖主C
            drag_tries = self._equip_drag_tries.get(stage, 0)
            fallback_slots = [
                r for r in self.match_db.field_records()
                if r.equip_count < 3 and len(r.worn_items) < 3
            ]
            if fallback_slots and drag_tries < 1:
                self._equip_drag_tries[stage] = drag_tries + 1
                pick = fallback_slots[0]
                ex, ey = self.match_db.inventory[0].x, self.match_db.inventory[0].y
                tbox = AREA_BOXES[pick.zone][pick.index]
                tx, ty = (tbox[0] + tbox[2]) // 2, (tbox[1] + tbox[3]) // 2
                CUS_LOGGER.info("直拖未收录道具 (%d,%d) 到 %s第%d格 (%d,%d)", ex, ey, "前台" if pick.zone == "front" else "后台", pick.index + 1, tx, ty)
                key_mouse_manager.drag_point(ex, ey, tx, ty, duration=0.35)
                key_mouse_manager.wait()
                key_mouse_manager.sleep(0.25)
                return True

        self._equip_done_stage = stage
        self.match_db.save()
        return False

    def _ordered_occupied_slots(self, vacancies: dict, targets: tuple[str, ...]) -> list[tuple[str, int]]:
        """兼容接口：返回前后排未满 3 件装备的可配装格位，主C优先且不再跳过后台辅助。"""
        rank = {name: i for i, name in enumerate(targets)}
        entries: list[tuple[tuple[int, int, int], tuple[str, int]]] = []
        for area in ("front", "back"):
            vac = set(vacancies.get(area, ()))
            for idx in range(len(AREA_BOXES[area])):
                if idx in vac:
                    continue
                rec = self.match_db.slot(area, idx)
                if rec.equip_count >= 3 or len(rec.worn_items) >= 3:
                    continue
                pos_rank = rank.get(rec.name, len(rank) + idx)
                is_carry = 0 if rec.name in ("阿格莱雅", "那刻夏") else (1 if area == "front" else 2)
                entries.append(((is_carry, pos_rank, idx), (area, idx)))
        entries.sort(key=lambda item: item[0])
        return [slot for _, slot in entries]

    def s2_handle_equipment_panel(self):
        """在装备推荐面板按角色定位评分挑选「穿戴」或「合成」，记录入库后再关闭弹层。"""
        self.ts.forward(self.get_screen())
        stage = self._read_stage()
        tries = self._equip_attempts.get(stage, 0)
        pending = self._pending_equip_slot
        unit_rec = self.match_db.slot(pending[0], pending[1]) if pending else None
        target_hint = pending[2] if pending and len(pending) > 2 else ""

        if tries < 4 and "不符合穿戴条件" not in self._ocr_box_text([430, 205, 1500, 474]):
            rows = self._ocr_box_rows([920, 460, 1460, 974], redundancy=20)
            scored_actions: list[tuple[int, str, list[int], str]] = []
            for label in ("穿戴", "合成"):
                for row in rows:
                    if label not in row["raw_text"]:
                        continue
                    rb = row["box"]
                    bx = (rb[0] + rb[2]) // 2
                    by = (rb[1] + rb[3]) // 2
                    col_x = min((1010, 1147, 1284), key=lambda cx: abs(bx - cx))
                    row_y = 656 if by < 795 else 845
                    item_name = self._identify_crop(col_x, row_y) or ""
                    if not item_name and label == "合成":
                        p1 = self._identify_crop(col_x + 59, row_y - 19)
                        p2 = self._identify_crop(col_x + 59, row_y + 18)
                        if p1 and p2:
                            spec = lookup_recipe(p1, p2)
                            if spec is not None:
                                item_name = spec.name
                    if unit_rec is not None:
                        sc = score_item_for_unit(
                            unit_rec.name or "",
                            unit_rec.zone,
                            unit_rec.stars,
                            tuple(unit_rec.worn_items),
                            unit_rec.equip_count,
                            item_name,
                            hp=self.match_db.hp,
                            is_final=(stage == "3-7"),
                            tracked_prefs=tuple(self.match_db.equip_tracking_prefs.get(unit_rec.name, ())),
                        )
                        if target_hint and item_name == target_hint:
                            sc += 100
                        # 已确定装备定位且明确不适合该角色（如给阿格莱雅穿护盾肉装）则禁止点击
                        if item_name and sc <= 0:
                            continue
                    else:
                        sc = 50
                    if label == "穿戴":
                        sc += 8
                    scored_actions.append((sc, label, rb, item_name))

            if scored_actions:
                scored_actions.sort(key=lambda item: item[0], reverse=True)
                _, label, rb, picked_name = scored_actions[0]
                self._equip_attempts[stage] = tries + 1
                if unit_rec is not None:
                    self.match_db.record_equip(unit_rec.zone, unit_rec.index, picked_name)
                    self.match_db.save()
                CUS_LOGGER.debug("推荐面板执行 %s（%s）", label, picked_name or "待确认")
                key_mouse_manager.click_point((rb[0] + rb[2]) // 2, (rb[1] + rb[3]) // 2, box=rb)
                key_mouse_manager.wait()
                return 1

        self._pending_equip_slot = None
        self._close_detail_sidebar()
        key_mouse_manager.sleep(0.1)
        self.ts.forward(self.get_screen())
        if "推荐装备" in self._ocr_box_text([900, 520, 1200, 660], redundancy=20):
            key_mouse_manager.click_point(12, 12)
            key_mouse_manager.wait()
        return 1

    # ---------- S2 特殊节点与弹窗 ----------

    def s2_handle_supply(self):
        self.ts.forward(self.get_screen())
        _, targets = self._current_plan()
        # 若已有卡片被选中且确认已亮起，直接点击确认完成补给
        if self._click_lit_confirm([1495, 949, 1890, 1021], ("确认",)):
            self._supply_click_idx = 0
            return 1
        rows = self._ocr_box_rows([100, 520, 1830, 740], redundancy=25)
        rows.sort(key=lambda r: (r["box"][0] + r["box"][2]) // 2)
        candidates: list[int] = []
        prioritized: list[int] = []
        for row in rows:
            cx = (row["box"][0] + row["box"][2]) // 2
            if any(t in row["raw_text"] for t in targets):
                if all(abs(cx - px) >= 120 for px in prioritized):
                    prioritized.append(cx)
            elif all(abs(cx - px) >= 120 for px in candidates):
                candidates.append(cx)
        ordered = prioritized + [x for x in candidates if all(abs(x - px) >= 120 for px in prioritized)]
        if not ordered:
            ordered = list(SUPPLY_FIVE_X)
        pick_x = ordered[self._supply_click_idx % len(ordered)]
        self._supply_click_idx += 1
        CUS_LOGGER.info("补给阶段点击第 %d 张候选卡片 (x=%d)", min(self._supply_click_idx, len(ordered)), pick_x)
        # y=430 点中补给卡上半部立绘区，避开卡片左下角(y≈620)附带散件图标以防弹出「可合成列表」预览
        key_mouse_manager.click_point(pick_x, 430)
        key_mouse_manager.wait()
        key_mouse_manager.sleep(0.25)
        self.ts.forward(self.get_screen())
        if self._click_lit_confirm([1495, 949, 1890, 1021], ("确认",)):
            self._supply_click_idx = 0
        return 1

    def s2_handle_halo(self):
        self.ts.forward(self.get_screen())
        # 新手关会锁死其他点击并提示刷新当前不适配的光环
        hint_text = self._ocr_box_text([520, 420, 1220, 740], redundancy=25)
        if "刷新试试" in hint_text or "不同的光环适配" in hint_text:
            self._click_text_fast("刷新", [310, 720, 620, 840], fallback=(471, 772))
            return 1

        status = self._ocr_box_text([740, 864, 1170, 918])
        force_drag = self._halo_force_next_drag
        if not force_drag and "尚未选择" not in status and self._click_lit_confirm([760, 913, 1152, 987], ("确认",)):
            return 1

        if not force_drag and ("尚未选择主角角色和光环" in status or not status):
            left_text = self._ocr_box_text([235, 401, 610, 758], redundancy=25)
            right_text = self._ocr_box_text([1320, 401, 1700, 758], redundancy=25)
            pick_right = score_halo_text(right_text) > score_halo_text(left_text)
            if "点击卡片" in hint_text:
                pick_right = False
            pt = (1499, 435) if pick_right else (420, 435)
            key_mouse_manager.click_point(*pt)
            key_mouse_manager.wait()
            key_mouse_manager.sleep(0.25)
            self.ts.forward(self.get_screen())
            status = self._ocr_box_text([740, 864, 1170, 918])

        if force_drag or "尚未选择" in status:
            self._halo_force_next_drag = False
            front_centers = ((803, 618), (905, 600), (1010, 600), (1115, 600))
            sx, sy = front_centers[self._halo_drag_idx % len(front_centers)]
            self._halo_drag_idx += 1
            key_mouse_manager.drag_point(sx, sy, 960, 446)
            key_mouse_manager.wait()
            key_mouse_manager.sleep(0.3)
            self.ts.forward(self.get_screen())
            status = self._ocr_box_text([740, 864, 1170, 918])

        if "尚未选择" not in status:
            if not self._click_lit_confirm([760, 913, 1152, 987], ("确认",)):
                key_mouse_manager.click_point(969, 954)
                key_mouse_manager.wait()
        elif self._halo_drag_idx >= 4:
            self._halo_drag_idx = 0
            self._click_text_fast("刷新", [310, 720, 620, 840], fallback=(471, 772))
        return 1

    def s2_handle_halo_replace(self):
        """替换原有光环确认弹窗：前 3 次取消改换下一前排角色，第 4 次直接确认覆盖。"""
        self.ts.forward(self.get_screen())
        self._halo_replace_tries += 1
        if self._halo_replace_tries <= 3:
            self._halo_force_next_drag = True
            CUS_LOGGER.info("所选前排已有光环，第 %d/3 次取消并改试下一前排", self._halo_replace_tries)
            self._click_text_fast("取消", [550, 625, 935, 725], fallback=(770, 671))
            return 1
        CUS_LOGGER.info("已遍试前排仍需替换光环，直接点击确认覆盖")
        self._click_text_fast(("确认", "确定"), [940, 625, 1370, 725], fallback=(1187, 671))
        self._halo_replace_tries = 0
        self._halo_force_next_drag = False
        return 1

    def s2_handle_halo_tutorial(self):
        """处理 2-1 等节点出现的「重选主角光环」高亮引导浮层，点击左上角光环图标进入详情闭环。"""
        self.ts.forward(self.get_screen())
        CUS_LOGGER.info("检测到「重选主角光环」引导提示，点击左上角高亮光环图标 @(142,174)")
        key_mouse_manager.click_point(142, 174)
        key_mouse_manager.wait()
        key_mouse_manager.sleep(0.35)
        return 1

    def s2_handle_halo_detail(self):
        """关闭「主角光环与奇遇详情」弹窗；若含「前往重选」教学气泡则连续点击两次右上关闭返回备战。"""
        self.ts.forward(self.get_screen())
        has_hint = "前往重选" in self._ocr_box_text([580, 150, 1380, 310], redundancy=20)
        CUS_LOGGER.info("关闭「主角光环与奇遇详情」弹窗%s", "（含前往重选教学气泡）" if has_hint else "")
        key_mouse_manager.click_point(1863, 65)
        key_mouse_manager.wait()
        if has_hint:
            key_mouse_manager.sleep(0.3)
            key_mouse_manager.click_point(1863, 65)
            key_mouse_manager.wait()
        key_mouse_manager.sleep(0.25)
        return 1

    def s2_handle_tea(self):
        self.ts.forward(self.get_screen())
        self._tea_steps += 1
        if self._tea_steps > 15:
            if self.click_text("返回备战界面", box=[1600, 20, 1919, 115], click=True, allow_fail=True):
                self._tea_steps = 0
                return 1
            key_mouse_manager.press("esc")
            key_mouse_manager.wait()
            self._tea_steps = 0
            return 1

        # 确认与收尾按钮必须最优先：点选卡片后会就地出现「确认」，避免再次重选卡片死循环
        confirm_and_exit = (
            ("确认", [1580, 340, 1880, 920]),
            ("开启", [735, 927, 1170, 1032]),
            ("领取", [700, 455, 1875, 1020]),
            ("继续旅程", [700, 455, 1875, 1020]),
            ("结束茶歇", [700, 455, 1875, 1020]),
            ("离开茶歇", [700, 455, 1875, 1020]),
            ("点击空白处继续", [525, 650, 1780, 1065]),
        )
        for text, box in confirm_and_exit:
            if self._click_text_fast(text, box):
                return 1

        # 其次点击已知或首个可用茶歇选项卡片
        card_choices = (
            ("美餐时间", [1120, 340, 1856, 560]),
            ("使出苍龙", [1120, 340, 1860, 760]),
            ("圆硬币咖啡", [1120, 340, 1860, 760]),
        )
        for text, box in card_choices:
            if self._click_text_fast(text, box):
                return 1

        for row in self._ocr_box_rows([1120, 340, 1650, 760], redundancy=20):
            txt = str(row.get("raw_text", "")).strip()
            if txt and "返回" not in txt:
                rb = row["box"]
                key_mouse_manager.click_point((rb[0] + rb[2]) // 2, (rb[1] + rb[3]) // 2, box=rb)
                key_mouse_manager.wait()
                return 1

        for text, box in (
            ("摇一摇", [1350, 785, 1780, 880]),
            ("聊一聊", [180, 785, 574, 880]),
            ("返回", [1120, 340, 1860, 860]),
        ):
            if self._click_text_fast(text, box):
                return 1
        key_mouse_manager.click_point(960, 900)
        key_mouse_manager.wait()
        return 1

    def s2_handle_encounter(self):
        self.ts.forward(self.get_screen())
        # 动态定位遭遇卡片中心 X（同时兼容 S2 两卡居中与三卡布局）
        title_rows = [
            row
            for row in self._ocr_box_rows([150, 260, 1770, 720], redundancy=25)
            if "遭遇其" in str(row.get("raw_text", ""))
            or any(k in str(row.get("raw_text", "")) for k in ("普通遭遇", "困难挑战", "稳妥迎战"))
        ]
        if title_rows:
            cards = sorted(title_rows, key=lambda r: (r["box"][0] + r["box"][2]) // 2)
            xs = [(r["box"][0] + r["box"][2]) // 2 for r in cards]
            cols = [
                [max(120, cx - 220), 260, min(1800, cx + 220), 820]
                for cx in xs
            ]
        else:
            cols = [[195, 250, 700, 860], [700, 250, 1225, 860], [1225, 250, 1740, 860]]
            xs = [(c[0] + c[2]) // 2 for c in cols]
        texts = [self._ocr_box_text(c, redundancy=25) for c in cols]
        idx = pick_encounter_index(texts)
        cx = xs[idx]
        key_mouse_manager.click_point(cx, 500)
        key_mouse_manager.wait()
        key_mouse_manager.sleep(0.25)
        self.ts.forward(self.get_screen())
        if not self._click_lit_confirm([740, 845, 1180, 960], ("选择", "确认选择", "确认", "挑战")):
            CUS_LOGGER.warning("遭遇节点选中 X=%d 后未检测到亮起确认按钮，按居中位置兜底", cx)
            key_mouse_manager.click_point(960, 899)
            key_mouse_manager.wait()
        return 1

    def s2_handle_svarog(self):
        self.ts.forward(self.get_screen())
        # 末卡同时兼容 3 列和 4 列布局
        key_mouse_manager.click_point(1444, 285)
        key_mouse_manager.wait()
        key_mouse_manager.sleep(0.2)
        self.ts.forward(self.get_screen())
        if not self._click_lit_confirm([1280, 541, 1700, 626], ("确认选择",)):
            key_mouse_manager.click_point(1492, 585)
            key_mouse_manager.wait()
        return 1

    def s2_handle_board_effect(self):
        self.ts.forward(self.get_screen())
        cols = [[440, 270, 840, 440], [845, 270, 1245, 440], [1255, 270, 1660, 440]]
        texts = [self._ocr_box_text(c, redundancy=25) for c in cols]
        best = max(range(3), key=lambda i: score_investment_text(texts[i], False, ()))
        cx = (cols[best][0] + cols[best][2]) // 2
        key_mouse_manager.click_point(cx, 350)
        key_mouse_manager.wait()
        key_mouse_manager.sleep(0.2)
        self.ts.forward(self.get_screen())
        if not self._click_lit_confirm([1300, 550, 1680, 640], ("确认选择",)):
            key_mouse_manager.click_point(1492, 599)
            key_mouse_manager.wait()
        return 1

    def s2_handle_dance(self):
        self.ts.forward(self.get_screen())
        details = self._ocr_box_rows([440, 380, 1650, 440], redundancy=25)
        xs = sorted({(r["box"][0] + r["box"][2]) // 2 for r in details})[:2] or [760, 1140]
        for x in xs:
            key_mouse_manager.click_point(x, 320)
            key_mouse_manager.sleep(0.15)
        key_mouse_manager.wait()
        self.ts.forward(self.get_screen())
        if not self._click_lit_confirm([1280, 541, 1700, 626], ("确认选择",)):
            key_mouse_manager.click_point(1487, 576)
            key_mouse_manager.wait()
        return 1

    def s2_handle_ranger(self):
        self.ts.forward(self.get_screen())
        key_mouse_manager.click_point(973, 355)
        key_mouse_manager.wait()
        key_mouse_manager.sleep(0.2)
        self.ts.forward(self.get_screen())
        if not self._click_lit_confirm([1280, 541, 1700, 626], ("确认选择",)):
            key_mouse_manager.click_point(1492, 598)
            key_mouse_manager.wait()
        return 1

    def s2_handle_master(self):
        """处理「从师」节点：优先按当前阵容目标选师父并点击右侧确认选择。"""
        self.ts.forward(self.get_screen())
        confirm_box = [1260, 480, 1710, 635]
        if self._click_lit_confirm(confirm_box, ("确认选择", "确认")):
            self._master_tries = 0
            return 1

        self._master_tries += 1
        if self._master_tries > 6:
            CUS_LOGGER.error("从师选择连续 %d 次未确认，停机防止死循环", self._master_tries)
            self.stop()
            return 0

        _, targets = self._current_plan()
        rows = self._ocr_box_rows([390, 320, 1730, 395], redundancy=25)
        best_x = None
        best_score = (-1, -1)
        for row in rows:
            box = row.get("box")
            if not box:
                continue
            cx = (box[0] + box[2]) // 2
            name = clean_unit_name(str(row.get("raw_text", "")))
            if not name:
                continue
            if name in targets:
                score = (1000 - targets.index(name), -abs(cx - 1040))
            elif 500 <= cx <= 1450:
                score = (10, -abs(cx - 1040))
            else:
                score = (0, -abs(cx - 1040))
            if score > best_score:
                best_score = score
                best_x = cx

        if best_x is None:
            fallback_xs = (544, 824, 1105, 1386)
            pick_x = fallback_xs[(self._master_tries - 1) % len(fallback_xs)]
        else:
            pick_x = min(max(480, best_x), 1640)

        key_mouse_manager.click_point(pick_x, 270)
        key_mouse_manager.wait()
        key_mouse_manager.sleep(0.25)
        self.ts.forward(self.get_screen())
        if self._click_lit_confirm(confirm_box, ("确认选择", "确认")):
            self._master_tries = 0
            return 1

        hint = self._ocr_box_text([1320, 455, 1665, 510], redundancy=20)
        if "请选择" not in hint or self._master_tries >= 2:
            if self._click_text_fast(("确认选择", "确认"), confirm_box, fallback=(1492, 536)):
                self._master_tries = 0
        return 1

    def s2_handle_item_drawer(self):
        """处理简易武装箱、星徽秘典等「请选择1个」顶部抽屉选卡，按阵容自动拾取。"""
        self.ts.forward(self.get_screen())
        plan_id, _ = self._current_plan()
        rows = self._ocr_box_rows([460, 310, 1600, 368], redundancy=25)
        candidates = [
            row
            for row in rows
            if row.get("box") and "查看详情" not in str(row.get("raw_text", ""))
        ]
        if candidates:
            best = max(
                candidates,
                key=lambda r: (
                    score_item_choice(str(r.get("raw_text", "")), plan_id),
                    -(abs((r["box"][0] + r["box"][2]) // 2 - 1020)),
                ),
            )
            rb = best["box"]
            cx = (rb[0] + rb[2]) // 2
            CUS_LOGGER.info("武装/星徽选择拾取「%s」(x=%d)", best["raw_text"], cx)
        else:
            cx = 885
        key_mouse_manager.click_point(cx, 265)
        key_mouse_manager.wait()
        key_mouse_manager.sleep(0.25)
        self.ts.forward(self.get_screen())
        self._click_lit_confirm([760, 480, 1700, 990], ("确认选择", "确认"))
        # 新到手的装备/星徽立即重新放开本小关穿装检查
        self._equip_done_stage = None
        self._equip_checked_slots.clear()
        return 1

    def s2_handle_strategy_guide(self):
        """处理右上角「货币战争 攻略」页面：匹配或选用最高热度推荐配对一键追踪后返回。"""
        self.ts.forward(self.get_screen())
        self._guide_tries += 1
        if self._guide_tries > 4:
            self._guide_applied_run = True
            key_mouse_manager.press("esc")
            key_mouse_manager.wait()
            return 1

        status_text = self._ocr_box_text([1660, 220, 1900, 900], redundancy=20)
        if "取消应用" in status_text or "应用中" in status_text:
            self._guide_applied_run = True
            CUS_LOGGER.info("当前推荐攻略已处于应用追踪状态，返回备战")
            key_mouse_manager.press("esc")
            key_mouse_manager.wait()
            return 1

        plan_id, _ = self._current_plan()
        keywords_map = {
            "s2_anaxa_aglaea": ("阿格莱雅", "那刻夏", "连击", "奇兽花园"),
            "s2_anaxa": ("那刻夏", "连击"),
            "s2_dot": ("持续", "海瑟音", "卡芙卡"),
            "s2_followup": ("飞霄", "追击"),
            "s2_xianzhou": ("仙舟", "景元"),
        }
        title_boxes = (
            [170, 235, 1200, 295],
            [170, 465, 1200, 525],
            [170, 690, 1200, 750],
        )
        apply_boxes = (
            ([1680, 345, 1890, 415], (1784, 377)),
            ([1680, 575, 1890, 645], (1784, 606)),
            ([1680, 800, 1890, 875], (1784, 835)),
        )
        wanted = keywords_map.get(plan_id, ())
        pick_idx = 0
        for idx, tbox in enumerate(title_boxes):
            title = self._ocr_box_text(tbox, redundancy=20)
            if wanted and any(k in title for k in wanted):
                pick_idx = idx
                break
        box, fallback = apply_boxes[pick_idx]
        CUS_LOGGER.info("在热门攻略页一键应用第 %d 套推荐阵容追踪（计划：%s）", pick_idx + 1, plan_id)
        self._click_text_fast("应用攻略", box, fallback=fallback)
        key_mouse_manager.sleep(0.4)
        self._guide_applied_run = True
        self.match_db.guide_applied = True
        self.match_db.save()
        key_mouse_manager.press("esc")
        key_mouse_manager.wait()
        return 1

    def s2_handle_equip_tracking(self):
        """应用带推荐装备的攻略后弹出的「装备追踪」二级确认窗：自动勾选核心「追踪」并关闭。"""
        self.ts.forward(self.get_screen())
        rows = self._ocr_box_rows([1170, 380, 1340, 845], redundancy=20)
        clicked = 0
        for row in sorted(rows, key=lambda r: r["box"][1]):
            cleaned = clean_text(str(row.get("raw_text", "")), 1)
            if cleaned == "追踪":
                rb = row["box"]
                key_mouse_manager.click_point((rb[0] + rb[2]) // 2, (rb[1] + rb[3]) // 2, box=rb)
                key_mouse_manager.sleep(0.15)
                clicked += 1
                if clicked >= 4:
                    break
        if clicked:
            CUS_LOGGER.info("在「装备追踪」弹窗自动勾选 %d 位核心角色优选装备追踪", clicked)
        self._guide_applied_run = True
        self.match_db.guide_applied = True
        self.match_db.save()
        key_mouse_manager.click_point(1343, 269)
        key_mouse_manager.sleep(0.25)
        key_mouse_manager.wait()
        self.ts.forward(self.get_screen())
        if "装备追踪" in self._ocr_box_text([480, 225, 760, 315], redundancy=15):
            key_mouse_manager.press("esc")
            key_mouse_manager.wait()
        return 1

    def s2_handle_special_shop(self):
        self.ts.forward(self.get_screen())
        self._special_shop_tries += 1
        if self._special_shop_tries > 6:
            CUS_LOGGER.error("特殊商店连续触发 %d 次仍未关闭，停机防止死循环", self._special_shop_tries)
            self.stop()
            return 0
        header = self._ocr_box_text([640, 45, 1460, 115], redundancy=20)
        if "开拓商店" in header:
            promo = self._ocr_box_text([820, 340, 1200, 420], redundancy=20)
            if "星穹列车" in promo or "跃迁" in promo:
                CUS_LOGGER.info("领取开拓商店 0 费「星穹列车，跃迁！」并收起抽屉")
                key_mouse_manager.click_point(960, 290)
                key_mouse_manager.wait()
                key_mouse_manager.sleep(0.25)
            key_mouse_manager.click_point(960, 520)
            key_mouse_manager.wait()
            return 1
        if self._click_text_fast(("离开", "返回", "关闭"), [1200, 20, 1900, 1040]):
            return 1
        key_mouse_manager.press("esc")
        key_mouse_manager.wait()
        return 1

    def s2_handle_generic_choice(self):
        """通用选择兜底：同时兼容底部确认与抽屉右上确认，并限制最大重试次数。"""
        self.ts.forward(self.get_screen())
        confirm_boxes = ([700, 860, 1700, 1050], [1260, 480, 1720, 650])
        for box in confirm_boxes:
            if self._click_lit_confirm(box, ("确认选择", "确认")):
                self._generic_choice_tries = 0
                return 1

        self._generic_choice_tries += 1
        if self._generic_choice_tries > 10:
            CUS_LOGGER.error("通用节点选择连续 %d 次未确认，停机防止死循环", self._generic_choice_tries)
            self.stop()
            return 0

        candidates = (
            (684, 398),
            (960, 398),
            (544, 270),
            (824, 270),
            (1105, 270),
            (1386, 270),
        )
        cx, cy = candidates[(self._generic_choice_tries - 1) % len(candidates)]
        key_mouse_manager.click_point(cx, cy)
        key_mouse_manager.wait()
        key_mouse_manager.sleep(0.2)
        self.ts.forward(self.get_screen())
        for box in confirm_boxes:
            if self._click_lit_confirm(box, ("确认选择", "确认")):
                self._generic_choice_tries = 0
                return 1

        if "确认" in self._ocr_box_text([1260, 480, 1720, 650], redundancy=20):
            key_mouse_manager.click_point(1492, 536)
        else:
            key_mouse_manager.click_point(960, 985)
        key_mouse_manager.wait()
        return 1

    def s2_handle_match_result(self):
        """结算翻页、记账、更新轮换计划状态并准备下一局。"""
        self.ts.forward(self.get_screen())
        title = self._ocr_box_text([410, 135, 1350, 410], redundancy=30)
        if any(k in title for k in ("对局胜利", "晋升成功", "挑战成功", "触发晋升赛")):
            self._match_result = "win"
        elif any(k in title for k in ("对局失败", "对局未完成", "晋升失败")):
            self._match_result = "loss"

        if self._click_text_fast("返回货币战争", [500, 735, 1900, 1055]):
            plan_id, _ = self._current_plan()
            outcome = self._match_result or "unknown"
            state = self._load_s2_state()
            state.setdefault("history", []).append({
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "plan": plan_id,
                "result": outcome,
            })
            if outcome == "win" and plan_id not in state.setdefault("done_plans", []):
                if any(item.get("id") == plan_id for item in self.s2_cfg.get("rotation", [])):
                    state["done_plans"].append(plan_id)
            self._save_s2_state(state)
            self._s2_run_count += 1
            self.run_history.finish_run()
            self.match_db.reset_run()
            self._reset_run_stage_flags()
            self.update_state("currency_main")
            CUS_LOGGER.info("第 %d 局结算完成，结果=%s，计划=%s", self._s2_run_count, outcome, plan_id)
            return 1

        for btn in ("下一步", "下一页", "前往结算"):
            if self._click_text_fast(btn, [650, 785, 1670, 1030]):
                return 1
        self.update_state("currency_main")
        key_mouse_manager.click_point(960, 900)
        key_mouse_manager.wait()
        return 1
