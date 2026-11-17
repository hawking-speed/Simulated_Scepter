import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from core.currency.automation import SimulatedCurrency
from core.currency.preparation import AREA_BOXES, BattlePreparation, BenchSlot
from core.currency.run_history import CurrencyRunHistory
from core.currency.settings import load_s2_settings


class CurrencyS2AutomationTests(unittest.TestCase):
    @staticmethod
    def make_s2_currency(tmp_path=None):
        c = object.__new__(SimulatedCurrency)
        c._stop = False
        c.state = "currency_main"
        c.exit_requested = False
        c.exit_started = False
        c.exit_plane = None
        c.set_exit_plane = None
        c.has_fresh = False
        c.screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        c.get_screen = Mock(return_value=c.screen)
        c.ts = Mock()
        c.ts.find_with_box.return_value = []
        c.ts.ocr_one_row.return_value = ""
        c.update_state = Mock(side_effect=lambda s: setattr(c, "state", s))
        c.stop = Mock(side_effect=lambda *a, **k: setattr(c, "_stop", True))
        c.click_text = Mock(return_value=False)
        c.preparation = Mock()
        c.preparation.scan_bench.return_value = [
            BenchSlot(i, "", "empty", ()) for i in range(9)
        ]
        c.preparation.read_areas.return_value = {"front": [1, 2, 3], "back": [0, 1, 2, 3, 4, 5]}
        c.investment_tracker = Mock()
        c.run_history = CurrencyRunHistory(
            record_path=str((tmp_path or Path(tempfile.gettempdir())) / "hist.txt")
        )
        c.ENVIR_BOXES = [[266, 375, 610, 413], [780, 375, 1134, 414], [1314, 373, 1645, 413]]
        c.BLESS_BOXES = [[298, 472, 608, 513], [783, 472, 1095, 513], [1298, 472, 1605, 513]]
        c.init_s2()
        db_file = (tmp_path or Path(tempfile.gettempdir())) / f"test_currency_match_db_{id(c)}.json"
        c.match_db = type(c.match_db)(db_path=db_file)
        c._reset_run_stage_flags()
        if tmp_path is not None:
            c._s2_state_path = lambda: tmp_path / "currency_s2_state.json"
        return c

    def test_load_s2_settings_parses_custom_rotation_and_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            cfg_path = Path(d) / "currency.json"
            cfg_path.write_text(json.dumps({
                "s2": {
                    "mode": "normal",
                    "rank": "current",
                    "max_runs": 10,
                    "rotation": [{"id": "dot_4", "targets": ["海瑟音", "卡芙卡"]}],
                }
            }), encoding="utf-8")
            cfg = load_s2_settings(cfg_path)
            self.assertEqual(cfg["mode"], "normal")
            self.assertEqual(cfg["rank"], "current")
            self.assertEqual(cfg["max_runs"], 10)
            self.assertEqual(len(cfg["rotation"]), 1)

    @patch("core.currency.s2_actions.ZZZ_LOCK_PATH")
    @patch("core.currency.s2_actions.key_mouse_manager")
    def test_home_stops_when_zzz_lock_exists_or_max_runs_reached(self, manager, lock_path):
        c = self.make_s2_currency()
        lock_path.exists.return_value = True
        self.assertEqual(c.s2_handle_home(), 0)
        c.stop.assert_called_once()
        manager.click_point.assert_not_called()

        c = self.make_s2_currency()
        lock_path.exists.return_value = False
        c._s2_run_count = c.s2_cfg["max_runs"]
        self.assertEqual(c.s2_handle_home(), 0)
        c.stop.assert_called_once()

    @patch("core.currency.s2_actions.key_mouse_manager")
    def test_modes_resumes_unfinished_match_before_starting_new(self, manager):
        c = self.make_s2_currency()
        c._ocr_box_text = Mock(return_value="继续进度放弃并结算")
        c.click_text = Mock(return_value=True)

        self.assertEqual(c.s2_handle_modes(), 1)
        c.click_text.assert_called_once()
        self.assertEqual(c.state, "currency_main")
        manager.click_point.assert_not_called()

    @patch("core.currency.s2_actions.key_mouse_manager")
    def test_rank_steps_down_when_above_lowest_then_starts(self, manager):
        c = self.make_s2_currency()
        c.s2_cfg["rank"] = "lowest"
        c._read_int = Mock(return_value=4)
        self.assertEqual(c.s2_handle_rank(), 1)
        self.assertEqual(manager.click_point.call_count, 3)

        manager.reset_mock()
        c._read_int = Mock(return_value=1)
        c.click_text = Mock(return_value=True)
        self.assertEqual(c.s2_handle_rank(), 1)
        c.click_text.assert_called_once()
        self.assertEqual(c.state, "difficulty_select")
        self.assertTrue(c.run_history.is_active)

    @patch("core.currency.s2_actions.key_mouse_manager")
    def test_manage_shop_buys_target_then_closes_when_done(self, manager):
        c = self.make_s2_currency()
        c._read_stage = Mock(return_value="1-2")
        c._read_board_gold = Mock(return_value=5)
        c._read_board_level = Mock(return_value=4)
        c.read_team_count = Mock(return_value=(3, 4))
        c._read_xp_cost = Mock(return_value=4)
        c._read_refresh_cost = Mock(return_value=2)
        c._ocr_box_rows = Mock(side_effect=[
            [{"raw_text": "那刻夏1", "box": [370, 290, 500, 320]}],
            [], [], [], [],
        ])

        self.assertEqual(c.s2_manage_shop(), 1)
        # 购买第 1 列商店卡
        manager.click_point.assert_called_once_with(367 + 106, 194)
        self.assertEqual(c._shop_ops["1-2"], 1)

        # 预算和刷新耗尽时关闭商店并记录本阶段逛店完成
        manager.reset_mock()
        c._read_board_gold = Mock(return_value=0)
        c._ocr_box_rows = Mock(return_value=[])
        self.assertEqual(c.s2_manage_shop(), 1)
        manager.click_point.assert_called_once_with(1620, 980)
        self.assertEqual(c._shop_done_stage, "1-2")

    @patch("core.currency.s2_actions.key_mouse_manager")
    def test_prepare_and_advance_clicks_fight_and_enters_battle_state(self, manager):
        c = self.make_s2_currency()
        c._read_stage = Mock(return_value="1-3")
        c._shop_done_stage = "1-3"
        c._equip_done_stage = "1-3"
        c.prepare_battle = Mock(return_value=1)
        c._ocr_box_rows = Mock(return_value=[{"raw_text": "出战", "box": [1780, 730, 1850, 770]}])

        self.assertEqual(c.s2_prepare_and_advance(), 1)
        self.assertEqual(c.state, "startbattle")
        manager.click_point.assert_called_once_with(1815, 750, box=[1780, 730, 1850, 770])

    @patch("core.currency.s2_actions.key_mouse_manager")
    def test_halo_drags_front_unit_and_picks_safer_side(self, manager):
        c = self.make_s2_currency()
        c._ocr_box_text = Mock(side_effect=[
            "",                           # 新手引导气泡不存在
            "尚未选择主角角色和光环",     # 初始尚未选卡也未选角色
            "角色开局献祭并离场",         # 左卡高风险
            "前台强度提高50%且恢复能量",   # 右卡安全增益
            "尚未选择角色作为主角",       # 选卡后仍需拖拽主角
            "",                           # 拖拽后状态清空，允许点击确认
        ])
        c._click_lit_confirm = Mock(return_value=True)

        self.assertEqual(c.s2_handle_halo(), 1)
        manager.click_point.assert_called_once_with(1499, 435)
        manager.drag_point.assert_called_once_with(803, 618, 960, 446)
        c._click_lit_confirm.assert_called_once()

    @patch("core.currency.s2_actions.key_mouse_manager")
    def test_stage_level_preserve_ascii_and_shop_uses_cached_counts(self, _manager):
        c = self.make_s2_currency()
        c.ts.find_with_box.side_effect = [
            [{"raw_text": "1-9", "box": [450, 70, 490, 90]}],
            [{"raw_text": "Lv.6", "box": [275, 900, 320, 920]}],
        ]
        self.assertEqual(c._read_stage(), "1-9")
        self.assertEqual(c._read_board_level(), 6)

        c._last_team_counts = (5, 6)
        c.read_team_count = Mock(side_effect=AssertionError("商店遮挡时不重读人数区"))
        c._read_stage = Mock(return_value="1-9")
        c._read_board_gold = Mock(return_value=0)
        c._read_board_level = Mock(return_value=6)
        c._read_xp_cost = Mock(return_value=4)
        c._read_refresh_cost = Mock(return_value=2)
        c._ocr_box_rows = Mock(return_value=[])
        self.assertEqual(c.s2_manage_shop(), 1)
        self.assertEqual(c._shop_done_stage, "1-9")

    @patch("core.currency.s2_actions.key_mouse_manager")
    def test_match_result_updates_rotation_done_plans(self, _manager):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            c = self.make_s2_currency(tmp)
            c.s2_cfg["rotation"] = [
                {"id": "bond_dot", "targets": ["海瑟音", "卡芙卡"]},
                {"id": "bond_follow", "targets": ["飞霄", "托帕"]},
            ]
            c.run_history.start_run()
            c._ocr_box_text = Mock(return_value="对局胜利")
            c.click_text = Mock(side_effect=lambda text, **kw: text == "返回货币战争")

            self.assertEqual(c.s2_handle_match_result(), 1)
            state = json.loads((tmp / "currency_s2_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["done_plans"], ["bond_dot"])
            self.assertEqual(state["history"][0]["result"], "win")
            self.assertEqual(c._current_plan()[0], "bond_follow")

    def test_read_areas_detects_s2_empty_slots_without_false_positive(self):
        prep = BattlePreparation.__new__(BattlePreparation)
        prep.area_tiles = {
            area: [np.zeros((10, 10, 3), dtype=np.uint8) for _ in boxes]
            for area, boxes in AREA_BOXES.items()
        }
        frame = np.full((1080, 1920, 3), (107, 36, 46), dtype=np.uint8)
        # front[0] 和 back[0] 放入高方差噪点模拟角色卡；back[1] 为 S2 紫底低方差空位
        rng = np.random.default_rng(7)
        for box in (AREA_BOXES["front"][0], AREA_BOXES["back"][0], AREA_BOXES["back"][2]):
            left, top, right, bottom = box
            frame[top:bottom, left:right] = rng.integers(
                0, 255, size=(bottom - top, right - left, 3), dtype=np.uint8
            )

        scores = iter([0.2, 0.1, 0.1, 0.1, 0.2, 0.77, 0.77, 0.77, 0.77, 0.77])
        with patch("core.currency.preparation.match_score", side_effect=lambda *_: next(scores)):
            vacancies = prep.read_areas(frame)

        self.assertEqual(vacancies["front"], [])
        self.assertEqual(vacancies["back"], [1, 3, 4, 5])

    @patch("core.currency.automation.key_mouse_manager")
    @patch("core.currency.s2_actions.key_mouse_manager")
    def test_unlock_quest_nav_and_terminal_interact(self, _s2_km, auto_km):
        c = self.make_s2_currency()
        c.click_text = Mock(side_effect=lambda text, **kw: text in ("前往", "传送"))
        self.assertEqual(c.s2_handle_unlock_quest(), 1)
        self.assertEqual(c.s2_handle_unlock_nav(), 1)
        c._ocr_box_text = Mock(return_value="货币战争")
        c.goto_currency()
        auto_km.press.assert_called_with("interact")

    @patch("core.currency.s2_actions.key_mouse_manager")
    def test_unlock_guard_stops_infinite_loop(self, _km):
        c = self.make_s2_currency()
        c.click_text = Mock(return_value=True)
        c._unlock_steps = 8
        self.assertEqual(c.s2_handle_unlock_quest(), 0)
        c.stop.assert_called_once()

    @patch("core.currency.s2_actions.key_mouse_manager")
    def test_click_text_fast_uses_cached_ocr_row_first(self, manager):
        c = self.make_s2_currency()
        c.ts.find_with_box.return_value = [{"raw_text": "传送", "box": [1720, 955, 1810, 990]}]
        self.assertEqual(c.s2_handle_unlock_nav(), 1)
        manager.click_point.assert_called_once_with(1765, 972, box=[1720, 955, 1810, 990])
        c.click_text.assert_not_called()

    @patch("core.currency.s2_actions.key_mouse_manager")
    def test_special_shop_claims_zero_cost_kaituo_and_clicks_outside_to_close(self, manager):
        c = self.make_s2_currency()
        c._ocr_box_text = Mock(side_effect=["【开拓商店】", "星穹列车，跃迁！0"])
        self.assertEqual(c.s2_handle_special_shop(), 1)
        self.assertEqual(manager.click_point.call_args_list[0].args, (960, 290))
        self.assertEqual(manager.click_point.call_args_list[1].args, (960, 520))

    @patch("core.currency.s2_actions.key_mouse_manager")
    def test_special_shop_guard_stops_infinite_loop(self, _manager):
        c = self.make_s2_currency()
        c._special_shop_tries = 6
        self.assertEqual(c.s2_handle_special_shop(), 0)
        c.stop.assert_called_once()

    @patch("core.currency.s2_actions.key_mouse_manager")
    def test_prepare_returns_from_supply_peek_and_caps_shop_retries(self, manager):
        c = self.make_s2_currency()
        c.ts.find_with_box.return_value = [{"raw_text": "返回补给阶段", "box": [1700, 50, 1880, 80]}]
        self.assertEqual(c.s2_prepare_and_advance(), 1)
        manager.click_point.assert_called_once_with(1790, 65, box=[1700, 50, 1880, 80])

        c.ts.find_with_box.return_value = []
        c._read_stage = Mock(return_value="2-3")
        c._item_done_stage = "2-3"
        c._shop_open_tries["2-3"] = 3
        c.prepare_battle = Mock(return_value=0)
        c.s2_prepare_and_advance()
        self.assertEqual(c._shop_done_stage, "2-3")

    @patch("core.currency.s2_actions.key_mouse_manager")
    def test_supply_clicks_live_ocr_card_center_not_gap(self, manager):
        c = self.make_s2_currency()
        c._click_lit_confirm = Mock(side_effect=[False, True])
        c.ts.find_with_box.return_value = [
            {"raw_text": "折叠小刀", "box": [200, 680, 312, 720]},
            {"raw_text": "和平手枪", "box": [555, 680, 667, 720]},
        ]
        self.assertEqual(c.s2_handle_supply(), 1)
        manager.click_point.assert_called_once_with(256, 430)
        self.assertEqual(c._supply_click_idx, 0)

    @patch("core.currency.automation.time.monotonic")
    @patch("core.currency.automation.cv2")
    @patch("core.currency.automation.key_mouse_manager")
    @patch("core.currency.automation.ZZZ_LOCK_PATH")
    def test_loop_watchdog_stops_pingpong_noncombat_triggers(self, lock_path, _km, _cv2, monotonic):
        lock_path.exists.return_value = False
        now = [0.0]

        def step_time():
            now[0] += 5.0
            return now[0]

        monotonic.side_effect = step_time
        c = self.make_s2_currency()
        c._action_seq = 0
        c._read_stage = Mock(return_value="2-4")

        def pingpong():
            c._action_seq += 1
            return ("出战前准备" if c._action_seq % 2 else "商店开启"), 1

        c.run_static = Mock(side_effect=pingpong)
        c.loop()
        c.stop.assert_called_once()
        self.assertEqual(c._action_seq, 30)

    @patch("core.currency.s2_actions.key_mouse_manager")
    def test_sell_failure_marks_shop_done_and_pause_modal_exits_safely(self, manager):
        c = self.make_s2_currency()
        c._current_plan = Mock(return_value=("s2_anaxa", ("刻律德菈",)))
        c._read_stage = Mock(return_value="2-4")
        c._read_board_gold = Mock(return_value=13)
        c._read_board_level = Mock(return_value=7)
        c._last_team_counts = (7, 7)
        c._read_xp_cost = Mock(return_value=4)
        c._read_refresh_cost = Mock(return_value=2)
        c.preparation.scan_bench.return_value = [
            BenchSlot(i, f"char_{i}", "character", ("front",)) for i in range(9)
        ]
        c._ocr_box_rows = Mock(side_effect=[
            [{"raw_text": "刻律德菈1", "box": [370, 290, 500, 320]}],
            [], [], [], [],
        ])
        c._sell_one_non_target = Mock(return_value=False)
        self.assertEqual(c.s2_manage_shop(), 1)
        self.assertEqual(c._shop_done_stage, "2-4")

        manager.reset_mock()
        c.state = "currency_main"
        c.s2_handle_pause_modal()
        manager.press.assert_called_once_with("esc")

    @patch("core.currency.s2_actions.key_mouse_manager")
    def test_handle_encounter_dynamic_two_cards_and_lit_select(self, manager):
        c = self.make_s2_currency()
        c.screen[870:925, 920:1000] = 225
        c._ocr_box_rows = Mock(side_effect=[
            [
                {"raw_text": "遭遇其-", "box": [608, 371, 704, 405]},
                {"raw_text": "遭遇其二", "box": [1192, 370, 1309, 406]},
            ],
            [
                {"raw_text": "选择", "box": [931, 882, 987, 913]},
            ],
        ])
        c._ocr_box_text = Mock(side_effect=["遭遇其-金币", "遭遇其二经验"])

        self.assertEqual(c.s2_handle_encounter(), 1)
        self.assertEqual(manager.click_point.call_count, 2)
        self.assertEqual(manager.click_point.call_args_list[0].args, (656, 500))
        self.assertEqual(manager.click_point.call_args_list[1].args, (959, 897))

    @patch("core.currency.s2_actions.key_mouse_manager")
    def test_halo_replace_rotates_front_slots_then_confirms(self, manager):
        c = self.make_s2_currency()
        c._halo_drag_idx = 1
        c._ocr_box_rows = Mock(return_value=[])
        c.click_text = Mock(return_value=False)

        for step, expected_sx in enumerate((905, 1010, 1115), start=1):
            c.s2_handle_halo_replace()
            self.assertTrue(c._halo_force_next_drag)
            self.assertEqual(c._halo_replace_tries, step)
            c._ocr_box_text = Mock(return_value="已有主角角色")
            c.s2_handle_halo()
            self.assertEqual(manager.drag_point.call_args.args[:2], (expected_sx, 600))

        manager.reset_mock()
        c.s2_handle_halo_replace()
        self.assertEqual(c._halo_replace_tries, 0)
        self.assertFalse(c._halo_force_next_drag)
        manager.click_point.assert_called_with(1187, 671)

    @patch("core.currency.s2_actions.key_mouse_manager")
    def test_split_level_ocr_and_truncated_core_name_protection(self, manager):
        c = self.make_s2_currency()
        c._ocr_box_text = Mock(return_value="V7")
        self.assertEqual(c._read_board_level(), 7)

        c.preparation.scan_bench.return_value = [
            BenchSlot(3, "character_4", "character", ("back",)),
        ]
        c._ocr_box_text = Mock(return_value="刻律德")
        c._click_text_fast = Mock(return_value=True)
        self.assertFalse(c._sell_one_non_target(("刻律德菈", "白露")))
        c._click_text_fast.assert_not_called()

    @patch("core.currency.s2_actions.key_mouse_manager")
    def test_bench_overflow_sells_lowest_priority_core_when_all_core(self, manager):
        c = self.make_s2_currency()
        c._current_plan = Mock(return_value=("s2_anaxa", ("那刻夏", "符玄", "白露")))
        c.preparation.scan_bench.return_value = [
            BenchSlot(0, "character_1", "character", ("front",)),
            BenchSlot(7, "character_8", "character", ("back",)),
            BenchSlot(8, "character_9", "character", ("back",)),
        ]
        # 首次检测到右侧已有详情页“出售”先 ESC，再逆序检查格 9/格 8/格 1（白露在目标末位）
        c._ocr_box_text = Mock(side_effect=["出售", "白露", "符玄", "那刻夏"])
        c._click_text_fast = Mock(return_value=True)

        self.assertEqual(c.s2_handle_bench_overflow(), 1)
        c._click_text_fast.assert_called_once_with("出售", [1680, 875, 1915, 965])
        # 最终再次点击了格 9 白露 (x=1438, y=910) 并出售
        self.assertEqual(manager.click_point.call_args.args, (1438, 910))

    @patch("core.currency.s2_actions.key_mouse_manager")
    def test_s2_home_guide_advantage_and_left_panel_rank_stepdown(self, manager):
        c = self.make_s2_currency()
        c._ocr_box_text = Mock(return_value="优势布局也更新了，记得激活。")
        c._click_text_fast = Mock(return_value=True)
        self.assertEqual(c.s2_handle_home(), 1)
        c._click_text_fast.assert_called_once_with("优势布局", [50, 320, 260, 410], fallback=(154, 364))

        c._ocr_box_text = Mock(return_value="可布局")
        c._click_lit_confirm = Mock(return_value=True)
        self.assertEqual(c.s2_handle_advantage(), 1)

        manager.reset_mock()
        c.s2_cfg["rank"] = "lowest"
        c._ocr_box_text = Mock(side_effect=["A5紫金", "A5紫金"])
        c._read_int = Mock(return_value=32)
        self.assertEqual(c.s2_handle_rank(), 1)
        self.assertEqual(manager.click_point.call_count, 3)
        self.assertEqual(manager.click_point.call_args.args, (270, 780))

        manager.reset_mock()
        c._ocr_box_text = Mock(side_effect=["AO黑铁", "AO黑铁"])
        c._read_int = Mock(return_value=None)
        c._click_text_fast = Mock(return_value=True)
        self.assertEqual(c.s2_handle_rank(), 1)
        c._click_text_fast.assert_called_once_with("开始对局", [1550, 910, 1850, 1020], fallback=(1692, 965))

    @patch("core.currency.automation.key_mouse_manager")
    @patch("core.currency.automation.time")
    def test_startbattle_silent_hang_breaks_out_after_grace_rounds(self, mock_time, manager):
        from core.currency.automation import SimulatedCurrency

        c = self.make_s2_currency()
        c.state = "startbattle"
        c.run_static = Mock(return_value=(None, 0))
        c._save_loop_screen = Mock(return_value="unknown.png")
        mock_time.time.side_effect = [1000.0 + 15.0 * i for i in range(40)]
        mock_time.monotonic.return_value = 0.0
        mock_time.strftime.return_value = "20261116_190000"

        def stop_on_call():
            c._stop = True

        c.stop.side_effect = stop_on_call
        SimulatedCurrency.loop(c)

        c.stop.assert_called_once()
        manager.press.assert_any_call("esc")
        self.assertEqual(c.state, "currency_main")

    @patch("core.currency.s2_actions.key_mouse_manager")
    def test_equipment_panel_closes_without_spurious_esc_and_zero_gold_o(self, manager):
        c = self.make_s2_currency()
        c._read_stage = Mock(return_value="1-2")
        c._ocr_box_text = Mock(return_value="")
        c._ocr_box_rows = Mock(return_value=[])
        self.assertEqual(c.s2_handle_equipment_panel(), 1)
        manager.click_point.assert_called_once_with(500, 150)
        manager.press.assert_not_called()

        c._read_int = Mock(return_value=None)
        c._ocr_box_text = Mock(return_value="O")
        self.assertEqual(c._read_board_gold(), 0)

    @patch("core.currency.s2_actions.key_mouse_manager")
    def test_tea_prioritizes_confirm_and_clicks_unknown_card_fallback(self, manager):
        c = self.make_s2_currency()
        seen = []
        c._click_text_fast = Mock(side_effect=lambda label, box: seen.append(label) or (label == "确认"))
        self.assertEqual(c.s2_handle_tea(), 1)
        self.assertEqual(seen, ["确认"])

        c2 = self.make_s2_currency()
        c2._click_text_fast = Mock(return_value=False)
        c2._ocr_box_rows = Mock(return_value=[
            {"raw_text": "未知特饮", "box": [1145, 390, 1280, 425]},
            {"raw_text": "返回", "box": [1144, 587, 1200, 618]},
        ])
        self.assertEqual(c2.s2_handle_tea(), 1)
        manager.click_point.assert_called_once_with(1212, 407, box=[1145, 390, 1280, 425])

    def test_empty_patch_guards_prevent_resize_crash(self):
        from core.common.ocr import My_TS
        from core.currency.automation import read_count_number, split_count_line

        # 斜杠在最左边缘 x=0 时，不应返回宽为 0 的左切片
        edge_patch = np.zeros((20, 40, 3), dtype=np.uint8)
        for y in range(20):
            edge_patch[19 - y, min(6, y // 3)] = (230, 220, 40)
        self.assertIsNone(split_count_line(edge_patch))

        fake_ts = Mock()
        self.assertIsNone(read_count_number(fake_ts, np.zeros((20, 0, 3), dtype=np.uint8)))
        fake_ts.ocr_one_row.assert_not_called()

        ts_inst = object.__new__(My_TS)
        ts_inst._ts = Mock()
        self.assertEqual(ts_inst.ocr_one_row(np.zeros((20, 0, 3), dtype=np.uint8)), "")
        ts_inst._ts.text_recognizer.assert_not_called()

    @patch("core.currency.s2_actions.key_mouse_manager")
    def test_master_picks_plan_target_and_generic_choice_bounds_retries(self, manager):
        c = self.make_s2_currency()
        c._current_plan = Mock(return_value=("s2_dot", ("海瑟音", "黑天鹅", "桂乃芬")))
        c._ocr_box_rows = Mock(return_value=[
            {"raw_text": "波提欧", "box": [510, 342, 578, 370]},
            {"raw_text": "黑天鹅", "box": [1352, 342, 1420, 370]},
            {"raw_text": "海瑟音", "box": [1628, 342, 1696, 370]},
        ])
        c._click_lit_confirm = Mock(side_effect=[False, True])
        self.assertEqual(c.s2_handle_master(), 1)
        manager.click_point.assert_called_once_with(1640, 270)
        self.assertEqual(c._master_tries, 0)

        manager.reset_mock()
        c._click_lit_confirm = Mock(return_value=False)
        c._ocr_box_text = Mock(return_value="")
        for _ in range(10):
            self.assertEqual(c.s2_handle_generic_choice(), 1)
        self.assertEqual(c.s2_handle_generic_choice(), 0)
        c.stop.assert_called_once()

    @patch("core.currency.automation.key_mouse_manager")
    @patch("core.currency.automation.time")
    def test_loop_stagnation_stops_even_on_noop_action(self, mock_time, manager):
        from core.currency.automation import SimulatedCurrency

        c = self.make_s2_currency()
        c.run_static = Mock(return_value=("出战前准备", 0))
        c._read_stage = Mock(return_value="3-2")
        c._save_loop_screen = Mock(return_value="stuck.png")
        c._self_diagnose_and_repair = Mock()
        mock_time.time.return_value = 1000.0
        mock_time.monotonic.side_effect = [0.0 + 3.5 * i for i in range(60)]

        def stop_on_call():
            c._stop = True

        c.stop.side_effect = stop_on_call
        SimulatedCurrency.loop(c)

        tiers = [call.args[0] for call in c._self_diagnose_and_repair.call_args_list]
        self.assertEqual(tiers, [1, 2, 3])
        c.stop.assert_called_once()

    @patch("core.currency.s2_actions.key_mouse_manager")
    def test_prepare_stall_recovery_and_bound(self, manager):
        c = self.make_s2_currency()
        c._read_stage = Mock(return_value="2-3")
        c._item_done_stage = "2-3"
        c._shop_done_stage = "2-3"
        c._equip_done_stage = "2-3"
        c.prepare_battle = Mock(return_value=0)
        c.s2_handle_empty_front = Mock(return_value=0)
        c._read_board_gold = Mock(return_value=0)
        c.read_team_count = Mock(return_value=(3, 4))

        results = [c.s2_prepare_and_advance() for _ in range(6)]
        self.assertEqual(results, [0, 1, 0, 0, 0, 0])
        manager.click_point.assert_any_call(500, 150)
        c.stop.assert_called_once()

    @patch("core.currency.s2_actions.key_mouse_manager")
    def test_collect_orbs_held_bench_open_item_drawer_and_loose_equip_drag(self, manager):
        import cv2

        from core.currency.preparation import BenchSlot

        c = self.make_s2_currency()
        c._read_stage = Mock(return_value="2-4")
        c._ocr_box_text = Mock(return_value="")
        cv2.circle(c.screen, (1410, 230), 26, (240, 180, 40), -1)
        c.preparation.scan_bench.return_value = [
            BenchSlot(0, "", "empty", ()),
        ]
        self.assertTrue(c._collect_orbs("2-4"))
        self.assertTrue(any(
            abs(call.args[0] - 1410) <= 5 and abs(call.args[1] - 230) <= 5
            for call in manager.click_point.call_args_list
        ))

        manager.reset_mock()
        c._ocr_box_rows = Mock(return_value=[{"raw_text": "开启", "box": [535, 926, 591, 956]}])
        c._click_open_bench_item(563, 910)
        manager.drag_point.assert_called_once_with(563, 941, 563, 943, duration=0.15)

        manager.reset_mock()
        c._current_plan = Mock(return_value=("s2_followup", ("飞霄",)))
        c._ocr_box_rows = Mock(return_value=[
            {"raw_text": "治疗星徽", "box": [1120, 322, 1230, 350]},
            {"raw_text": "仙舟舰队星徽", "box": [830, 322, 940, 350]},
        ])
        c._click_lit_confirm = Mock(return_value=False)
        c._equip_done_stage = "2-4"
        self.assertEqual(c.s2_handle_item_drawer(), 1)
        manager.click_point.assert_called_once_with(885, 265)
        self.assertIsNone(c._equip_done_stage)

        manager.reset_mock()
        c2 = self.make_s2_currency()
        c2._read_stage = Mock(return_value="2-4")
        c2._orbs_done_stage = "2-4"
        c2._item_done_stage = "2-4"
        c2._shop_done_stage = "2-4"
        c2.prepare_battle = Mock(return_value=1)
        c2.preparation.read_areas.return_value = {"front": [1, 2, 3], "back": [0, 1, 2, 3, 4, 5]}
        c2.preparation.members = {"front": ["character_飞霄", None, None, None], "back": [None] * 6}
        # 在右栏放置闲置装备且前排第 0 格未满 3 槽（已查过推荐）时触发右栏精确中心直拖
        c2.screen[223:267, 1822:1866] = (230, 200, 60)
        c2.screen[233:257, 1832:1856] = (40, 40, 40)
        c2._equip_checked_slots["2-4"] = {("front", 0)}
        self.assertEqual(c2.s2_prepare_and_advance(), 1)
        self.assertEqual(manager.drag_point.call_args.args[:2], (1844, 245))

    @patch("core.currency.s2_actions.key_mouse_manager")
    def test_strategy_guide_tracking_and_proactive_bench_cleanup(self, manager):
        from core.currency.preparation import BenchSlot

        c = self.make_s2_currency()
        c._current_plan = Mock(return_value=("s2_followup", ("飞霄",)))
        c._ocr_box_text = Mock(side_effect=["【6仙舟舰队3追击】", "【6仙舟舰队3追击】"])
        c._click_text_fast = Mock(return_value=True)
        self.assertEqual(c.s2_handle_strategy_guide(), 1)
        self.assertTrue(c._guide_applied_run)
        manager.press.assert_called_once_with("esc")

        manager.reset_mock()
        # 备战格 1 与 3 未追踪、格 7 有金标保留，主动清理只卖出最右侧未追踪格 3
        c.screen[848:870, 383 + 125 * 6 + 4:383 + 125 * 6 + 28] = (20, 160, 240)
        c.preparation.scan_bench.return_value = [
            BenchSlot(0, "char_a", "character", ("front",)),
            BenchSlot(2, "char_b", "character", ("back",)),
            BenchSlot(6, "char_c", "character", ("front",)),
        ]
        c._ocr_box_text = Mock(return_value="")
        c._click_text_fast = Mock(return_value=True)
        self.assertTrue(c._cleanup_untracked_bench("2-4"))
        self.assertEqual(manager.click_point.call_args_list[0].args, (383 + 125 * 2 + 55, 910))
        c._click_text_fast.assert_called_once_with("出售", [1680, 875, 1915, 965])

    @patch("core.currency.s2_actions.key_mouse_manager")
    def test_halo_tutorial_detail_and_self_repair_overlay_escape(self, manager):
        c = self.make_s2_currency()
        self.assertEqual(c.s2_handle_halo_tutorial(), 1)
        manager.click_point.assert_called_once_with(142, 174)

        manager.reset_mock()
        c._ocr_box_text = Mock(return_value="前往重选会同时重选主角角色和光环")
        self.assertEqual(c.s2_handle_halo_detail(), 1)
        self.assertEqual(manager.click_point.call_args_list, [((1863, 65),), ((1863, 65),)])

        # 通用「提示：」青框高亮目标可由自检准确识别并点击，且「收起」未消除时不阻断 Tier 2 推进
        manager.reset_mock()
        c2 = self.make_s2_currency()
        c2.screen[120:220, 60:180] = (230, 230, 40)
        c2._ocr_box_rows = Mock(return_value=[{"raw_text": "提示：测试教学", "box": [320, 200, 680, 250]}])
        self.assertTrue(c2._self_diagnose_and_repair(1, stage="2-1", reason="冻结测试"))
        cx, cy = manager.click_point.call_args.args
        self.assertTrue(50 <= cx <= 190 and 110 <= cy <= 230)

        manager.reset_mock()
        c2._ocr_box_rows = Mock(return_value=[{"raw_text": "可合成列表", "box": [860, 300, 1020, 340]}])
        self.assertTrue(c2._self_diagnose_and_repair(1, stage="1-5", reason="补给装备预览"))
        manager.click_point.assert_called_with(1863, 65)

        manager.reset_mock()
        c2._ocr_box_rows = Mock(return_value=[{"raw_text": "数据统计", "box": [380, 230, 520, 280]}])
        self.assertTrue(c2._self_diagnose_and_repair(1, stage="2-2", reason="统计弹窗"))
        manager.click_point.assert_called_with(1495, 265)

        manager.reset_mock()
        c3 = self.make_s2_currency()
        c3._ocr_box_rows = Mock(return_value=[{"raw_text": "备战阶段", "box": [240, 50, 320, 85]}])
        c3._ocr_box_text = Mock(return_value="收起")
        self.assertTrue(c3._self_diagnose_and_repair(2, stage="2-1", reason="商店遮挡"))
        clicked = [call.args[:2] for call in manager.click_point.call_args_list]
        self.assertIn((1620, 980), clicked)
        self.assertIn((1816, 749), clicked)

    @patch("core.currency.automation.get_hwnd_and_text", return_value=(12345, "绝区零"))
    @patch("core.currency.automation.key_mouse_manager")
    @patch("core.currency.automation.ZZZ_LOCK_PATH")
    def test_loop_pauses_immediately_when_foreground_is_not_starrail(self, lock_path, manager, _hwnd):
        from core.currency.automation import SimulatedCurrency

        lock_path.exists.return_value = False
        c = self.make_s2_currency()
        c._guard_foreground = True
        c.run_static = Mock()
        SimulatedCurrency.loop(c)
        c.get_screen.assert_not_called()
        c.run_static.assert_not_called()
        manager.click_point.assert_not_called()
        c.stop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
