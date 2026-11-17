"""S2「连击阿格莱雅/那刻夏」装备库、动态记忆数据库、经济策略与视觉校验单元测试。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from core.currency.combo_strategy import (
    decide_combo_shop,
    plan_combo_deployment,
    plan_combo_sell,
    streak_regime,
)
from core.currency.equipment_db import (
    SYNTHESIS_TABLE,
    UnitEquipSnapshot,
    lookup_recipe,
    pairable_recipes,
    plan_best_equip_action,
    plan_wrench_transfer,
    score_item_for_unit,
)
from core.currency.match_db import MatchDatabase
from core.currency.preparation import (
    AREA_BOXES,
    BenchSlot,
    ItemIconCatalog,
    detect_card_equip_count,
    detect_inventory_rail_positions,
)
from core.currency.s2_strategy import ShopOffer


class CurrencyS2ComboDatabaseTests(unittest.TestCase):
    def test_synthesis_table_covers_all_36_s2_pairs_and_role_guards(self):
        self.assertEqual(len(SYNTHESIS_TABLE), 36)
        self.assertEqual(lookup_recipe("轮滑鞋", "轮滑鞋").name, "反重力皮靴")
        self.assertEqual(lookup_recipe("生命之花", "轮滑鞋").name, "反螺旋舞步")
        self.assertEqual(lookup_recipe("光能电池", "轮滑鞋").name, "星航履")
        self.assertEqual(lookup_recipe("光能电池", "生命之花").name, "绝对热量")
        self.assertEqual(lookup_recipe("光能电池", "光能电池").name, "永动机")

        pairs = pairable_recipes(["轮滑鞋", "光能电池", "以太钻头"])
        self.assertEqual(pairs[0][2].name, "光速螺旋桨")

        # 已满 3 槽的角色绝不能再分任何装备
        self.assertEqual(
            score_item_for_unit("阿格莱雅", "front", 2, ("a", "b", "c"), 3, "反重力皮靴"),
            -999,
        )
        # 主 C 绝不能吃纯防御/肉装
        self.assertEqual(
            score_item_for_unit("阿格莱雅", "front", 2, (), 0, "很硬的甲", hp=100, is_final=False),
            0,
        )
        self.assertEqual(
            score_item_for_unit("那刻夏", "front", 2, (), 0, "量产型装甲", hp=100, is_final=False),
            0,
        )
        # 后台辅助与前排治疗正常拿对应回能/全队回血装备
        support_score = score_item_for_unit("刻律德菈", "back", 2, (), 0, "永动机")
        sustain_score = score_item_for_unit("白露", "back", 2, (), 0, "绝对热量", hp=50)
        self.assertGreater(support_score, 110)
        self.assertGreater(sustain_score, 120)

    def test_equip_planner_avoids_full_unit_holds_single_shoe_and_uses_wrench(self):
        units = [
            UnitEquipSnapshot("front", 0, "阿格莱雅", 2, (), 0),
            UnitEquipSnapshot("front", 1, "那刻夏", 2, ("反重力皮靴", "火力风暴潮", "星航履"), 3),
            UnitEquipSnapshot("back", 0, "刻律德菈", 2, (), 0),
            UnitEquipSnapshot("back", 1, "白露", 2, (), 0),
        ]
        # 1. 有拆装扳手且那刻夏穿满核心输出装、阿格莱雅刚上场空装时应触发拆装转交
        wrench = plan_wrench_transfer(units, has_wrench=True)
        self.assertIsNotNone(wrench)
        self.assertEqual(wrench.source_name, "那刻夏")
        self.assertEqual(wrench.target_name, "阿格莱雅")

        # 危险星徽环境下且只穿星徽时不乱拆
        emblem_only = [
            UnitEquipSnapshot("front", 0, "阿格莱雅", 2, (), 0),
            UnitEquipSnapshot("front", 1, "雪衣", 2, ("连击星徽",), 1),
        ]
        self.assertIsNone(plan_wrench_transfer(emblem_only, has_wrench=True, has_risky_emblem_reset=True))

        # 2. 非终局仅有一只独立散件「轮滑鞋」时保留等双鞋拼反重力皮靴，不乱拖
        held = plan_best_equip_action(units, [((1844, 268), "轮滑鞋")], hp=90, is_final=False)
        self.assertIsNone(held)

        # 3. 两只轮滑鞋在手直接指派给未满 3 格的阿格莱雅合「反重力皮靴」，绝不选已满 3 格的那刻夏
        crafted = plan_best_equip_action(
            units,
            [((1844, 268), "轮滑鞋"), ((1844, 345), "轮滑鞋")],
            hp=90,
            is_final=False,
        )
        self.assertIsNotNone(crafted)
        self.assertEqual(crafted.item_name, "反重力皮靴")
        self.assertEqual(crafted.unit_name, "阿格莱雅")

        # 4. 现成防装「很硬的甲」不会给主 C 阿格莱雅，而是精准拖给后台生存位白露
        tank_act = plan_best_equip_action(
            units,
            [((1844, 268), "很硬的甲")],
            hp=75,
            is_final=False,
        )
        self.assertIsNotNone(tank_act)
        self.assertEqual(tank_act.unit_name, "白露")

    def test_match_db_tracks_purchases_moves_combat_streaks_and_roundtrips(self):
        with tempfile.TemporaryDirectory() as d:
            db = MatchDatabase(Path(d) / "match_db.json")
            db.reset_run("A5紫金1")
            db.sync_economy(stage="1-3", gold=18, level=4, exp=(2, 8), counts=(2, 4), hp=100)
            db.record_purchase("那刻夏", 1, is_tracked=True)
            db.sync_bench([BenchSlot(0, "character_hash1", "character", ("front",))])
            self.assertEqual(db.slot("bench", 0).name, "那刻夏")
            self.assertEqual(db.template_names["character_hash1"], "那刻夏")

            db.record_detail("bench", 0, "那刻夏", ["反重力皮靴"], stars=2)
            db.record_move("bench", 0, "front", 0)
            self.assertFalse(db.slot("bench", 0).occupied)
            self.assertEqual(db.slot("front", 0).name, "那刻夏")
            self.assertEqual(db.slot("front", 0).worn_items, ["反重力皮靴"])
            self.assertEqual(db.slot("front", 0).stars, 2)

            db.record_combat_outcome(True, 100)
            db.record_combat_outcome(True, 100)
            self.assertEqual(db.win_streak, 2)
            self.assertEqual(streak_regime(db), "win_streak")

            db.record_combat_outcome(False, 58)
            self.assertEqual(db.win_streak, 0)
            self.assertEqual(db.loss_streak, 1)
            self.assertEqual(streak_regime(db), "recovery")

            db.save()
            db2 = MatchDatabase(Path(d) / "match_db.json")
            self.assertTrue(db2.load())
            self.assertEqual(db2.rank_text, "A5紫金1")
            self.assertEqual(db2.loss_streak, 1)
            self.assertEqual(db2.slot("front", 0).worn_items, ["反重力皮靴"])

    def test_combo_strategy_switches_win_vs_loss_streak_and_saves_reward_interest(self):
        with tempfile.TemporaryDirectory() as d:
            db = MatchDatabase(Path(d) / "db.json")
            db.reset_run("A5紫金1")
            db.slot("front", 0).occupied = True
            db.slot("front", 0).name = "那刻夏"
            db.slot("front", 0).stars = 2
            db.slot("back", 0).occupied = True
            db.slot("back", 0).name = "白露"
            db.slot("back", 0).stars = 1

            # 1. 奖励前 1-3 连胜高血 34 金（刚跨 30 息线），应稳住 30 利息并用溢出 4 金升经验冲高费
            db.sync_economy(stage="1-3", gold=34, level=5, exp=(4, 12), counts=(2, 2), hp=96)
            db.win_streak = 3
            db.loss_streak = 0
            action, offer = decide_combo_shop(
                db,
                offers=[ShopOffer(0, "波提欧", 1), ShopOffer(1, "其他", 3)],
                tracked_cols=(),
                refresh_cost=2,
                xp_cost=4,
                refreshes=0,
                ops=0,
            )
            self.assertEqual(action, "exp")

            # 手里正好 31 金（不可破 30 利息线，且升级差 8 金），不乱 D 牌卡利息直接收手
            db.sync_economy(stage="1-3", gold=31, level=5, exp=(0, 8), counts=(2, 2), hp=96)
            action, _ = decide_combo_shop(
                db,
                offers=[ShopOffer(0, "波提欧", 2)],
                tracked_cols=(),
                refresh_cost=2,
                xp_cost=4,
                refreshes=0,
                ops=0,
            )
            self.assertEqual(action, "done")

            # 2. 奖励前金币 29 + 备战有 1 费垃圾未追踪过渡卡，触发卖出凑满 30 整息
            db.sync_economy(stage="1-3", gold=29, level=5, counts=(2, 2), hp=92)
            db.slot("bench", 4).occupied = True
            db.slot("bench", 4).name = "过渡甲"
            db.slot("bench", 4).cost = 1
            sell_pick = plan_combo_sell(db, for_interest=True)
            self.assertIsNotNone(sell_pick)
            self.assertEqual(sell_pick.index, 4)

            # 3. 连败/低血止损期（那刻夏未 2 星且缺治疗），优先级切到 D 牌补质量而非裸升人口
            db.slot("front", 0).stars = 1
            db.slot("back", 0).clear_unit()
            db.win_streak = 0
            db.loss_streak = 2
            db.sync_economy(stage="2-3", gold=24, level=5, exp=(4, 12), counts=(1, 5), hp=46)
            action, _ = decide_combo_shop(
                db,
                offers=[ShopOffer(0, "佩佩", 1)],
                tracked_cols=(),
                refresh_cost=2,
                xp_cost=4,
                refreshes=0,
                ops=0,
            )
            self.assertEqual(action, "refresh")

            # 4. 商店刷出 5 费阿格莱雅，直接果断买入
            action, offer = decide_combo_shop(
                db,
                offers=[ShopOffer(2, "阿格莱雅", 5)],
                tracked_cols=(2,),
                refresh_cost=2,
                xp_cost=4,
                refreshes=1,
                ops=1,
            )
            self.assertEqual(action, "buy")
            self.assertEqual(offer.name, "阿格莱雅")

            # 5. 前后台精准站位：刻律德菈优先后台、阿格莱雅优先前台
            db.sync_economy(stage="3-2", gold=12, level=8, counts=(2, 4), hp=80)
            db.slot("bench", 1).occupied = True
            db.slot("bench", 1).name = "阿格莱雅"
            db.slot("bench", 1).can_front = True
            db.slot("bench", 1).can_back = False
            mv = plan_combo_deployment(db)
            self.assertIsNotNone(mv)
            self.assertEqual(mv[0].name, "阿格莱雅")
            self.assertEqual(mv[1], "front")

    def test_vision_equip_count_rail_offset_and_icon_catalog(self):
        # 1. 探测「装备追踪中」状态下右栏下移 23px 的格子坐标（心区 [260..280] 有高亮图案而未追格 [229..261] 全黑/低方差）
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[262:284, 1832:1856] = 240
        frame[266:276, 1838:1850] = 30
        self.assertEqual(detect_inventory_rail_positions(frame, tracked_bar=True), [(1844, 268)])
        self.assertEqual(detect_inventory_rail_positions(frame, tracked_bar=False), [])

        # 2. 探测场上角色卡片下方 0/2 件已穿戴图标数量，且 2px 高的紫色分隔横条不误判
        box0 = AREA_BOXES["front"][0]
        self.assertEqual(detect_card_equip_count(frame, box0), 0)
        cx0 = (box0[0] + box0[2]) // 2
        y2 = box0[3]
        frame[y2 + 20:y2 + 22, cx0 - 50:cx0 + 50] = (230, 190, 210)
        self.assertEqual(detect_card_equip_count(frame, box0), 0)
        frame[y2 + 20:y2 + 22, cx0 - 50:cx0 + 50] = 0
        # 右栏暗紫低对比墙缝（max<=153, pct165=0）不误识别为未知散件
        wall_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        wall_frame[246:290, 1822:1866] = (70, 37, 71)
        wall_frame[246:290, 1822:1844] = (150, 90, 110)
        self.assertEqual(detect_inventory_rail_positions(wall_frame, tracked_bar=True), [])
        # 绘制两枚宽 26px 亮金高方差图标块（模拟已穿 2 件）
        frame[y2 + 4:y2 + 30, cx0 - 32:cx0 - 6] = (40, 210, 240)
        frame[y2 + 8:y2 + 24, cx0 - 26:cx0 - 12] = (180, 160, 220)
        frame[y2 + 4:y2 + 30, cx0 + 4:cx0 + 30] = (40, 210, 240)
        frame[y2 + 8:y2 + 24, cx0 + 10:cx0 + 24] = (180, 160, 220)
        self.assertEqual(detect_card_equip_count(frame, box0), 2)

        # 3. ItemIconCatalog 保存与精准匹配
        with tempfile.TemporaryDirectory() as d:
            cat = ItemIconCatalog(d)
            patch = np.zeros((44, 44, 3), dtype=np.uint8)
            cv2.circle(patch, (22, 22), 14, (40, 200, 245), -1)
            cv2.rectangle(patch, (14, 14), (30, 30), (220, 80, 60), 2)
            cat.save_item("反重力皮靴", patch)
            self.assertEqual(cat.identify(patch), "反重力皮靴")

    @patch("core.currency.s2_actions.key_mouse_manager")
    def test_highest_rank_climb_and_equip_tracking_modal(self, manager):
        from test.test_currency_s2_automation import CurrencyS2AutomationTests

        # 0. 上次保留了低职级但界面出现「返回最高职级」时，先点击升回 A5 再开局
        c_low = CurrencyS2AutomationTests.make_s2_currency()
        c_low.s2_cfg["rank"] = "highest"
        c_low._ocr_box_text = Mock(side_effect=["A0黑铁", "返回最高职级"])
        c_low._click_text_fast = Mock(return_value=True)
        self.assertEqual(c_low.s2_handle_rank(), 1)
        c_low._click_text_fast.assert_called_once_with(
            "返回最高职级", [1240, 910, 1545, 1020], fallback=(1392, 965),
        )
        self.assertEqual(c_low._rank_steps, 1)

        c = CurrencyS2AutomationTests.make_s2_currency()
        c.s2_cfg["rank"] = "highest"
        c._ocr_box_text = Mock(return_value="A5紫金1")
        c._click_text_fast = Mock(return_value=True)
        self.assertEqual(c.s2_handle_rank(), 1)
        # 不再点左侧降级 (270, 780)，直接保留最高已解锁职级开局
        for call in manager.click_point.call_args_list:
            self.assertNotEqual(call.args, (270, 780))
        self.assertEqual(c.match_db.rank_text, "A5紫金1")

        # 点选场上未知角色卡：应过滤「试用」和费用后缀、跳过暗色空装备格，并用空白处点击关闭而非按 Esc
        manager.reset_mock()
        c.match_db.slot("front", 0).occupied = True
        c.match_db.slot("front", 0).name = ""
        c._ocr_box_text = Mock(return_value="试用 那刻夏 1")
        self.assertTrue(c._learn_field_identity_if_unknown("1-2"))
        self.assertEqual(c.match_db.slot("front", 0).name, "那刻夏")
        manager.press.assert_not_called()
        self.assertEqual(manager.click_point.call_args_list[-1].args, (500, 150))

        manager.reset_mock()
        c._ocr_box_rows = Mock(return_value=[
            {"raw_text": "追踪", "box": [1229, 413, 1281, 443]},
            {"raw_text": "取消追踪", "box": [1227, 531, 1283, 565]},
            {"raw_text": "追踪", "box": [1231, 656, 1281, 682]},
        ])
        c._ocr_box_text = Mock(return_value="")
        self.assertEqual(c.s2_handle_equip_tracking(), 1)
        clicked = [call.args for call in manager.click_point.call_args_list]
        self.assertEqual(clicked, [(1255, 428), (1256, 669), (1343, 269)])
        self.assertTrue(c.match_db.guide_applied)

        # OCR 别名、商店读金失败时不盲用旧金币按 F、以及两件直拖先下后上合成顺位
        from core.currency.match_db import _infer_template_name
        from core.currency.s2_strategy import clean_unit_name

        self.assertEqual(clean_unit_name("刻律德拉"), "刻律德菈")
        self.assertEqual(_infer_template_name("character_霍霍"), "藿藿")

        manager.reset_mock()
        c.match_db.reset_run(rank_text="A5紫金")
        c.match_db.gold = 8
        c._read_stage = Mock(return_value="1-3")
        c._read_board_gold = Mock(return_value=None)
        c._read_board_level = Mock(return_value=4)
        c._read_board_exp = Mock(return_value=(0, 8))
        c._read_xp_cost = Mock(return_value=4)
        c._read_refresh_cost = Mock(return_value=2)
        c._last_team_counts = (4, 4)
        c.preparation.scan_bench.return_value = []
        c._ocr_box_rows = Mock(return_value=[])
        self.assertEqual(c.s2_manage_shop(), 1)
        manager.press.assert_not_called()

        manager.reset_mock()
        c._sync_match_db = Mock()
        c.match_db.sync_economy(stage="1-3", gold=4, level=4, counts=(1, 4), hp=80)
        c.match_db.slot("front", 0).occupied = True
        c.match_db.slot("front", 0).name = "阿格莱雅"
        c.match_db.slot("front", 0).equip_count = 0
        c.match_db.sync_inventory([((1844, 268), "轮滑鞋"), ((1844, 422), "光能电池")])
        c._equip_checked_slots["1-3"] = {("front", 0)}
        c._item_catalog.identify = Mock(side_effect=["轮滑鞋", "光能电池"])
        self.assertTrue(c._run_smart_equip("1-3"))
        self.assertEqual(manager.drag_point.call_count, 2)
        self.assertEqual(manager.drag_point.call_args_list[0].args[:2], (1844, 422))
        self.assertEqual(manager.drag_point.call_args_list[1].args[:2], (1844, 268))
        self.assertIn("星航履", c.match_db.slot("front", 0).worn_items)

        # 低血濒死（15 HP, 6 金, 4/16 exp）时优先 D 牌搜升星/战力，绝不浪费 4 金换半管经验
        db_desp = MatchDatabase(c.match_db.path)
        db_desp.reset_run("A5紫金")
        db_desp.sync_economy(stage="1-8", gold=6, level=5, exp=(4, 16), counts=(5, 5), hp=15)
        db_desp.slot("bench", 0).clear_unit()
        act_desp, _ = decide_combo_shop(
            db_desp,
            offers=[ShopOffer(0, "三月七", 1)],
            tracked_cols=(),
            refresh_cost=2,
            xp_cost=4,
            refreshes=0,
            ops=0,
        )
        self.assertEqual(act_desp, "refresh")

        # 1★ 刻律德菈不可踢掉已在场的 1★ 藿藿核心奶妈
        db_desp.slot("back", 0).occupied = True
        db_desp.slot("back", 0).name = "藿藿"
        db_desp.slot("bench", 0).occupied = True
        db_desp.slot("bench", 0).name = "刻律德菈"
        db_desp.slot("bench", 0).can_front = False
        db_desp.slot("bench", 0).can_back = True
        self.assertIsNone(plan_combo_deployment(db_desp))

        # 晶球识别支持金色高阶晶球并忽略低亮灰圆钮，且 _collect_orbs 每关至多扫盘 2 次防死锁
        from core.currency.s2_strategy import detect_orb_points

        orb_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.circle(orb_frame, (1420, 265), 24, (52, 167, 213), -1)  # 金球
        cv2.circle(orb_frame, (1651, 247), 18, (148, 126, 156), -1)  # 灰白速合对勾圆钮
        pts = detect_orb_points(orb_frame, limit=12)
        self.assertEqual(pts, [(1420, 265)])

        manager.reset_mock()
        c.screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.circle(c.screen, (1420, 265), 24, (240, 180, 40), -1)
        c._orbs_done_stage = None
        c._orb_overflow_stage = None
        c._orb_tries = {}
        c._ocr_box_text = Mock(return_value="")
        c.preparation.scan_bench.return_value = [BenchSlot(0, "", "empty", ())]
        self.assertTrue(c._collect_orbs("1-3"))
        self.assertTrue(c._collect_orbs("1-3"))
        self.assertFalse(c._collect_orbs("1-3"))

    @patch("core.currency.automation.set_forground")
    @patch("core.currency.s2_actions.key_mouse_manager")
    @patch("core.currency.automation.key_mouse_manager")
    @patch("core.currency.automation.time")
    def test_freeze_detector_self_repair_tiers_and_reset(self, mock_time, auto_km, s2_km, _fg):
        from core.currency.automation import SimulatedCurrency
        from test.test_currency_s2_automation import CurrencyS2AutomationTests

        # 1. 像素长时间静止（哪怕每次都在跑战斗或同一状态）应先逐级触发 1→2→3 级自检修复，145s 仍无效才停机
        c = CurrencyS2AutomationTests.make_s2_currency()
        c.screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        c.run_static = Mock(return_value=("战斗中", 1))
        c._save_loop_screen = Mock(return_value="freeze.png")
        c._self_diagnose_and_repair = Mock()
        mock_time.time.return_value = 2000.0
        mock_time.monotonic.side_effect = [float(i * 10) for i in range(30)]

        def stop_on_call():
            c._stop = True

        c.stop.side_effect = stop_on_call
        SimulatedCurrency.loop(c)
        self.assertEqual([call.args[0] for call in c._self_diagnose_and_repair.call_args_list], [1, 2, 3])
        auto_km.press.assert_any_call("esc")
        c.stop.assert_called_once()

        # 2. 只要画面像素重新发生变化（第 4 帧跳变），冻结计时器立即归零，不触发停机
        auto_km.reset_mock()
        c2 = CurrencyS2AutomationTests.make_s2_currency()
        frames = [
            np.zeros((1080, 1920, 3), dtype=np.uint8),
            np.zeros((1080, 1920, 3), dtype=np.uint8),
            np.zeros((1080, 1920, 3), dtype=np.uint8),
            np.full((1080, 1920, 3), 80, dtype=np.uint8),
        ]
        idx = [0]

        def next_frame():
            f = frames[min(idx[0], len(frames) - 1)]
            c2.screen = f
            idx[0] += 1
            if idx[0] >= 5:
                c2._stop = True
            return f

        c2.get_screen = Mock(side_effect=next_frame)
        c2._save_loop_screen = Mock(return_value="freeze2.png")
        c2.run_static = Mock(return_value=("战斗中", 1))
        c2._self_diagnose_and_repair = Mock()
        mock_time.monotonic.side_effect = [0.0, 0.0, 20.0, 34.0, 40.0, 45.0, 50.0]
        SimulatedCurrency.loop(c2)
        c2._self_diagnose_and_repair.assert_not_called()
        c2.stop.assert_not_called()

        # 3. _self_diagnose_and_repair 1 级点「点击空白处继续」、2 级关「收起」商店与推进「出战」
        c3 = CurrencyS2AutomationTests.make_s2_currency()
        s2_km.reset_mock()
        c3._ocr_box_text = Mock(return_value="")
        c3._ocr_box_rows = Mock(return_value=[{"raw_text": "点击空白处继续", "box": [820, 920, 1100, 970]}])
        self.assertTrue(c3._self_diagnose_and_repair(1, stage="2-3", reason="冻结测试"))
        s2_km.click_point.assert_called_with(960, 945, box=[820, 920, 1100, 970])

        s2_km.reset_mock()
        c3._ocr_box_text = Mock(side_effect=["", "收起", ""])
        c3._ocr_box_rows = Mock(return_value=[])
        self.assertTrue(c3._self_diagnose_and_repair(2, stage="2-3", reason="商店遮挡"))
        s2_km.click_point.assert_called_with(1620, 980)
        self.assertEqual(c3._orbs_done_stage, "2-3")
        self.assertEqual(c3._equip_done_stage, "2-3")
        self.assertEqual(c3._shop_done_stage, "2-3")


if __name__ == "__main__":
    unittest.main()
