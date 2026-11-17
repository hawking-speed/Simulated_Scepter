import unittest

import numpy as np

from core.currency.s2_strategy import (
    ANAXA_BACK,
    ANAXA_FRONT,
    ShopOffer,
    clean_unit_name,
    decide_shop_action,
    detect_bench_guide_badges,
    detect_loose_equipment_slots,
    detect_orb_points,
    detect_shop_guide_badges,
    has_guide_badge,
    is_button_lit,
    parse_shop_offers,
    pick_encounter_index,
    score_halo_text,
    score_investment_text,
    score_item_choice,
)


class S2StrategyTests(unittest.TestCase):
    def test_clean_unit_name_fixes_ocr_typos(self):
        self.assertEqual(clean_unit_name("「干冶·刃」"), "千冶·刃")
        self.assertEqual(clean_unit_name("刻律德"), "刻律德菈")
        self.assertEqual(clean_unit_name("丹恒•饮月 "), "丹恒·饮月")

    def test_parse_shop_offers_handles_inline_and_fallback_costs(self):
        offers = parse_shop_offers([
            (0, "那刻夏1", None),
            (1, "干冶·刃 3", None),
            (2, "刻律德", 1),
            (3, "银狼LV.9994", None),
            (4, "", None),
        ])
        self.assertEqual(
            offers,
            [
                ShopOffer(0, "那刻夏", 1),
                ShopOffer(1, "千冶·刃", 3),
                ShopOffer(2, "刻律德菈", 1),
                ShopOffer(3, "银狼LV.999", 4),
            ],
        )

    def test_decide_shop_prioritizes_target_units_and_space(self):
        targets = ANAXA_FRONT + ANAXA_BACK
        offers = [ShopOffer(0, "瓦尔特", 1), ShopOffer(1, "那刻夏", 1), ShopOffer(2, "白露", 2)]

        action, item = decide_shop_action(
            gold=5, level=4, placed=3, capacity=4, bench_empty=2,
            offers=offers, target_names=targets, refresh_cost=2, xp_cost=4,
            refreshes=0, ops=0,
        )
        self.assertEqual(action, "buy")
        self.assertEqual(item, ShopOffer(1, "那刻夏", 1))

        # 备战栏满且有目标卡时要求出售腾位
        action, _ = decide_shop_action(
            gold=5, level=4, placed=4, capacity=4, bench_empty=0,
            offers=offers, target_names=targets, refresh_cost=2, xp_cost=4,
            refreshes=0, ops=0,
        )
        self.assertEqual(action, "sell")

    def test_decide_shop_buys_filler_when_under_capacity_and_no_target(self):
        offers = [ShopOffer(0, "虎克", 1), ShopOffer(1, "灵砂", 1)]
        action, item = decide_shop_action(
            gold=4, level=4, placed=2, capacity=4, bench_empty=3,
            offers=offers, target_names=("那刻夏",), refresh_cost=2, xp_cost=4,
            refreshes=0, ops=0,
        )
        self.assertEqual(action, "buy")
        self.assertEqual(item, ShopOffer(1, "灵砂", 1))

    def test_decide_shop_buys_exp_then_refreshes_and_respects_limits(self):
        action, _ = decide_shop_action(
            gold=8, level=5, placed=5, capacity=5, bench_empty=3,
            offers=[], target_names=("那刻夏",), refresh_cost=2, xp_cost=4,
            refreshes=0, ops=0,
        )
        self.assertEqual(action, "exp")

        action, _ = decide_shop_action(
            gold=6, level=8, placed=8, capacity=8, bench_empty=3,
            offers=[], target_names=("那刻夏",), refresh_cost=2, xp_cost=4,
            refreshes=0, ops=1,
        )
        self.assertEqual(action, "refresh")

        action, _ = decide_shop_action(
            gold=20, level=5, placed=5, capacity=5, bench_empty=3,
            offers=[ShopOffer(0, "那刻夏", 1)], target_names=("那刻夏",),
            refresh_cost=2, xp_cost=4, refreshes=0, ops=8, max_ops=8,
        )
        self.assertEqual(action, "done")

    def test_investment_and_halo_scoring_avoids_fatal_risks(self):
        s_catalog = score_investment_text("失去1点生命，获得随机装备", True, ("连击概念股",))
        s_fatal = score_investment_text("小队生命值变为1并随机移除角色", True, ("连击概念股",))
        self.assertGreater(s_catalog, 0)
        self.assertLess(s_fatal, -500)

        self.assertLess(score_halo_text("角色开局献祭并离场"), -500)
        self.assertGreater(score_halo_text("前台强度提高且施放战技恢复能量"), 0)

    def test_pick_encounter_chooses_lowest_tier(self):
        self.assertEqual(pick_encounter_index(["遭遇其三", "遭遇其一", "遭遇其二"]), 1)
        self.assertEqual(pick_encounter_index(["遭遇其二", "遭遇其-"]), 1)
        self.assertEqual(pick_encounter_index(["困难挑战", "普通遭遇"]), 1)

    def test_is_button_lit_distinguishes_bright_vs_dim(self):
        dim = np.full((24, 80, 3), 50, dtype=np.uint8)
        lit = np.full((24, 80, 3), 220, dtype=np.uint8)
        self.assertFalse(is_button_lit(dim))
        self.assertTrue(is_button_lit(lit))

    def test_detect_orb_points_equipment_rail_and_item_scoring(self):
        import cv2

        blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.assertEqual(detect_orb_points(blank), [])
        self.assertEqual(detect_loose_equipment_slots(blank), [])

        # 在托盘放两颗亮蓝/银晶球并验证质心坐标
        cv2.circle(blank, (1400, 220), 28, (240, 180, 40), -1)
        cv2.circle(blank, (1520, 310), 18, (220, 220, 220), -1)
        pts = detect_orb_points(blank)
        self.assertEqual(len(pts), 2)

        # 在右栏第 1、2 装备格放置道具图案，工具行 y=188 不应混入
        blank[225:265, 1830:1875] = (230, 200, 60)
        blank[235:255, 1840:1865] = (40, 40, 40)
        blank[305:345, 1830:1875] = (190, 80, 220)
        blank[315:335, 1840:1865] = (30, 30, 30)
        self.assertEqual(detect_loose_equipment_slots(blank), [(1852, 245), (1852, 325)])

        self.assertGreater(
            score_item_choice("仙舟舰队星徽", "s2_followup"),
            score_item_choice("治疗星徽", "s2_followup"),
        )
        self.assertGreater(
            score_item_choice("折刀", "s2_followup"),
            score_item_choice("未知散件", "s2_followup"),
        )

    def test_guide_badge_detection_and_tracked_shop_priority(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # BGR 橙金色徽标 (B=20, G=160, R=240 -> H~19, S~233, V~240)
        frame[78:105, 375:405] = (20, 160, 240)
        frame[848:870, 1138:1162] = (20, 160, 240)
        self.assertTrue(has_guide_badge(frame[70:115, 371:415]))
        self.assertEqual(detect_shop_guide_badges(frame), (True, False, False, False, False))
        self.assertEqual(detect_bench_guide_badges(frame), (False,) * 6 + (True, False, False))

        action, offer = decide_shop_action(
            gold=8, level=7, placed=7, capacity=7, bench_empty=2,
            offers=[ShopOffer(0, "砂金", 4), ShopOffer(1, "其他", 1)],
            target_names=("不相关",), refresh_cost=2, xp_cost=4,
            refreshes=0, ops=0, tracked_cols=(0,),
        )
        self.assertEqual(action, "buy")
        self.assertEqual(offer, ShopOffer(0, "砂金", 4))

        action, _ = decide_shop_action(
            gold=8, level=7, placed=7, capacity=7, bench_empty=2,
            offers=[ShopOffer(1, "景元", 1)],
            target_names=("景元",), refresh_cost=2, xp_cost=4,
            refreshes=0, ops=0, tracked_cols=(), guide_active=True,
        )
        self.assertEqual(action, "exp")

        # 备战区已有足够待上阵角色时，不应再买未追踪低费凑数卡
        action, _ = decide_shop_action(
            gold=6, level=9, placed=8, capacity=9, bench_empty=3,
            offers=[ShopOffer(0, "其他", 1)],
            target_names=("砂金",), refresh_cost=2, xp_cost=4,
            refreshes=0, ops=0, tracked_cols=(), guide_active=True, bench_chars=2,
        )
        self.assertEqual(action, "refresh")


if __name__ == "__main__":
    unittest.main()
