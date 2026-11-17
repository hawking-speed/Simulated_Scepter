import json
import unittest
from pathlib import Path
from unittest.mock import Mock

from core.common.engine import StateKernel
from core.currency.automation import SimulatedCurrency
from core.currency.engine import CurrencyWar
from core.currency.run_history import RUN_END_ACTION


class CurrencyWarActionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config_path = Path(__file__).resolve().parents[1] / "core" / "currency" / "actions" / "currencywar.json"
        with config_path.open(encoding="utf-8") as config_file:
            cls.actions = json.load(config_file)

    def test_all_action_handlers_exist_on_engine(self):
        missing = [
            (entry["name"], act)
            for entry in self.actions
            for act in entry["actions"]
            if isinstance(act, str) and not hasattr(CurrencyWar, act)
        ]
        self.assertEqual(missing, [])

    def test_popup_and_exit_events_trigger_once(self):
        once_names = {
            action["name"]
            for action in self.actions
            if action["trigger"].get("once")
        }
        self.assertEqual(once_names, {"开始退出"})

    def test_shop_open_precedes_board_preparation(self):
        names = [action["name"] for action in self.actions]
        self.assertLess(names.index("装备推荐弹层"), names.index("商店开启"))
        self.assertLess(names.index("角色详情弹层"), names.index("商店开启"))
        self.assertLess(names.index("商店开启"), names.index("出战前准备"))

    def test_s2_events_precede_generic_choice(self):
        names = [action["name"] for action in self.actions]
        generic_idx = names.index("通用节点选择")
        for s2_name in (
            "选择投资环境",
            "选择投资策略",
            "补给",
            "主角光环",
            "幕后茶歇",
            "遭遇节点",
            "机心巧匠改装",
            "奇兽花园",
            "骇入效果",
            "梦境交织共舞",
            "巡海游侠任命",
        ):
            self.assertIn(s2_name, names)
            self.assertLess(names.index(s2_name), generic_idx)

    def test_run_end_action_keeps_expected_name(self):
        action_names = {action.get("name") for action in self.actions}
        self.assertIn(RUN_END_ACTION, action_names)

    def test_rank_button_routes_to_s2_handle_rank(self):
        matches = [
            action
            for action in self.actions
            if action["trigger"].get("text") == "开始对局"
        ]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["actions"], ["s2_handle_rank"])

    def test_run_static_dispatches_shop_before_board(self):
        shop_action = next(a for a in self.actions if a["name"] == "商店开启")
        board_action = next(a for a in self.actions if a["name"] == "出战前准备")
        currency = object.__new__(SimulatedCurrency)
        StateKernel.__init__(currency)
        currency.state = "currency_main"
        currency.get_screen = Mock(return_value=255)
        currency.ts = Mock()
        currency.ts.find_with_box.return_value = [{"raw_text": "收起", "box": [1597, 967, 1644, 992]}]
        currency.do_action = Mock(return_value=1)
        currency._on_static_action_completed = Mock()

        result = currency.run_static(
            json_file={"商店": [shop_action], "备战": [board_action]}
        )

        self.assertEqual(result, ("商店开启", 1))
        currency.do_action.assert_called_once_with("s2_manage_shop")


if __name__ == "__main__":
    unittest.main()
