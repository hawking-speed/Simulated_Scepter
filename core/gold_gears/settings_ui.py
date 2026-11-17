"""黄金与机械暂时沿用寰宇命途的配置界面（运行时设置相同）。

黄金与机械自己的跑图计划（每局骰子 / 难度 / 命途 / 认知值目标）写在
user/core_settings/gold_gears.json 的 runs 里，目前直接改 JSON，不走界面。
"""

from core.any_fate.settings_ui import SettingsDialog as AnyFateSettingsDialog


class SettingsDialog(AnyFateSettingsDialog):
    pass
