"""由寰宇蝗灾的 insect.json 生成黄金与机械的 nous.json（蝗灾脚本更新后重新运行一次即可）。

改动：
- 去掉寰宇蝗灾专属入口（开始探索按钮位置、章节页、扩展装置选卡）；
- 换成黄金与机械入口：扩展装置选卡 → 主界面（先自定义拼装选骰，再开始探索）→ 骰面装配；
- 「其他宇宙有进行中的存档」不再点「快捷结束并结算」，改为停内核，保护不可知域进度。

用法：.venv/Scripts/python.exe core/gold_gears/build_nous.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "any_fate", "actions", "insect.json")
DST = os.path.join(HERE, "actions", "nous.json")

DROP = {"开始寰宇蝗灾", "退出寰宇蝗灾章节", "进入寰宇蝗灾", "扩展装置选中寰宇蝗灾"}

ENTRY = [
    {"name": "骰面装配", "_comment": "黄金与机械：自定义拼装里切到本局骰子并核对右侧标题，选好后关掉",
     "trigger": {"text": "骰面装配", "box": [93, 65, 213, 95], "interval": 1, "redundancy": 30},
     "actions": ["gg_dice_assembly"]},
    {"name": "黄金与机械主界面", "_comment": "主界面「开始探索」在 (958,812)，比寰宇蝗灾的低 60 像素；没选骰子先去自定义拼装",
     "trigger": {"text": "开始探索", "box": [878, 795, 1038, 830], "interval": 2, "redundancy": 30},
     "actions": ["gg_main"]},
    {"name": "扩展装置选中黄金与机械", "_comment": "把「黄金与机械」卡片转到正中再点进去",
     "trigger": {"text": "扩展装置", "box": [53, 53, 210, 85], "interval": 2, "redundancy": 30},
     "actions": ["gg_pick_card"]},
    {"name": "误入寰宇蝗灾主界面", "_comment": "寰宇蝗灾主界面的「开始探索」在 (962,751)：退回扩展装置重新选卡",
     "trigger": {"text": "开始探索", "box": [872, 728, 1052, 774], "interval": 2, "redundancy": 30},
     "actions": [{"press": "esc"}]},
    {"name": "误入不可知域主界面", "_comment": "实机 2026-11-15：不可知域主界面按 Esc 退不出去，点右上角 X 回扩展装置",
     "trigger": {"text": "不可知域", "box": [100, 64, 205, 95], "interval": 2, "redundancy": 20},
     "actions": [{"position": [1863, 65]}]},
    {"name": "命途详情确认", "_comment": "实机 2026-11-15/16：点命途后进详情页（不管禁用槽是否已填满，页中「激活命途回响后」始终存在），核对命途/禁用后点确认",
     "trigger": {"text": "激活命途回响后", "box": [750, 575, 1350, 620], "interval": 1, "redundancy": 20},
     "actions": ["gg_fate_confirm"]},
    {"name": "骰子初始效果弹窗", "_comment": "实机 2026-11-15：进位面后弹「初始效果「行者共生」」，「点击空白处关闭」在 y≈743，原触发器框在底部够不着",
     "trigger": {"text": "点击空白处关闭", "box": [840, 725, 1080, 762], "interval": 2, "redundancy": 20},
     "actions": [{"position": [960, 1000]}]},
    {"name": "星神秘闻解锁弹窗", "_comment": "实机 2026-11-16：觐见星神解锁星神秘闻（「反有机战争」等）时「点击空白处关闭」在 y≈820",
     "trigger": {"text": "点击空白处关闭", "box": [840, 795, 1080, 855], "interval": 2, "redundancy": 20},
     "actions": [{"position": [960, 820]}]},
]


def main():
    with open(SRC, encoding="utf-8") as f:
        events = json.load(f)
    out = []
    inserted = False
    for ev in events:
        name = ev.get("name")
        if name in DROP:
            if not inserted:
                out.extend(ENTRY)
                inserted = True
            continue
        if name == "结束进行中的存档":
            ev = dict(ev)
            ev["_comment"] = "黄金与机械：不可知域有暂离存档时绝不快捷结束，停内核并写 tools/GG_BLOCKED.txt"
            ev["actions"] = ["gg_blocked_by_save"]
        if name == "祝福商店":
            ev = dict(ev)
            ev["_comment"] = "黄金与机械：在交易格祝福商店里挑选购买优先祝福，避免空手退出"
            ev["actions"] = ["gg_bless_shop"]
        if name == "开始游戏":
            # 实机 2026-11-15：黄金与机械难度页多了「认知区间」一行，「首次通关奖励」下移到 y≈367
            ev = json.loads(json.dumps(ev))
            ev["trigger"]["box"] = [1150, 220, 1400, 420]
        if name == "选择命途":
            # 实机 2026-11-16：黄金与机械「请选择想要踏上的命途」在 y≈139，原框 [837,155,1083,186] 偏低漏触发
            ev = json.loads(json.dumps(ev))
            ev["trigger"]["box"] = [810, 110, 1110, 195]
        out.append(ev)
    if not inserted:
        raise SystemExit("insect.json 里没找到寰宇蝗灾入口触发器，生成中止")
    with open(DST, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=4)
    print(f"已生成 {DST}：{len(events)} → {len(out)} 条触发器")


if __name__ == "__main__":
    main()
