"""触发器体检：用项目自己的 OCR 检查一张游戏画面能命中 insect.json 里的哪些触发器。

用途：游戏版本更新后界面会移位/改字，脚本就会卡在某个画面循环。与其等脚本卡住再逐个排查，
不如在每个画面上跑一次本工具，一次性看清：
  1. 按脚本的真实判定逻辑（整屏 OCR → 文字落在触发器框外扩 redundancy 像素内 → 包含目标文字），
     哪些触发器命中；列表里第一个命中且满足 condition 的就是脚本此刻会执行的动作；
  2. 「疑似移位」：目标文字在屏幕上出现了，但不在触发器框内——多半是新版本界面挪了位置，
     会给出文字实际所在的框，可直接用来修正 box；
  3. 全屏 OCR 原文，便于补新触发器。
图片触发器（photo）只列出，不做模板匹配。

用法：
    uv run python tool/scripts/trigger_audit.py 截图.png [--crop X,Y] [--state init] [--json 路径]
      --crop X,Y   截图是整个屏幕时，游戏客户区左上角在截图中的坐标（会裁出 1920x1080）
      --state S    模拟脚本当前 state，用于判断带 condition 的触发器
"""

import argparse
import json
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.common.ocr import My_TS  # noqa: E402
from tool.public_ocr import box_contain, merge_text, sort_text  # noqa: E402

DEFAULT_JSON = ROOT / "core" / "any_fate" / "actions" / "insect.json"


def load_image(path: str, crop: str | None) -> np.ndarray:
    img = cv.imdecode(np.fromfile(path, dtype=np.uint8), cv.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"读不了图片：{path}")
    if crop:
        x, y = (int(v) for v in crop.split(","))
        img = img[y:y + 1080, x:x + 1920]
    if img.shape[:2] != (1080, 1920):
        raise SystemExit(f"游戏画面必须是 1920x1080，当前 {img.shape[1]}x{img.shape[0]}；整屏截图请加 --crop")
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--crop")
    ap.add_argument("--state", default=None)
    ap.add_argument("--json", default=str(DEFAULT_JSON))
    ap.add_argument("--raw", action="store_true", help="打印全屏 OCR 原文")
    args = ap.parse_args()

    img = load_image(args.image, args.crop)
    actions = json.loads(Path(args.json).read_text(encoding="utf-8"))
    ts = My_TS()
    ts.forward(img)
    rows = ts.res

    print(f"== 全屏识别到 {len(rows)} 段文字")
    if args.raw:
        for r in sort_text(list(rows)):
            print(f"   {r['raw_text']!r:<24} box={r['box']}")

    first = None
    print("== 命中的触发器（按脚本执行顺序）")
    for idx, act in enumerate(actions):
        trig = act["trigger"]
        text = trig.get("text")
        if not text:
            continue
        box = trig["box"]
        red = trig.get("redundancy", 30)
        inside = [r for r in rows if box_contain(box, r["box"], redundancy=red)]
        merged = merge_text(sort_text(inside)) if inside else ""
        if text in merged:
            cond = trig.get("condition")
            ok = cond is None or cond == args.state
            mark = "执行" if ok and first is None else ("条件不满足" if not ok else "命中(排在后面)")
            if ok and first is None:
                first = act["name"]
            print(f"   [{idx:2d}] {act['name']:<10} 文字「{text}」 cond={cond} -> {mark}")

    print(f"== 脚本此刻会执行：{first or '（无文字触发器命中，可能落到图片触发器或刷「匹配不到任何图标」）'}")

    print("== 疑似移位（目标文字在屏幕上，但不在触发器框内）")
    shifted = 0
    for idx, act in enumerate(actions):
        trig = act["trigger"]
        text = trig.get("text")
        if not text:
            continue
        box = trig["box"]
        red = trig.get("redundancy", 30)
        hits = [r for r in rows if text in r["raw_text"]]
        if hits and not any(box_contain(box, r["box"], redundancy=red) for r in hits):
            shifted += 1
            where = "; ".join(f"{r['raw_text']!r}@{r['box']}" for r in hits[:3])
            print(f"   [{idx:2d}] {act['name']:<10} 框={box} 实际={where}")
    if not shifted:
        print("   无")

    photos = [f"[{i}] {a['name']}({a['trigger']['photo']})" for i, a in enumerate(actions) if a["trigger"].get("photo")]
    print("== 图片触发器（未检查）：" + "，".join(photos))
    return 0


if __name__ == "__main__":
    sys.exit(main())
