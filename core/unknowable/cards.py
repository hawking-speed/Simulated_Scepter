"""事件牌识别（纯函数，离线可测）。

不可知域的事件物件是一块立着的画：深洋红色画框，里面是事件插画；还没揭开的那种里面是紫底白色「?」。
远处头顶没有白字标签，所以靠画框认：画框颜色围出的一个竖长方形（宽高比约 0.75）。
"""

import cv2 as cv
import numpy as np

MIN_H = 72            # 画框高度下限（像素，1080p）：再小就分不清了
ASPECT = (0.5, 1.0)   # 宽 / 高；斜着看会变窄
# 画框颜色每张牌不一样（见过洋红、蓝紫、棕、金、绿），所以按色相分段各找一遍：(色相下限, 上限)
BANDS = ((0, 16), (8, 24), (16, 32), (24, 40), (32, 52), (44, 70), (60, 95), (90, 122), (118, 140), (136, 166), (160, 180))


def _base_mask(hsv):
    """够鲜艳、不太暗也不太亮的像素（画框的饱和度和亮度范围），去掉界面。"""
    m = cv.inRange(hsv, (0, 45, 35), (180, 255, 190))
    m[:180, :500] = 0        # 左上角区域名
    m[:, 1620:] = 0          # 右侧队伍栏
    m[760:, 1500:] = 0       # 右下角技能键
    return m


def _cards_in(mask, hsv, question=False):
    """一个色相段里像画框的轮廓：竖长方形的一圈，圈外不是同色（独立立着），圈里比圈亮（插画）。
    question=True 专找「?」牌：画框常和周围的暗紫色连在一起，所以连内圈轮廓也看、不要求圈外干净，但里面必须是亮紫色。"""
    out = []
    contours, _ = cv.findContours(mask, cv.RETR_LIST if question else cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        if h < MIN_H or h > 900 or not ASPECT[0] <= w / h <= ASPECT[1]:
            continue
        if cv.contourArea(cv.convexHull(c)) < 0.78 * w * h:
            continue
        cx, cy = x + w // 2, y + h // 2
        if y + h < 200 or (h < 400 and 840 < cx < 1080 and cy > 430) or x + w >= 1615 or h > 700:
            continue   # 画面顶上的界面装饰；角色身上的衣服花纹；贴着右侧队伍栏的大片颜色
        t = max(4, int(0.15 * w))
        roi = mask[y:y + h, x:x + w] > 0
        top, bottom, left, right = (float(roi[:t].any(axis=0).mean()), float(roi[-t:].any(axis=0).mean()),
                                    float(roi[:, :t].any(axis=1).mean()), float(roi[:, -t:].any(axis=1).mean()))
        # 正前方的牌下沿会被角色的头挡住：这时不看下边
        hidden = 820 < cx < 1100 and y + h > 380
        sides = [top, left, right] if hidden else [top, bottom, left, right]
        if len(sides) == 4 and h < 150:
            sides = sorted(sides)[1:]   # 远处的牌允许一条边被挡（武器、栏杆、另一张牌）：只看最好的三条
        score = float(np.mean(sides))
        # 远处的小框要各边齐全（暗处的窗格、栏杆会凑出缺边的小方块）；走近后会被角色挡住一部分，放宽
        if score < (0.85 if h < 150 else 0.78) or min(sides) < (0.6 if h < 150 else 0.35):
            continue
        # 圈外一圈不能还是这个颜色：墙上的窗格、大片同色里的方块不算
        x1, y1, x2, y2 = max(x - t, 0), max(y - t, 0), min(x + w + t, 1920), min(y + h + t, 1080)
        outer = mask[y1:y2, x1:x2] > 0
        ring_out = (outer.sum() - roi.sum()) / max(outer.size - roi.size, 1)
        if ring_out > 0.3 and not question:
            continue
        # 圈里是插画：比画框亮
        iy, ix = slice(y + h // 4, y + h - h // 4), slice(x + w // 4, x + w - w // 4)
        iv = hsv[iy, ix][..., 2]
        edge = roi.copy()
        edge[t:-t, t:-t] = False   # 只看最外面一圈带子：那才是画框
        frame_v = hsv[y:y + h, x:x + w][..., 2][edge]
        if np.median(iv) < np.median(frame_v) + 15 and (mask[iy, ix] > 0).mean() > 0.4:
            continue
        ih, isat = hsv[iy, ix][..., 0], hsv[iy, ix][..., 1]
        purple = ((ih >= 118) & (ih <= 152) & (isat >= 28) & (iv >= 105)).mean()
        if question and purple < 0.5:
            continue
        out.append({'box': [int(x), int(y), int(x + w), int(y + h)], 'kind': '?' if purple >= 0.5 else '画', 'score': round(score, 2)})
    return out


def find_cards(screen):
    """1920x1080 BGR 画面 → 事件牌列表，从大到小：
    [{'box': [x1, y1, x2, y2], 'kind': '?' 或 '画', 'score': 画框完整度 0~1}]"""
    hsv = cv.cvtColor(screen, cv.COLOR_BGR2HSV)
    base, hue = _base_mask(hsv), hsv[..., 0]
    kernel = np.ones((3, 3), np.uint8)
    found = []
    for lo, hi in BANDS:
        mask = cv.morphologyEx(cv.bitwise_and(cv.inRange(hue, lo, hi), base), cv.MORPH_CLOSE, kernel)
        found += _cards_in(mask, hsv)
    # 「?」牌：画框是暗紫色，里面和周围的雾气是亮一些的紫色，色相跨两段。单独按「暗的紫色」再找一遍，框才不会和里外连成一片
    dark = cv.inRange(hsv, (120, 35, 30), (166, 150, 125))
    dark[:180, :500] = 0
    dark[:, 1620:] = 0
    dark[760:, 1500:] = 0
    found += _cards_in(cv.morphologyEx(dark, cv.MORPH_CLOSE, kernel), hsv, question=True)
    found.sort(key=lambda d: -(d['box'][3] - d['box'][1]))
    # 同一块牌在相邻色相段里会各认出一次：只留最大的
    keep = []
    for d in found:
        x1, y1, x2, y2 = d['box']
        if not any(k['box'][0] <= (x1 + x2) // 2 <= k['box'][2] and k['box'][1] <= (y1 + y2) // 2 <= k['box'][3] for k in keep):
            keep.append(d)
    return keep
