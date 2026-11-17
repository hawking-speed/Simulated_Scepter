"""前进监视：用两帧画面的光流判断「按着 W 时角色到底有没有在往前走」。

不可知域没有小地图，只能从画面判断：
- 正常前进：场景从画面中心向四周扩张，光流幅度大、径向分量为正；
- 正面顶墙：背景不动，光流幅度接近 0；
- 贴着墙斜着滑：画面整体横移，幅度不小，但径向扩张接近 0。
"""

import cv2 as cv
import numpy as np

W, H = 192, 108


def _valid_mask():
    m = np.ones((H, W), bool)
    m[int(H * 0.30):, int(W * 0.36):int(W * 0.64)] = False   # 角色身体
    m[:int(H * 0.17), :int(W * 0.25)] = False                # 左上角区域名、碎片
    m[int(H * 0.18):int(H * 0.62), int(W * 0.85):] = False   # 右侧队伍栏
    m[int(H * 0.72):, int(W * 0.82):] = False                # 右下角技能键
    m[:int(H * 0.12), int(W * 0.80):] = False                # 右上角菜单
    return m


MASK = _valid_mask()
_ys, _xs = np.mgrid[0:H, 0:W].astype(np.float32)
_cx, _cy = W / 2, H * 0.45   # 第三人称镜头下前进的扩张中心大约在角色头顶
_rx, _ry = _xs - _cx, _ys - _cy
_rn = np.sqrt(_rx ** 2 + _ry ** 2) + 1e-3
RX, RY = _rx / _rn, _ry / _rn


def small_gray(screen):
    """1920x1080 的 BGR 画面 → 192x108 灰度。"""
    return cv.cvtColor(cv.resize(screen, (W, H), interpolation=cv.INTER_AREA), cv.COLOR_BGR2GRAY)


def measure(prev, cur):
    """返回 (幅度, 扩张, 横移)：
    幅度 = 有效区域光流长度的中位数（像素 / 帧，192x108 尺度）；
    扩张 = 光流沿径向的平均分量（正 = 场景向外扩 = 在前进）；
    横移 = 水平光流的平均值（正 = 画面整体向右移）。"""
    flow = cv.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    fx, fy = flow[..., 0], flow[..., 1]
    mag = np.sqrt(fx ** 2 + fy ** 2)
    radial = fx * RX + fy * RY
    return float(np.median(mag[MASK])), float(radial[MASK].mean()), float(fx[MASK].mean())


class Odometer:
    """光流里程计：前进时把每两帧之间的径向扩张量累加起来当作「走了多远」。
    比「按键时间 × 固定速度」可靠：顶墙、贴墙滑的时候扩张量是 0，不会虚增；同一房间同一路线上数值可重复。
    rate 是最近的里程/秒，用来折算没有逐帧测量的移动（绕行时的后退、横移）。"""

    def __init__(self, rate=3.0):
        self.total = 0.0
        self.rate = rate

    def add(self, expand, dt, forward):
        """一帧的测量结果入账，返回这一帧前进的里程。"""
        if not forward or expand <= 0:
            return 0.0
        self.total += expand
        if dt > 0.05:
            self.rate = min(15.0, max(0.5, 0.9 * self.rate + 0.1 * expand / dt))
        return expand

    def timed(self, seconds):
        """没有逐帧测量的移动：按最近的速度折算。"""
        return seconds * self.rate
