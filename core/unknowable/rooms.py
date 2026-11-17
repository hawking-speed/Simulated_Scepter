"""房间指纹 + 航点库 + 位姿推算（纯函数，离线可测）。

不可知域没有小地图。同一种房间每次进来出生点和朝向都一样，所以：
- 入口第一帧画面做成「指纹」（颜色分布 + 缩略图），认出是哪种房间；
- 认出来就按上次走通的路线（位置 + 途经折点）直接过去，不再原地扫视；
- 没见过的房间照旧扫视，走通后把路线记下来，越跑越熟。

坐标系：出生点为原点，出生朝向为 +y，右手边为 +x；朝向单位是度，向右转为正。
长度单位是「里程」：前进时光流扩张分量的累计值（见 motion.Odometer），同一房间同一路线上可重复。
"""

import json
import math
import os

import cv2 as cv
import numpy as np

VERSION = 2
THUMB = (24, 14)
HIST_BINS = (12, 4, 2)   # 亮度只分两档：对曝光变化不敏感


def _scene_mask(h, w):
    """指纹只看场景：去掉角色、左上角区域名、右侧队伍栏、右下角技能键、右上角菜单。"""
    m = np.full((h, w), 255, np.uint8)
    m[int(h * 0.32):, int(w * 0.36):int(w * 0.64)] = 0
    m[:int(h * 0.17), :int(w * 0.26)] = 0
    m[int(h * 0.15):int(h * 0.64), int(w * 0.84):] = 0
    m[int(h * 0.72):, int(w * 0.80):] = 0
    m[:int(h * 0.12), int(w * 0.78):] = 0
    return m


def fingerprint(screen):
    """1920x1080 BGR 画面 → {'hist': [...], 'thumb': [...]}（都是普通列表，可直接存 JSON）。"""
    small = cv.resize(screen, (192, 108), interpolation=cv.INTER_AREA)
    mask = _scene_mask(108, 192)
    hsv = cv.cvtColor(small, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv], [0, 1, 2], mask, HIST_BINS, [0, 180, 0, 256, 0, 256]).flatten()
    hist = hist / max(float(hist.sum()), 1.0)
    gray = cv.cvtColor(small, cv.COLOR_BGR2GRAY)
    gray[mask == 0] = int(gray[mask > 0].mean())
    thumb = cv.resize(gray, THUMB, interpolation=cv.INTER_AREA).astype(np.float32).flatten()
    thumb = (thumb - thumb.mean()) / (thumb.std() + 1e-3)
    return {'hist': [round(float(v), 4) for v in hist], 'thumb': [round(float(v), 2) for v in thumb]}


def similarity(a, b):
    """两个指纹的相似度 0~1：直方图交集和缩略图相关系数各占一半。"""
    ha, hb = np.asarray(a['hist'], np.float32), np.asarray(b['hist'], np.float32)
    ta, tb = np.asarray(a['thumb'], np.float32), np.asarray(b['thumb'], np.float32)
    inter = float(np.minimum(ha, hb).sum())
    corr = float((ta * tb).mean())
    return round(0.5 * inter + 0.5 * max(corr, 0.0), 3)


# ---------- 位姿推算 ----------
class Pose:
    """出生点坐标系下的位置和朝向，外加走过的折线。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.x = self.y = 0.0
        self.heading = 0.0
        self.track = [(0.0, 0.0)]
        self.valid = True      # 位姿可信：从出生点起一直在记
        self.anchor = 0        # 上一个目标在 track 里的下标（下一段路线从这里起算）

    def turn(self, degrees):
        self.heading = (self.heading + degrees + 180) % 360 - 180

    def move(self, dist, offset=0.0):
        """沿「当前朝向 + offset 度」走 dist 里程（左移 offset=-90，后退 offset=180）。"""
        if abs(dist) < 1e-6:
            return
        rad = math.radians(self.heading + offset)
        self.x += dist * math.sin(rad)
        self.y += dist * math.cos(rad)
        last = self.track[-1]
        if math.hypot(self.x - last[0], self.y - last[1]) >= 0.5:
            self.track.append((round(self.x, 2), round(self.y, 2)))

    def bearing_to(self, point):
        """到 point 的（需要转的角度, 距离）。"""
        dx, dy = point[0] - self.x, point[1] - self.y
        want = math.degrees(math.atan2(dx, dy))
        return (want - self.heading + 180) % 360 - 180, math.hypot(dx, dy)

    def mark_anchor(self):
        self.track.append((round(self.x, 2), round(self.y, 2)))
        self.anchor = len(self.track) - 1

    def path_since_anchor(self):
        return self.track[self.anchor:] + [(round(self.x, 2), round(self.y, 2))]


def simplify(points, tol=1.2):
    """折线抽稀（道格拉斯-普克）：保留绕障碍的折点，去掉小抖动。"""
    if len(points) <= 2:
        return list(points)
    (x1, y1), (x2, y2) = points[0], points[-1]
    seg = math.hypot(x2 - x1, y2 - y1)
    best, idx = 0.0, 0
    for i in range(1, len(points) - 1):
        px, py = points[i]
        d = abs((x2 - x1) * (y1 - py) - (x1 - px) * (y2 - y1)) / seg if seg > 1e-6 else math.hypot(px - x1, py - y1)
        if d > best:
            best, idx = d, i
    if best <= tol:
        return [points[0], points[-1]]
    return simplify(points[:idx + 1], tol)[:-1] + simplify(points[idx:], tol)


def steer_after_detour(remaining, lateral):
    """绕行横移了 lateral 里程（向右为正）后，要回到原目标该转多少度（限制在 ±40°）。"""
    if remaining <= 0.5:
        return 0.0
    return max(-40.0, min(40.0, -math.degrees(math.atan2(lateral, remaining))))


# ---------- 航点库 ----------
class RoomLibrary:
    """user/core_settings/unknowable_rooms.json（v2）：
    {"version": 2, "rooms": [{"id", "kind", "fp", "seen", "targets": {名字: {"pos", "via", "sec", "ok", "fail"}}}],
     "legacy": 旧版「房间类型 → 物件 → 朝向角度」}"""

    def __init__(self, path, match_min=0.72):
        self.path = path
        self.match_min = match_min
        self.rooms, self.legacy = [], {}
        self.load()

    def load(self):
        try:
            with open(self.path, encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = {}
        if data.get('version') == VERSION:
            self.rooms, self.legacy = data.get('rooms', []), data.get('legacy', {})
        else:
            self.rooms, self.legacy = [], data   # 旧版文件整个就是 legacy

    def save(self):
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
        tmp = self.path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump({'version': VERSION, 'rooms': self.rooms, 'legacy': self.legacy}, f, ensure_ascii=False)
        os.replace(tmp, self.path)

    def match(self, kind, fp):
        """同类型房间里最像的一个：(房间, 相似度)；不够像返回 (None, 最高相似度)。"""
        best, score = None, 0.0
        for room in self.rooms:
            if room['kind'] != kind:
                continue
            s = similarity(room['fp'], fp)
            if s > score:
                best, score = room, s
        return (best, score) if score >= self.match_min else (None, score)

    def add(self, kind, fp):
        n = sum(1 for r in self.rooms if r['kind'] == kind) + 1
        room = {'id': f"{kind}_{n:02d}", 'kind': kind, 'fp': fp, 'seen': 1, 'targets': {}}
        self.rooms.append(room)
        return room

    @staticmethod
    def waypoints(room, name):
        """去 name 的折点列表（最后一个是目标位置）；没有可用路线返回 None。"""
        t = (room or {}).get('targets', {}).get(name)
        if not t or t.get('fail', 0) >= 2:
            return None
        return [tuple(p) for p in t.get('via', [])] + [tuple(t['pos'])]

    @staticmethod
    def learn(room, name, path, seconds):
        """记下走通的路线。path 是从上一个目标（或出生点）到这个目标的折线。"""
        pts = simplify(path)
        old = room['targets'].get(name)
        pos = pts[-1]
        if old and old.get('fail', 0) < 2:
            # 已有路线：位置取新旧平均，折点保留用时更短的那次
            pos = ((pos[0] + old['pos'][0]) / 2, (pos[1] + old['pos'][1]) / 2)
            via = old['via'] if old.get('sec', 1e9) <= seconds else pts[1:-1]
            sec = min(old.get('sec', 1e9), seconds)
            ok = old.get('ok', 0) + 1
        else:
            via, sec, ok = pts[1:-1], seconds, 1
        room['targets'][name] = {'pos': [round(pos[0], 2), round(pos[1], 2)],
                                 'via': [[round(p[0], 2), round(p[1], 2)] for p in via],
                                 'sec': round(seconds if sec > 1e8 else sec, 1), 'ok': ok, 'fail': 0}

    @staticmethod
    def fail(room, name):
        t = (room or {}).get('targets', {}).get(name)
        if t:
            t['fail'] = t.get('fail', 0) + 1
