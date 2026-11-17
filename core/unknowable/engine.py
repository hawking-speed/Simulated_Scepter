"""模拟宇宙：不可知域 自动化。

位面内（区域识别、传送门、战斗、事件）沿用差分宇宙引擎；这里只处理不可知域特有的部分：
入口（扩展装置 → 不可知域 → 关卡）、权杖/组件选择、交易/精英区域，以及卡住时用「暂离」
代替差分引擎的「结束并结算」（不可知域要完整通关才有奖励）。
"""

import bisect
import collections
import glob
import json
import math
import os
import re
import time

import cv2 as cv
import numpy as np
import pyautogui

import core.divergent.keyops as keyops
from core.divergent.config import config
from core.divergent.engine import DivergentUniverse
from core.unknowable import cards, motion, rooms
from tool.GLOBAL import key_mouse_manager
from tool.log import CUS_LOGGER
from tool.mask import MASK
from tool.public_ocr import clean_text, load_actions, merge_text
from tool.storage import resource_path

# 选择权杖界面三张卡的中心 x；卡片右上角的「书本」图标表示图鉴里还没有（新）。
CARD_XS = (490, 960, 1430)
# 选择界面卡片中心：3 张 / 2 张 / 1 张时的位置（实拍）
CARD_LAYOUTS = ((490, 960, 1430), (725, 1193), (960,))
AREA_TYPES = ['事件', '奖励', '遭遇', '交易', '商店', '首领', '精英', '战斗', '财富', '休整', '谬误']
# 不可知域新增的区域按差分宇宙里行为最接近的区域处理
AREA_ALIAS = {'交易': '商店', '精英': '战斗', '谬误': '首领'}  # 谬误：第三位面首领房间里通往第四位面的洋红色门
HOLD_FILE = os.path.join("user", "core_settings", "unknowable_hold_start")  # 存在时开局前停下
PLANE4_HOLD = os.path.join("user", "core_settings", "unknowable_hold_plane4")  # 存在时第三位面首领打完先停下（调查第四位面入口）
AREA_HOLD = os.path.join("user", "core_settings", "unknowable_hold_area")  # 存在时：进入区域后停在出生点，等人工调查
BENCH_HOLD = os.path.join("user", "core_settings", "unknowable_hold_bench")  # 存在时：打开造物调试台就停下，等人工调查界面
SETTLE_FILE = os.path.join("user", "core_settings", "unknowable_settle_now")  # 存在时：一回到位面就结束并结算本局（交接前清掉暂离存档），点完确认后自动删除
LOOT_WORDS = ('战利品', '宝箱')  # 财富区域宝箱的 F 交互文字
TPL_DIR = str(resource_path(__file__, "tpl"))
ROOMS_FILE = os.path.join("user", "core_settings", "unknowable_rooms.json")  # 航点库：房间指纹 + 各目标的位置和折点
MOVE_OFFSET = {'move_w': 0, 'move_d': 90, 'move_s': 180, 'move_a': -90}  # 按方向键时相对镜头朝向的移动方向（度）
TARGET_THRESHOLD = {'宝箱': 0.65, '扭蛋机': 0.6}
TARGET_TPL = {'宝箱': 'chest_*.png', '扭蛋机': 'gacha_*.png'}
FLIGHT_DIR = os.path.join("logs", "flight")  # 黑匣子：出问题时把最近 20 秒画面拼成连环图
BATTLE_WORDS = ('总伤害', '我方行动', '敌方行动', '行动中')
STUCK_LIMIT = 3  # 同一区域暂离重进几次仍过不去就结束本局
EMPTY_SLOT_TPL = str(resource_path(__file__, "empty_slot.png"))
UNKNOWN_DIR = os.path.join("logs", "unknowable_unknown")
AREA_RE = re.compile(r'(\d)\D{0,2}(\d)\D{0,2}第(.)位面')  # 「(3/4)第一位面」去掉标点后是「34第一位面」
EVENT_WORDS = ('事件', '奖励', '遭遇', '交易')      # 事件类物件的 F 交互文字
PORTAL_WORDS = ('区域', '结束', '退出', '离开')      # 传送门的 F 交互文字
AREA_CSV = os.path.join("logs", "unknowable_areas.csv")  # 每个区域的用时、绕行次数、是否卡住
NAV_FILE = os.path.join("user", "core_settings", "unknowable_nav.json")
# 寻路阈值（192x108 光流，速度折算到 0.3 秒一帧）。2026-11-14 实测：
#   前进：速度 0.6~2.0，扩张/幅度 0.8~2（偶尔一帧 0.2）；顶墙：速度 0.01~0.05；贴墙滑：速度 3~5，扩张/幅度 -0.4~0.4
NAV_DEFAULT = {
    'forward_speed_min': 0.2,         # 速度低于它不算在前进
    'forward_expand_ratio_min': 0.45,  # 扩张/幅度 低于它不算在前进（是横着滑）
    'blocked_speed_max': 0.12,        # 连续 blocked_frames 帧速度都低于它 = 正面顶住
    'blocked_frames': 3,
    'sliding_frames': 4,              # 连续这么多帧都「不算在前进」= 贴着障碍滑
    'drift_side_min': 0.3,            # 平均横移超过它才按滑动方向选绕行的一侧
    'fight_seek_seconds': 20,         # 战斗区域朝正前方找怪最多走多久
    'event_seek_seconds': 18,         # 交互区域朝正前方找事件最多走多久
    'gacha_draws': 3,                 # 财富区域扭蛋机抽几次（0 = 不抽）
    'f_repeat_max': 3,                # 走路时一直按 F：同一个提示文字在一个区域里最多触发几次（防止可反复交互的东西卡死）
    'f_area_max': 6,                  # 一个区域里 F 累计触发几次后不再盲按
    'sift_min_matches': 15,           # 特征点识别：聚在一起的匹配点至少这么多才算认出物件
    'tpl_max_per_target': 12,         # 每种物件最多保留多少张模板（含自动收的）
    'route_enable': 1,                # 认出房间后按记下的航线走（0 = 关闭，全部照旧扫视）
    'room_match_min': 0.72,           # 入口指纹相似度达到它才算同一个房间（实测同房间 0.78~0.98，不同房间 ≤0.65）
    'route_terminal': 3.0,            # 离目标还有这么多里程时停止按航线走，交给视觉对准（约 3～5 米）
    'odo_rate': 3.0,                  # 里程/秒的初值（之后按实测更新），折算后退、横移用
    'near_lost_size': 150,            # 目标框这么大（已在跟前）时从侧面跟丢：停下回头找，不再蒙头直走
    'save_samples': 1,                # 存全分辨率样本到 logs/samples（区域入口、F 交互成功时），供离线改进识别
}
# 区域按行为分组：同一组用同一套寻路
AREA_GROUP = {'战斗': '战斗组', '事件': '交互组', '奖励': '交互组', '遭遇': '交互组',
              '财富': '财富组', '商店': '过路组', '休整': '过路组', '首领': '首领组'}
# 原地扫视的朝向顺序（相对进区朝向，度）：先正前方，再左右各一眼（前方 ±90° 内），最后背后
SAMPLE_DIR = os.path.join("logs", "samples")
SCAN_FRONT = (0, -55, 55)
SCAN_BACK = (-110, 110, 180)


class UnknowableDomain(DivergentUniverse):
    def __init__(self, debug=0, nums=-1, speed=0):
        super().__init__(debug, nums, speed)
        self.default_json_path = str(resource_path(__file__, "actions/unknowable.json"))
        self.default_json = load_actions(self.default_json_path)
        self.unknown_since = None
        self.unknown_saved = False
        self.unknown_esc = 0
        self.runs_done = 0
        self.need_claim = True
        self.cur_stage = None   # 主界面最近一次点进去的关卡名
        self.fallacy = None     # 第三位面首领房间找谬误门（第四位面入口）的进度：None / 'scanned' 转过一圈 / 'seen' 看到过
        self.last_area_key = ''
        self.stuck_area = ''
        self.stuck_count = 0
        self.entered_by_portal = False  # 刚从传送门走进来＝新区域，战斗区域不用先转圈找门
        self.empty_slot_tpl = None
        self.bb_frames = collections.deque(maxlen=40)  # (时间, 缩小画面, 当前步骤, 标记框, 按键)
        self.bb_last = 0
        self.bb_sample = time.time()
        self.bb_note = ''
        self.bb_marks = []
        self.bb_keys = []
        self.target_tpls = {name: [cv.imread(p) for p in sorted(glob.glob(os.path.join(TPL_DIR, pat)))]
                            for name, pat in TARGET_TPL.items()}
        self.sift = cv.SIFT_create(contrastThreshold=0.02)
        self.target_feats = {}
        for name in self.target_tpls:
            self.load_target_feats(name)
        self.last_found = None                    # 最近一次认出的物件：(名字, 画面, 框, 匹配点数)，交互成功后收作模板
        self.run_trait = self.load_state().get('run_trait')
        self.mv_prev = None                       # 前进监视：上一帧缩小灰度图
        self.mv_time = 0
        self.nav = self.load_nav()
        self.mv_hist = collections.deque(maxlen=8)  # 最近几帧 (速度, 是否在前进, 横移)
        self.fg_paused = False                    # 游戏刚被切到后台过：按住的键已全部松开
        self.area_t0 = time.time()
        self.area_kind = ''
        self.area_detours = 0
        self.area_stuck = 0
        self.area_frames = collections.deque(maxlen=60)  # 整个区域的低帧率录像（2 秒一帧），慢区域结束时存图
        self.area_frame_t = 0
        self.unknown_area_cnt = 0
        self.fight_started = False                # 本区域真的开过战（开过战就正常找门）
        self.f_count = collections.Counter()      # 本区域每个提示文字被 F 触发过几次（'?' = 没读到文字时盲按触发的）
        self.f_last = (0, '?')                    # 最近一次按 F 的时间和当时的提示文字
        self.pose = rooms.Pose()                  # 出生点坐标系下的位置、朝向和走过的折线
        self.pose.valid = False
        self.odo = motion.Odometer(self.nav['odo_rate'])
        self.room_lib = rooms.RoomLibrary(ROOMS_FILE, self.nav['room_match_min'])
        self.room = None                          # 认出来的房间（航点库里的一条）
        self.room_fp = None                       # 本区域入口画面的指纹
        self.at_spawn = False                     # 角色此刻确定站在区域出生点（刚进门 / 刚暂离重进 / 刚开局）
        self.respawn_pending = False              # 刚暂离重进：区域编号没变，但人回到了出生点
        self.room_pending = False                 # 刚进区域，等区域类型读出来后取入口指纹
        self.route_used = {}                      # 本区域按航线走过的目标
        self.route_detoured = False               # 按航线走的途中绕行过：要重新对准航点
        self.entry_cards = []                     # 入口画面里看到的事件牌：[{'ang': 相对出生朝向的角度, 'h': 画框高度, 'kind'}]
        os.makedirs(UNKNOWN_DIR, exist_ok=True)

    # ---------- 黑匣子 ----------
    @staticmethod
    def game_foreground():
        try:
            import win32gui
            return '星穹铁道' in win32gui.GetWindowText(win32gui.GetForegroundWindow())
        except Exception:
            return True

    def wait_foreground(self):
        """游戏被别的窗口盖住时（例如 Claude 窗口），截到的是别人的画面，按键也会打进别的窗口：
        先松开所有移动键，暂停到游戏回到前台。"""
        if self.game_foreground():
            return
        for k in ('move_w', 'move_a', 'move_s', 'move_d', 'sprint'):
            try:
                keyops.keyUp(k)
            except Exception:
                pass
        self.fg_paused = True
        CUS_LOGGER.warning("游戏不在前台，暂停操作等待切回")
        while not self._stop and not self.game_foreground():
            time.sleep(0.5)
        time.sleep(0.5)
        CUS_LOGGER.info("游戏已回到前台，继续")

    def get_screen(self):
        self.wait_foreground()
        scr = super().get_screen()
        now = time.time()
        if scr is not None and now - self.bb_last >= 0.5:
            self.bb_last = now
            try:
                small = cv.resize(np.array(scr), (320, 180))
                self.bb_frames.append((now, small, self.bb_note, list(self.bb_marks), list(self.bb_keys)))
            except Exception:
                pass
            if now - self.area_frame_t >= 2:
                self.area_frame_t = now
                self.area_frames.append((now, small, self.bb_note, [], []))
            self.bb_marks, self.bb_keys = [], []
        return scr

    def press(self, key, *args, **kwargs):
        self.bb_keys.append(key)
        if key in MOVE_OFFSET and args and isinstance(args[0], (int, float)):
            # 定时移动（绕行的后退 / 横移等）没有逐帧测量：按最近速度折算进位姿
            self.pose.move(self.odo.timed(args[0]) * 0.9, MOVE_OFFSET[key])
        return super().press(key, *args, **kwargs)

    def turn(self, degrees, fine=1):
        self.pose.turn(degrees)
        return super().turn(degrees, fine)

    def note(self, text):
        """记下当前在做什么，黑匣子每帧都带上这句。"""
        self.bb_note = text

    def mark(self, box, label):
        self.bb_marks.append((box, label))

    def dump_flight(self, reason, frames=None):
        """把最近 20 秒的画面拼成 6x5 连环图（每格标注时间、步骤、识别框、按键），存到 logs/flight。
        frames 给了就用它（区域低帧率录像，最多取均匀的 30 帧）。"""
        if frames is None:
            frames = list(self.bb_frames)[-30:]
        elif len(frames) > 30:
            frames = [frames[int(i * (len(frames) - 1) / 29)] for i in range(30)]
        if not frames:
            return
        try:
            os.makedirs(FLIGHT_DIR, exist_ok=True)
            cell_w, cell_h, cap = 320, 180, 34
            sheet = np.zeros(((cell_h + cap) * 5, cell_w * 6, 3), np.uint8)
            t_end = frames[-1][0]
            for i, (t, img, note, marks, keys) in enumerate(frames):
                img = img.copy()
                for box, label in marks:
                    x1, y1, x2, y2 = [int(v / 6) for v in box]
                    cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 1)
                r, c = divmod(i, 6)
                y0 = r * (cell_h + cap)
                sheet[y0:y0 + cell_h, c * cell_w:(c + 1) * cell_w] = img
                line1 = f"{t - t_end:+.1f}s {' '.join(keys)}"
                line2 = ' '.join(f"{lb}" for _, lb in marks)
                self.put_text(sheet, line1, (c * cell_w + 3, y0 + cell_h + 14))
                self.put_text(sheet, (note + ' ' + line2)[:40], (c * cell_w + 3, y0 + cell_h + 30))
            path = os.path.join(FLIGHT_DIR, time.strftime("%m%d_%H%M%S_") + reason + ".jpg")
            cv.imencode('.jpg', sheet)[1].tofile(path)
            CUS_LOGGER.warning(f"黑匣子已保存 {path}")
        except Exception as e:
            CUS_LOGGER.error(f"黑匣子保存失败：{e}")

    @staticmethod
    def put_text(img, text, org):
        """中文标注：用 PIL 画字（cv.putText 不支持中文）。"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            pil = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            font = ImageFont.truetype("msyh.ttc", 13)
            ImageDraw.Draw(pil).text((org[0], org[1] - 13), text, font=font, fill=(255, 255, 255))
            img[:] = cv.cvtColor(np.array(pil), cv.COLOR_RGB2BGR)
        except Exception:
            cv.putText(img, text.encode('ascii', 'ignore').decode(), org, cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # ---------- 主循环 ----------
    def route(self):
        while not self._stop:
            self.loop()
        CUS_LOGGER.info("停止运行")

    def in_run_text(self):
        return clean_text(self.ts.ocr_one_row(self.screen, [50, 3, 420, 35]), char=0)

    def loop(self):
        if time.time() - self.bb_sample > 480:
            # 例行抽查：不管有没有出错，每 8 分钟留一张连环图，巡检时主动看有没有异常行为
            self.bb_sample = time.time()
            self.dump_flight("例行抽查")
        self.ts.forward(self.get_screen())
        res = self.run_static()
        if res:
            self.unknown_since = None
            self.last_action_time = time.time()
            return
        area_text = self.in_run_text()
        if '位面' in area_text or '区域' in area_text:
            self.unknown_since = None
            self.area()
            self.last_action_time = time.time()
            return
        # 4.7 战斗界面左下角的自动战斗图标位置（与寰宇蝗灾 insect.json 相同）；旧位置作兜底
        if self.check("auto_battle", 21, 919, threshold=0.9) or self.check("auto_battle", 23, 969, threshold=0.925):
            self.da_hei_ta_effecting = False
            self.press("V")
            return
        self.on_unknown_screen()

    @staticmethod
    def cjk_count(text):
        return sum('一' <= ch <= '鿿' for ch in text)

    def handle_dialogue(self):
        """剧情对话（通关后黑塔等）：底部有字幕就点屏幕推进；右侧出现选项就点最后一个。"""
        speaker = merge_text(self.ts.find_with_box([700, 790, 1220, 848], redundancy=0))
        subtitle = merge_text(self.ts.find_with_box([300, 845, 1620, 905], redundancy=0))
        # 对话画面：字幕上方居中有说话人名字；战斗画面底部也有字，只看字幕会误判
        if self.cjk_count(subtitle) < 4 or not 1 <= self.cjk_count(speaker) <= 8:
            return False
        options = [i for i in self.ts.find_with_box([1300, 500, 1900, 830], redundancy=0)
                   if self.cjk_count(i['raw_text']) >= 3]
        if options:
            CUS_LOGGER.info(f"剧情对话：选择选项 {options[-1]['raw_text']}")
            self.click_box(options[-1]['box'])
        else:
            self.click_point(960, 960)
        time.sleep(1.2)
        return True

    def in_battle_screen(self):
        """战斗画面：有「总伤害」「我方行动中」这类战斗界面文字。"""
        text = merge_text(self.ts.find_with_box([0, 0, 1920, 1080], redundancy=0))
        return any(k in text for k in BATTLE_WORDS)

    def confirm_plane4(self):
        """进谬误门后的「第四位面开启」确认框（11-15 19:17 截图）：「额外轮次计数：N，需要消耗 1 额外轮次计数，确认是否前往
        （在第四位面战斗失败仍然视为成功通关）」，下方「取消」「确认前往」。计数够就点确认前往；计数为 0 点取消，本区域改走退出门。"""
        self.ts.forward(self.get_screen())
        btn = next((i for i in self.ts.find_with_box([900, 600, 1600, 900], redundancy=0) if '确认前往' in i['raw_text']), None)
        if btn is None:
            return False
        head = self.ts.find_with_box([500, 250, 1700, 600], redundancy=0)
        if not any('额外轮次' in i['raw_text'] or '第四位面' in i['raw_text'] for i in head):
            return False
        count = next((int(m.group(1)) for i in head if (m := re.search(r'额外轮次计数\D{0,2}(\d+)', i['raw_text']))), None)
        if count == 0:
            CUS_LOGGER.warning("第四位面开启：额外轮次计数为 0，点取消，本区域改走退出门")
            self.fallacy = 'skip'
            self.click_text_in(['取消'], [300, 600, 1000, 900])
        else:
            CUS_LOGGER.info(f"第四位面开启：额外轮次计数 {count}，点「确认前往」")
            st = self.load_state()
            if st.get('run_nandu', 1) >= 3:
                self.save_state(n3_tries=st.get('n3_tries', 0) + 1)
            self.click_box(btn['box'])
        time.sleep(2)
        return True

    def on_unknown_screen(self):
        now = time.time()
        if self.in_battle_screen():
            # 自动战斗在打：什么都不点。以前每 60 秒点 (960,1000) 会点到底部角色头像，触发终结技又取消
            self.note("战斗中，等待")
            self.unknown_since = None
            self.last_action_time = now
            return
        if self.handle_dialogue() or self.confirm_plane4():
            self.unknown_since = None
            self.last_action_time = now
            return
        if self.unknown_since is None:
            self.unknown_since = now
            self.unknown_saved = False
            return
        if now - self.unknown_since > 6 and not self.unknown_saved:
            self.unknown_saved = True
            path = os.path.join(UNKNOWN_DIR, time.strftime("%m%d_%H%M%S") + ".png")
            cv.imwrite(path, np.array(self.screen))
            text = merge_text(self.ts.find_with_box([0, 0, 1920, 1080], redundancy=0))
            CUS_LOGGER.warning(f"未识别界面已保存 {path}，文字：{text[:200]}")
        if now - self.unknown_since > 12 and now - self.unknown_esc > 15:
            # 没有对应处理的界面（例如盲按 F 打开的装置界面）：按 Esc 关掉，回到位面继续
            self.unknown_esc = now
            CUS_LOGGER.info("未识别界面持续 12 秒，按 Esc 关闭")
            self.press('esc')
            time.sleep(1.5)
            return
        if now - self.last_action_time > 60:
            # 过场 / 弹窗：点屏幕上方空白处推进。不能点底部——战斗画面底部是角色头像，点了会放终结技
            self.click_point(960, 170)
            self.last_action_time = now

    # ---------- 区域 ----------
    def get_now_area(self, deep=0):
        team_member = self.find_team_member(self.team_member)
        self.area_text = self.in_run_text()
        if not ('位面' in self.area_text or '区域' in self.area_text):
            return None
        if team_member != self.team_member:
            self.team_member = team_member
            for name in self.team_member:
                if name in config.long_range_list:
                    self.long_range = f"role_{self.team_member[name] + 1}"
                    break
        tail = self.area_text.split('区域')[-1]
        res = None
        for kind in AREA_TYPES:
            if kind[:1] in tail:
                res = kind
                break
        if res is None:
            for kind in AREA_TYPES:
                if kind in self.area_text:
                    res = kind
                    break
        if res is None and deep == 0:
            self.turn(20)
            scr = self.screen
            time.sleep(0.3)
            self.get_screen()
            self.turn(-20)
            res = self.get_now_area(deep=1)
            self.screen = scr
        CUS_LOGGER.debug(f"区域文字：{self.area_text} → {res}")
        return AREA_ALIAS.get(res, res) if res else '位面'

    def area_key(self, text=None):
        """区域编号「n/m-位面」，例如 (3/4)第一位面 → '3/4-一'；读不出来返回 ''。"""
        m = AREA_RE.search(self.in_run_text() if text is None else text)
        return f"{m.group(1)}/{m.group(2)}-{m.group(3)}" if m else ''

    def area_status(self, start_key):
        """对当前画面：'same' 还在原区域 / 'changed' 区域编号变了 / 'gone' 区域名不见了（战斗、过场、界面）。"""
        text = self.in_run_text()
        if not ('位面' in text or '区域' in text):
            return 'gone'
        key = self.area_key(text)
        return 'changed' if key and start_key and key != start_key else 'same'

    def area_record(self, key, kind):
        """区域换了：把上一个区域的用时、绕行次数记进 CSV；慢的区域（>75 秒）另存整段低帧率录像。"""
        now = time.time()
        if self.last_area_key and self.area_kind:
            secs = now - self.area_t0
            try:
                new = not os.path.exists(AREA_CSV)
                with open(AREA_CSV, 'a', encoding='utf-8-sig') as f:
                    if new:
                        f.write("时间,区域,类型,用时秒,绕行次数,卡住次数\n")
                    f.write(f"{time.strftime('%m-%d %H:%M:%S')},{self.last_area_key},{self.area_kind},"
                            f"{secs:.0f},{self.area_detours},{self.area_stuck}\n")
            except OSError:
                pass
            CUS_LOGGER.info(f"区域完成 {self.last_area_key} {self.area_kind}：用时 {secs:.0f}s，"
                            f"绕行 {self.area_detours} 次，卡住 {self.area_stuck} 次")
            if secs > 75 and (self.area_kind not in ('战斗', '首领') or self.area_detours or self.area_stuck):
                self.dump_flight(f"慢区域_{self.area_kind}_{secs:.0f}s", frames=list(self.area_frames))
        self.area_t0, self.area_kind = now, kind
        self.area_detours = self.area_stuck = 0
        self.area_frames.clear()

    def area(self):
        area_now = self.get_now_area()
        # 区域编号「(n/m)第X位面」变了就是新区域：走门时可能没按 F 就被传送了，状态要复位
        key = self.area_key(self.area_text)
        if key and key != self.last_area_key:
            CUS_LOGGER.info(f"进入新区域 {self.area_text}（{area_now}），复位区域状态")
            if self.last_area_key:
                self.init_floor()
            self.area_record(key, area_now or '')
            self.fight_started = False
            self.fallacy = None
            self.f_count.clear()
            self.save_sample(f"入口_{area_now or '未知'}")
            self.last_area_key = key
            self.respawn_pending = False
            self.room_pending = True
            self.room, self.pose.valid = None, False
        elif key and self.respawn_pending:
            # 暂离重进：同一个区域、回到出生点，位姿清零重新记
            self.respawn_pending = False
            self.at_spawn = self.room_pending = True
        if area_now in (None, '位面'):
            # 区域类型没读出来（文字被特效挡住等）：多读几次，实在不行就直接找门，不暂离
            self.unknown_area_cnt += 1
            if self.unknown_area_cnt < 6:
                time.sleep(0.5)
                return 0
            CUS_LOGGER.warning(f"区域类型一直读不出来（{self.area_text}），直接找门")
            self.unknown_area_cnt = 0
            self.portal_opening_days(static=1)
            return 1
        self.unknown_area_cnt = 0
        if not self.area_kind:
            self.area_kind = area_now
        self.area_now = area_now
        group = AREA_GROUP.get(area_now)
        if self.room_pending:
            # 区域类型读出来了、人还没动：取入口指纹认房间
            self.room_pending = False
            self.room_enter(area_now)
        if os.path.exists(SETTLE_FILE):
            CUS_LOGGER.warning("unknowable_settle_now 存在：结束并结算本局，不留暂离存档")
            self.stuck_count, self.stuck_area = STUCK_LIMIT - 1, self.area_text
            self.close_and_exit()
            return 1
        if os.path.exists(AREA_HOLD):
            CUS_LOGGER.warning("unknowable_hold_area 存在：进入区域后停在出生点等待调查（删掉该文件后重启继续）")
            self._stop = True
            return 1
        if group == '首领组':
            return super().area()
        self.check_dead()
        CUS_LOGGER.info(f"区域 {key} {area_now}（{group}） state:{self.area_state}")
        if group == '战斗组':
            if self.area_state == 0:
                return self.fight_area()
            if self.area_state == 1:
                # 打完才有带字的门。没有就说明刚才没开战：再找一次怪，不去追没激活的门
                self.area_state = 2
                if self.portal_opening_days(static=1, need_label=not self.fight_started) is False:
                    CUS_LOGGER.info("战斗区域：没看到带字的门，敌人可能还在，再找一次怪")
                    self.entered_by_portal = True
                    self.fight_area()
                    self.area_state = 2
                return 1
        elif group == '财富组':
            if self.area_state == 0:
                return self.wealth_area()
        elif group == '交互组':
            if self.area_state < 2:
                return self.interact_area()
        # 过路组（商店 / 休整）没有必须做的事，直接找门；其他组做完事也走到这里
        self.portal_opening_days(static=1)
        return 1

    # ---------- 前进监视 + 直走闭环 ----------
    def load_nav(self):
        """寻路阈值：默认值是 2026-11-14 实测标定的，user/core_settings/unknowable_nav.json 里可以改；文件不存在就生成。"""
        nav = dict(NAV_DEFAULT)
        try:
            with open(NAV_FILE, encoding='utf-8') as f:
                nav.update({k: v for k, v in json.load(f).items() if k in NAV_DEFAULT})
        except (OSError, ValueError):
            try:
                os.makedirs(os.path.dirname(NAV_FILE), exist_ok=True)
                with open(NAV_FILE, 'w', encoding='utf-8') as f:
                    json.dump(nav, f, ensure_ascii=False, indent=2)
            except OSError:
                pass
        return nav

    def motion_reset(self):
        self.mv_prev = None
        self.mv_hist.clear()

    def motion_state(self):
        """按着 W 时每帧调用一次（用 self.screen）：
        'ok' 在前进 / 'blocked' 正面顶住不动 / 'sliding' 贴着障碍横着滑。
        阈值来自实测（192x108、0.3 秒一帧）：前进 幅度 0.6~2、扩张/幅度 1~2；顶墙 幅度 0.03；
        贴墙滑 幅度 3~5、扩张/幅度 -0.4~0.4。"""
        now = time.time()
        g = motion.small_gray(np.array(self.screen))
        if self.mv_prev is None or now - self.mv_time > 1.2:
            self.mv_prev, self.mv_time = g, now
            return 'ok'
        dt = now - self.mv_time
        if dt < 0.2:
            return 'ok'
        mag, expand, shift = motion.measure(self.mv_prev, g)
        self.mv_prev, self.mv_time = g, now
        nav = self.nav
        rate = mag * 0.3 / dt
        forward = rate > nav['forward_speed_min'] and expand > nav['forward_expand_ratio_min'] * mag
        self.mv_hist.append((rate, forward, shift))
        self.pose.move(self.odo.add(expand, dt, forward))
        self.bb_keys.append(f"v{rate:.1f}e{expand / max(mag, 0.01):+.1f}")
        hist = list(self.mv_hist)
        if len(hist) >= nav['blocked_frames'] and all(r < nav['blocked_speed_max'] for r, _, _ in hist[-nav['blocked_frames']:]):
            return 'blocked'
        if len(hist) >= nav['sliding_frames'] and not any(f for _, f, _ in hist[-nav['sliding_frames']:]):
            return 'sliding'
        return 'ok'

    def drift(self):
        """最近几帧画面的平均横移：>0 画面往右移＝人在往左滑＝左边是空的。"""
        hist = list(self.mv_hist)
        return sum(s for _, _, s in hist) / len(hist) if hist else 0.0

    def detour_side(self, n, drift=0.0):
        """这次往哪边绕。画面横移有明显方向就听它的（那边是空的），并记住；
        没有线索时沿用这段路上记住的方向，从没有过线索就和上一次反着来（第一次向右）。"""
        if n == 0:
            self.detour_log, self.detour_hint = [], None
        lim = self.nav['drift_side_min']
        hint = 'move_a' if drift > lim else 'move_d' if drift < -lim else None
        if hint:
            # 同一侧已经连着绕了两次还没过去：线索不可信，换另一侧
            if self.detour_log[-2:] == [hint, hint]:
                hint = 'move_d' if hint == 'move_a' else 'move_a'
            self.detour_hint = hint
        if hint:
            side = hint
        elif getattr(self, 'detour_hint', None):
            side = self.detour_hint
        elif getattr(self, 'detour_log', None):
            side = 'move_d' if self.detour_log[-1] == 'move_a' else 'move_a'
        else:
            side = 'move_d'
        self.detour_log = getattr(self, 'detour_log', []) + [side]
        return side

    def detour(self, n, drift=0.0):
        """像人一样绕开障碍：退一步 → 往空的一侧横移 → 往前走过墙角。n 是这段路上第几次绕（从 0 起）。"""
        keyops.keyUp('move_w')
        keyops.keyUp('sprint')
        side = self.detour_side(n, drift)
        dur = 0.9 + 0.5 * (n // 2)
        self.area_detours += 1
        CUS_LOGGER.info(f"被挡住（第 {n + 1} 次，横移 {drift:+.1f}）：后退，向{'左' if side == 'move_a' else '右'}绕 {dur:.1f}s，再前进")
        self.note(f"绕行{n + 1}")
        if self.area_detours == 1:
            self.dump_flight("撞墙")
        self.press('move_s', 0.45)
        self.press(side, dur)
        self.press('move_w', 0.8)
        self.route_detoured = True
        self.motion_reset()

    def nudge(self, n, drift=0.0):
        """已经走到物件跟前却没出 F：左右小幅挪动找交互点，不绕远。"""
        keyops.keyUp('move_w')
        key = ('move_a', 'move_d')[n % 2]
        CUS_LOGGER.debug(f"贴近目标但没有交互提示，向{key}挪")
        self.press(key, 0.3 + 0.2 * (n // 2))
        if n >= 3:
            self.press('move_s', 0.4)
        self.motion_reset()

    def read_prompt(self):
        """右侧 F 交互提示的文字，没有返回 ''。只认一行提示的标准位置（OCR 一行，很快）；
        提示图标出现但这一行没字（多个选项叠着）时再整块读。"""
        text = clean_text(self.ts.ocr_one_row(self.screen, [1206, 587, 1437, 635]), char=0)
        if self.cjk_count(text) >= 2:
            return text
        if self.check("interact", mask=MASK["interact_icon"], threshold=0.95):
            return self.prompt_text()
        return ''

    def save_sample(self, tag):
        """存一张全分辨率画面到 logs/samples，文件名带标签；只给离线脚本用（评测 / 改进物件识别）。"""
        if not self.nav.get('save_samples'):
            return
        try:
            os.makedirs(SAMPLE_DIR, exist_ok=True)
            if len(os.listdir(SAMPLE_DIR)) > 800:
                return
            name = time.strftime("%m%d_%H%M%S_") + re.sub(r'[^\w一-鿿]', '', tag)[:20] + ".jpg"
            cv.imencode('.jpg', np.array(self.screen), [cv.IMWRITE_JPEG_QUALITY, 92])[1].tofile(os.path.join(SAMPLE_DIR, name))
        except Exception as e:
            CUS_LOGGER.debug(f"样本保存失败：{e}")

    def f_exhausted(self, text):
        """这个提示文字在本区域已经触发够多次了（可以反复交互的东西），不要再按。"""
        return self.f_count[text or '?'] >= self.nav['f_repeat_max']

    def tap_f(self, text):
        """走路时每帧都按一下 F：走到可交互物跟前，就算提示文字没读出来、提示位置不标准，也能交互上。
        防卡死：同一提示触发够次数、或本区域累计触发够次数后不再按。门的提示不在这里按（由走门逻辑确认后按）。"""
        if any(w in text for w in PORTAL_WORDS) or self.f_exhausted(text):
            return
        if sum(self.f_count.values()) >= self.nav['f_area_max']:
            return
        now = time.time()
        if now - self.f_last[0] < 0.25:
            return
        keyops.keyDown('interact')
        time.sleep(0.02)
        keyops.keyUp('interact')
        self.bb_keys.append('F')
        self.f_last = (now, text or '?')

    def f_triggered(self, text):
        """记一次 F 交互成功（界面打开了 / 目标办完了），并存样本。"""
        key = text or '?'
        self.f_count[key] += 1
        CUS_LOGGER.info(f"F 交互成功「{key}」（本区域第 {self.f_count[key]} 次，累计 {sum(self.f_count.values())}）")
        self.save_sample(f"F_{key}")

    def confirm_status(self, start_key, want):
        """区域状态再看一帧确认，免得 OCR 偶尔读错一个数字就当成换了区域。"""
        time.sleep(0.25)
        self.get_screen()
        return self.area_status(start_key) == want

    def walk_to(self, name, locate, on_prompt, timeout=25, aim_tol=130, first_box=None,
                lost_limit=None, max_detour=4, near=False, sprint=False, on_frame=None):
        """正对目标直走的闭环，所有「走过去」都用它：
        locate()        目标在当前画面里的框，看不见返回 None（看不见就保持方向继续走）；
        on_prompt(text) 出现 F 交互提示时调用，返回 True 表示事情办完；
        on_frame()      每帧额外检查，返回非空字符串就停下并把它作为结果返回；
        near            目标是贴上去才能交互的物件：被挡时只小幅挪动，不绕远。
        返回 'done' / 'changed'（区域换了）/ 'gone'（进了战斗或界面）/ 'lost' / 'stuck' / 'timeout'。"""
        start_key = self.area_key()
        tm = time.time()
        self.note(f"走向{name}")
        moving, lost, detours = False, 0, 0
        near_bias, recovers = None, 0
        last_frame = None
        self.fg_paused = False
        self.motion_reset()
        box = first_box
        try:
            while time.time() - tm < timeout and not self._stop:
                if box is None:
                    self.get_screen()
                    if self.fg_paused:
                        self.fg_paused, moving = False, False
                        self.motion_reset()
                    status = self.area_status(start_key)
                    if status != 'same':
                        frame = self.screen
                        if self.confirm_status(start_key, status):
                            CUS_LOGGER.info(f"走向{name}途中：{'区域已切换' if status == 'changed' else '离开了位面画面'}")
                            if status == 'gone' and time.time() - self.f_last[0] < 3:
                                self.screen = last_frame if last_frame is not None else frame
                                self.f_triggered(self.f_last[1])  # 刚按过 F 画面就变了：是 F 打开了界面
                            return status
                    last_frame = self.screen
                    text = self.read_prompt()
                    if text and (any(w in text for w in PORTAL_WORDS) or not self.f_exhausted(text)) and on_prompt(text):
                        if not any(w in text for w in PORTAL_WORDS):
                            self.f_triggered(text)
                        return 'done'
                    self.tap_f(text)
                    if on_frame is not None:
                        res = on_frame()
                        if res:
                            return res
                    box = locate()
                if box is not None:
                    lost = 0
                    bias = (box[0] + box[2]) // 2 - 950
                    big = max(box[2] - box[0], box[3] - box[1]) >= self.nav['near_lost_size']
                    near_bias = bias if big and abs(bias) > 250 else None
                    self.mark(box, name)
                    if abs(bias) > aim_tol:
                        keyops.keyUp('move_w')
                        moving = False
                        self.turn(self.bias_to_degrees(bias))
                        self.motion_reset()
                        time.sleep(0.15)
                        box = None
                        continue
                else:
                    lost += 1
                    if near_bias is not None and lost >= 2 and recovers < 3:
                        # 目标刚才就在跟前、偏在一侧，现在看不到了：是从它旁边走过去了。停下，朝它消失的一侧回头找
                        keyops.keyUp('move_w')
                        moving = False
                        recovers += 1
                        CUS_LOGGER.info(f"走向{name}：近处从{'右' if near_bias > 0 else '左'}侧跟丢，停下回头找（第 {recovers} 次）")
                        self.turn(60 if near_bias > 0 else -60)
                        near_bias = None
                        self.motion_reset()
                        time.sleep(0.2)
                        continue
                    if lost_limit is not None and lost >= lost_limit:
                        return 'lost'
                box = None
                if not moving:
                    keyops.keyDown('move_w')
                    if sprint:
                        keyops.keyDown('sprint')
                    moving = True
                    self.motion_reset()
                    time.sleep(0.3)
                    continue
                state = self.motion_state()
                if state != 'ok':
                    drift = self.drift()
                    moving = False
                    detours += 1
                    if detours > max_detour:
                        return 'stuck'
                    if near:
                        self.nudge(detours - 1, drift)
                    else:
                        self.detour(detours - 1, drift)
            return 'timeout'
        finally:
            keyops.keyUp('move_w')
            if sprint:
                keyops.keyUp('sprint')

    # ---------- 视觉找物件：入区扫视 + 按方向走过去 ----------
    def load_target_feats(self, name):
        """给每张模板算好 SIFT 特征（放大 2 倍：远处的物件只有 100 像素左右，原尺寸特征点太少）。"""
        feats = []
        for tpl in self.target_tpls.get(name, []):
            if tpl is None:
                continue
            big = cv.resize(cv.cvtColor(tpl, cv.COLOR_BGR2GRAY), None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
            kp, des = self.sift.detectAndCompute(big, None)
            if des is not None and len(kp) >= 10:
                feats.append(des)
        self.target_feats[name] = feats

    def sift_find(self, name):
        """特征点识别：模板的 SIFT 特征和画面逐一配对，配上的点聚在一处就是物件。
        实测（财富区域出生点，扭蛋机约 90x180 像素）：35 个匹配点全在真实位置，别处 0 个；
        同一张图上颜色模板匹配只有 0.47，还不如误报位置高。返回 (框, 匹配点数)。"""
        feats = self.target_feats.get(name)
        if not feats:
            return None, 0
        y0, y1, x1 = 150, 780, 1640
        img = np.array(self.screen)[y0:y1, :x1].copy()
        img[:20, :470] = 0
        img[380 - y0:, 820:1100] = 0   # 角色身体：衣服花纹会凑出十来个乱配的点（实测最多 14 个）
        big = cv.resize(cv.cvtColor(img, cv.COLOR_BGR2GRAY), None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
        kp, des = self.sift.detectAndCompute(big, None)
        if des is None or len(kp) < 2:
            return None, 0
        matcher = cv.BFMatcher()
        best_n, best_pts = 0, None
        for dt in feats:
            pts = np.array([kp[m.trainIdx].pt for m, n in matcher.knnMatch(dt, des, k=2) if m.distance < 0.75 * n.distance])
            if len(pts) < 4:
                continue
            pts = pts / 2
            med = np.median(pts, axis=0)
            keep = pts[(np.abs(pts[:, 0] - med[0]) < 140) & (np.abs(pts[:, 1] - med[1]) < 160)]
            if len(keep) > best_n:
                best_n, best_pts = len(keep), keep
        if best_pts is None:
            return None, 0
        bx1, by1 = best_pts.min(axis=0)
        bx2, by2 = best_pts.max(axis=0)
        pad_x, pad_y = max(20, (bx2 - bx1) * 0.25), max(20, (by2 - by1) * 0.25)
        box = [int(max(0, bx1 - pad_x)), int(max(0, by1 - pad_y) + y0), int(min(x1, bx2 + pad_x)), int(by2 + pad_y + y0)]
        return box, best_n

    def template_find(self, name):
        """缩小 4 倍的多尺度颜色模板匹配（快，约 0.13 秒）：物件离得近、和模板视角接近时好用。返回 (框, 得分)。"""
        img = np.array(self.screen)[:840].copy()
        img[:170, :470] = 0      # 左上角区域名、碎片
        img[220:660, 1640:] = 0  # 右侧队伍栏
        small = cv.resize(img, None, fx=0.25, fy=0.25)
        best, box = 0, None
        for tpl in self.target_tpls.get(name, []):
            for sc in (0.5, 0.7, 1.0, 1.4, 2.0, 2.8):
                t = cv.resize(tpl, None, fx=0.25 * sc, fy=0.25 * sc)
                if t.shape[0] < 10 or t.shape[0] >= small.shape[0] or t.shape[1] >= small.shape[1]:
                    continue
                _, v, _, loc = cv.minMaxLoc(cv.matchTemplate(small, t, cv.TM_CCOEFF_NORMED))
                if v > best:
                    best, box = v, [loc[0] * 4, loc[1] * 4, (loc[0] + t.shape[1]) * 4, (loc[1] + t.shape[0]) * 4]
        return box, best

    def find_target(self, name, fast=False):
        """在当前画面里找物件（宝箱 / 扭蛋机），返回框或 None。
        先用模板匹配（快）；不够阈值再用特征点识别（准，约 0.6 秒）。fast=True 只做模板匹配（边走边看时用）。"""
        box, score = self.template_find(name)
        if score >= TARGET_THRESHOLD[name]:
            self.mark(box, f"{name}{score:.2f}")
            self.last_found = (name, self.screen, box, 99)
            return box
        if fast:
            return None
        sbox, n = self.sift_find(name)
        CUS_LOGGER.debug(f"识别{name}：模板 {score:.2f}，特征点 {n} 个 {sbox}")
        if sbox is not None and n >= self.nav['sift_min_matches']:
            self.mark(sbox, f"{name}点{n}")
            self.last_found = (name, self.screen, sbox, n)
            return sbox
        return None

    def harvest_template(self, name):
        """走到并交互成功 = 刚才认出的确实是它：把当时画面里的那一块存成新模板，下次同一个房间一眼就能认出。
        已经认得很准（特征点 ≥30 或模板匹配命中）就不重复收；每种物件有数量上限。"""
        found, self.last_found = self.last_found, None
        if not found or found[0] != name or found[3] >= 30:
            return
        pat = TARGET_TPL[name]
        if len(glob.glob(os.path.join(TPL_DIR, pat))) >= self.nav['tpl_max_per_target']:
            return
        _, frame, box, n = found
        frame = np.array(frame)
        w, h = box[2] - box[0], box[3] - box[1]
        x1, y1 = max(0, box[0] - w // 6), max(0, box[1] - h // 6)
        x2, y2 = min(1640, box[2] + w // 6), min(840, box[3] + h // 6)
        if x2 - x1 < 60 or y2 - y1 < 60:
            return
        path = os.path.join(TPL_DIR, pat.replace('*', time.strftime("auto_%m%d_%H%M%S")))
        cv.imencode('.png', frame[y1:y2, x1:x2])[1].tofile(path)
        self.target_tpls[name].append(frame[y1:y2, x1:x2].copy())
        self.load_target_feats(name)
        CUS_LOGGER.info(f"新模板已保存：{name}（当时特征点 {n} 个）{path}")

    def load_rooms(self):
        """旧版预设（房间类型 → 物件 → 朝向角度），只给没认出房间时的扫视用。"""
        return self.room_lib.legacy

    def remember_heading(self, room, name, heading):
        hs = self.room_lib.legacy.setdefault(room, {}).setdefault(name, [])
        if all(abs(h - heading) > 15 for h in hs):
            hs.append(int(heading))
            self.room_lib.save()

    # ---------- 房间指纹 + 航线 ----------
    def room_enter(self, kind):
        """进入区域（或暂离重进）时调用：位姿清零；确定在出生点时取入口指纹，到航点库里认房间。"""
        self.pose.reset()
        # 首领房间走的是差分宇宙原有流程，移动不入账，不用航线
        self.pose.valid = bool(self.at_spawn and self.nav['route_enable'] and kind and AREA_GROUP.get(kind) != '首领组')
        self.at_spawn = False
        self.room, self.room_fp = None, None
        self.entry_cards = []
        self.route_used = {}
        self.route_detoured = False
        if not self.pose.valid:
            CUS_LOGGER.debug("房间指纹：不确定是否在出生点，本区域不用航线也不学习")
            return
        time.sleep(0.4)
        self.room_fp = rooms.fingerprint(np.array(self.get_screen()))
        if AREA_GROUP.get(kind) == '交互组':
            self.entry_cards = self.cards_in_view()
            if not self.entry_cards:
                # 正前方的牌会整个被角色挡住：往右挪一步再看一眼
                self.press('move_d', 0.35)
                time.sleep(0.25)
                self.get_screen()
                self.entry_cards = self.cards_in_view()
            if self.entry_cards:
                desc = '、'.join(f"{c['ang']:+d}°{'（?）' if c['kind'] == '?' else ''}" for c in self.entry_cards)
                CUS_LOGGER.info(f"事件牌：入口看到 {len(self.entry_cards)} 张（{desc}）")
            else:
                CUS_LOGGER.info("事件牌：入口没看到，只按白字标签找事件")
        room, score = self.room_lib.match(kind, self.room_fp)
        if room is None:
            CUS_LOGGER.info(f"房间指纹：没见过的{kind}房间（最像的相似度 {score}），照旧扫视，走通后登记")
            return
        room['seen'] = room.get('seen', 0) + 1
        self.room = room
        routes = [n for n in room['targets'] if rooms.RoomLibrary.waypoints(room, n)]
        CUS_LOGGER.info(f"房间指纹：认出 {room['id']}（相似度 {score}，第 {room['seen']} 次），已有航线：{'、'.join(routes) or '无'}")

    def route_to(self, name, seen=None):
        """认出房间且记过去 name 的路线时：按折点转向、按里程推进，不扫视。
        到目标附近（差 route_terminal 里程）或 seen() 已经看见目标就停，交给调用方的视觉闭环做最后对准。
        途中被挡由 walk_to 绕行，绕完按新位姿重新对准航点。
        返回 'none' 没有航线 / 'seen' / 'arrived' / 'failed' / 'gone' / 'changed'。"""
        pts = rooms.RoomLibrary.waypoints(self.room, name) if self.pose.valid else None
        if not pts:
            return 'none'
        self.route_used[name] = True
        CUS_LOGGER.info(f"航线：{self.room['id']} → {name}，{len(pts)} 个航点，不扫视")
        frames = {'n': 0}
        res = 'arrived'
        for i, pt in enumerate(pts):
            margin = self.nav['route_terminal'] if i == len(pts) - 1 else 1.5

            def on_frame(pt=pt, margin=margin, last=i == len(pts) - 1):
                frames['n'] += 1
                turn, dist = self.pose.bearing_to(pt)
                if dist <= margin or abs(turn) > 75:
                    return 'arrived'   # 方位角超过 75° 说明已经走过了
                if self.route_detoured:
                    return 'steer'
                if seen is not None and last and frames['n'] % 3 == 0 and seen():
                    return 'seen'
                return ''

            for _ in range(5):  # 每次绕行后重新对准一次，最多 5 次
                turn, dist = self.pose.bearing_to(pt)
                if dist <= margin:
                    res = 'arrived'
                    break
                if i < len(pts) - 1 and abs(turn) > 60 and abs(self.pose.bearing_to(pts[i + 1])[0]) < abs(turn):
                    res = 'arrived'   # 中间折点已经落在侧后方（绕行绕过去了）：不折回去，直接去下一个
                    break
                if abs(turn) > 4:
                    self.turn(turn)
                    time.sleep(0.15)
                self.route_detoured = False
                res = self.walk_to(f"航点{i + 1}/{len(pts)}", lambda: None, lambda text: False,
                                   timeout=max(6.0, (dist - margin) / max(self.odo.rate, 0.5) * 2 + 4),
                                   max_detour=3, on_frame=on_frame)
                if res != 'steer':
                    break
                CUS_LOGGER.info(f"航线：绕行后重新对准航点 {i + 1}（偏 {self.pose.bearing_to(pt)[0]:+.0f}°）")
            if res in ('gone', 'changed', 'seen'):
                return res
            if res != 'arrived':
                CUS_LOGGER.info(f"航线：去{name}的航点 {i + 1} 没走到（{res}），改用扫视")
                rooms.RoomLibrary.fail(self.room, name)
                self.room_lib.save()
                return 'failed'
        return 'arrived'

    def route_learn(self, name):
        """目标交互成功：把从上一个目标（或出生点）走到这里的折线记进航点库。"""
        if not self.pose.valid:
            return
        if self.room is None:
            if self.room_fp is None:
                return
            self.room = self.room_lib.add(self.area_now or self.area_kind or '', self.room_fp)
            CUS_LOGGER.info(f"房间指纹：登记新房间 {self.room['id']}")
        path = self.pose.path_since_anchor()
        rooms.RoomLibrary.learn(self.room, name, path, time.time() - self.area_t0)
        t = self.room['targets'][name]
        # 位姿对到记下的目标位置上，消掉累计误差，下一段从这里起算
        self.pose.x, self.pose.y = t['pos']
        self.pose.mark_anchor()
        self.room_lib.save()
        CUS_LOGGER.info(f"航线：记下 {self.room['id']} → {name}（位置 {t['pos']}，折点 {len(t['via'])} 个，第 {t['ok']} 次走通）")

    def scan_for(self, name, room=None, full=False):
        """原地扫视找物件：先看房间预设里记过的方向，再扫前方 ±90°（full 时扫一整圈）。
        找到就停在正对它的朝向，返回相对扫视起点的角度；找不到就回到起点朝向，返回 None。"""
        heading = 0
        self.note(f"扫视找{name}")

        def look(target_heading):
            nonlocal heading
            self.turn(target_heading - heading)
            heading = target_heading
            time.sleep(0.15)
            self.get_screen()
            box = self.find_target(name)
            if box:
                aim = heading + self.bias_to_degrees((box[0] + box[2]) // 2 - 950)
                self.turn(aim - heading)
                heading = aim
                return True
            return False

        presets = self.load_rooms().get(room, {}).get(name, []) if room else []
        angles = list(presets) + [0, -30, -60, -90, 30, 60, 90]
        if full:
            angles += [120, 150, 180, -150, -120]
        for a in angles:
            if look(a):
                CUS_LOGGER.info(f"扫视：{name} 在 {heading:+.0f}°")
                return heading
        self.turn(-heading)
        CUS_LOGGER.info(f"扫视：没找到{name}（{'一整圈' if full else '前方180°'}）")
        return None

    def prompt_text(self):
        """读右侧 F 交互提示的文字。不用图标模板：宝箱的提示在图标位置是个宝箱图案，传送门模板对不上。"""
        self.ts.forward(self.screen)
        items = self.ts.find_with_box([1150, 540, 1650, 690], redundancy=0)
        return merge_text(items) if items else ''

    def approach(self, name, f_words, timeout=20):
        """已经正对物件：直走到出现 F 交互（文字含 f_words；f_words 为 None 时接受任何不是传送门的交互）并按下。
        正前方的物件会被角色身体挡住，看不见不算丢：按扫视时的框大小估算要走多久，走过了还没到才重新扫视。"""
        self.get_screen()
        box = self.find_target(name)
        width = (box[2] - box[0]) if box else 200
        budget = min(8.0, max(1.5, 4.0 * 200 / max(width, 40)))  # 估计步行秒数，远的多走

        def on_prompt(text):
            if f_words:
                ok = any(w in text for w in f_words)
            else:
                ok = not any(w in text for w in PORTAL_WORDS) and 2 <= self.cjk_count(text) == len(text) <= 8
            if ok:
                keyops.keyUp('move_w')
                self.press('interact')
                CUS_LOGGER.info(f"{name}：按 F「{text}」")
            return ok

        found_at_scan = self.last_found
        frames = {'n': 0}

        def locate():
            frames['n'] += 1
            return self.find_target(name, fast=frames['n'] % 4 != 0)

        res = ''
        for _ in range(3):
            res = self.walk_to(name, locate, on_prompt, timeout=timeout, aim_tol=140,
                               lost_limit=int(budget / 0.4) + 3, near=True, max_detour=6)
            if res == 'done':
                self.last_found = found_at_scan
                self.harvest_template(name)
                return True
            if res != 'lost':
                break
            CUS_LOGGER.info(f"{name}：走了估计距离还没到，重新扫视")
            found_at_scan = None
            if self.scan_for(name, full=True) is None:
                break
            budget = 3.0
        self.dump_flight(f"走不到{name}_{res}")
        return False

    def after_gacha(self):
        """抽奖后的界面还没摸清：先截图留档，按通用规则处理弹窗，直到回到位面。"""
        time.sleep(2.5)
        self.get_screen()
        cv.imwrite(os.path.join(UNKNOWN_DIR, time.strftime("gacha_%m%d_%H%M%S") + ".png"), np.array(self.screen))
        tm = time.time()
        while time.time() - tm < 20 and not self._stop:
            self.ts.forward(self.get_screen())
            text = self.in_run_text()
            if '位面' in text or '区域' in text:
                return
            if not self.run_static():
                self.click_point(960, 1000)
            time.sleep(1)

    def wealth_area(self):
        """财富组：扫视找宝箱 → 走过去按 F 开箱（+300 碎片，门随之激活）→ 找扭蛋机抽 3 次 → 走门。"""
        self.area_state = 1
        pre = self.route_to('宝箱', lambda: self.find_target('宝箱', fast=True))
        if pre in ('gone', 'changed'):
            return 1
        preset = None if pre in ('seen', 'arrived') else '财富'  # 按航线到了跟前：目标就在正前方，不看旧预设角度
        h = self.scan_for('宝箱', room=preset)
        if h is None:
            h = self.scan_for('宝箱', room=preset, full=True)
        if h is None:
            self.dump_flight("找不到宝箱")
        if h is not None:
            self.remember_heading('财富', '宝箱', h)
            if self.approach('宝箱', LOOT_WORDS):
                self.route_learn('宝箱')
                time.sleep(3)
                self.press('move_s', 1.0)  # 贴着宝箱时旁边的扭蛋机太近、看不全，退一步再找
        for i in range(self.nav['gacha_draws']):
            if self._stop or self.route_to('扭蛋机', lambda: self.find_target('扭蛋机', fast=True)) in ('gone', 'changed'):
                break
            if self.scan_for('扭蛋机', full=True) is None:
                break
            if not self.approach('扭蛋机', None):
                break
            self.route_learn('扭蛋机')
            CUS_LOGGER.info(f"财富区域：扭蛋机第 {i + 1} 次")
            self.after_gacha()
        self.portal_opening_days(static=1)
        return 1

    def in_battle(self):
        """区域名消失 = 进入战斗（或过场）。"""
        self.get_screen()
        text = self.in_run_text()
        return not ('位面' in text or '区域' in text)

    def enemy_ahead(self):
        """屏幕上方居中的怪物信息条：正前方近处有怪时才出现。"""
        return self.check("divergent/z", mask=MASK["monster_info_divergent"], threshold=0.96)

    def fight_area(self):
        """战斗组：朝敌人直走（冲刺），怪物信息条出现就普攻开战。精英头顶有橙色标记，看得到就对准它。
        被挡住按统一规则绕行；走不到就原地扫视一圈找怪，还找不到才去找门（可能已经打完）。"""
        fresh = self.entered_by_portal
        self.entered_by_portal = False
        if not fresh and self.labeled_portal_around():
            # 门只在战斗结束后出现：暂离重进 / 被绝区零暂停后回来时，这个区域可能早就打完了
            CUS_LOGGER.info("战斗区域：已经看到「区域-xx」门，战斗已结束，直接找门")
            self.area_state = 1
            return 1
        CUS_LOGGER.info("战斗区域：寻找敌人")
        self.area_state = 1
        self.press('role_1')
        time.sleep(0.5)
        t0 = time.time()
        swing = {'t': t0 + 4}

        def on_frame():
            now = time.time()
            if self.enemy_ahead():
                keyops.keyUp('sprint')
                pyautogui.click()
                time.sleep(0.7)
                self.motion_reset()
            elif now > swing['t']:
                # 走了一阵还没碰到：边走边挥刀，蹭到小怪就开战（挥刀会顿一下，所以隔几帧才挥一次）
                swing['t'] = now + 2.5
                pyautogui.click()
                time.sleep(0.5)
                self.motion_reset()
            return ''

        def elite_mark():
            return self.portal_blob(orange_only=True)

        res = self.walk_to('敌人', elite_mark, lambda text: False, timeout=self.nav['fight_seek_seconds'],
                           aim_tol=150, sprint=True, on_frame=on_frame, max_detour=3)
        if res not in ('gone', 'changed'):
            # 正前方没走到怪：原地扫视，哪个方向出现怪物信息条或橙色标记就朝那边再走一段
            heading = 0
            for a in SCAN_FRONT[1:] + SCAN_BACK:
                if self._stop:
                    break
                self.turn(a - heading)
                heading = a
                time.sleep(0.25)
                self.get_screen()
                if self.enemy_ahead() or elite_mark():
                    CUS_LOGGER.info(f"战斗区域：扫视在 {a:+d}° 看到敌人")
                    res = self.walk_to('敌人', elite_mark, lambda text: False, timeout=12, aim_tol=150,
                                       sprint=True, on_frame=on_frame, max_detour=2)
                    break
            else:
                self.turn(-heading)
        started = res == 'gone'
        self.fight_started = started
        CUS_LOGGER.info(f"战斗区域：{'已开战' if started else f'未找到敌人（{res}），转去找门'}，用时 {time.time() - t0:.0f}s")
        if not started and res != 'changed':
            self.dump_flight("找不到敌人")
        return 1

    def event_label(self):
        """事件物件头顶的白字标签在当前画面里的框（多个时取最靠近正前方的）；没有返回 None。
        x>1600 是右侧「状态效果」几个字，不算。"""
        pts = [p for p in self.get_text_position(clean=1) if 300 < p[0] < 1600]
        if not pts:
            return None
        x, y = min(pts, key=lambda p: abs(p[0] - 950))
        return [x - 60, y - 15, x + 60, y + 15]

    def cards_in_view(self):
        """当前画面里的事件牌：相对出生朝向的角度、画框高度、种类。"""
        return [{'ang': round(self.pose.heading) + self.bias_to_degrees((c['box'][0] + c['box'][2]) // 2 - 950),
                 'h': c['box'][3] - c['box'][1], 'kind': c['kind']}
                for c in cards.find_cards(np.array(self.screen))]

    def event_box(self):
        """事件物件在当前画面里的框：先认事件牌（画框，远处就能认，不会被招牌上的字带偏），认不出再找头顶的白字标签。
        在出生点看过一眼、一张牌都没有的房间不认牌，只用白字标签：路上的窗格偶尔会被错认成牌，把人带偏（08:27 实测）。"""
        if self.entry_cards or not self.pose.valid:
            found = cards.find_cards(np.array(self.screen))
            if found:
                return min(found, key=lambda c: abs((c['box'][0] + c['box'][2]) // 2 - 950))['box']
        return self.event_label()

    def next_card_bearing(self):
        """入口看到过不止一张事件牌、刚做完一张：推算下一张在哪个方向，返回要转的角度；推不出返回 None。
        只用入口时的角度和画框高度、加上现在的朝向，不用里程（08:51 实测有的地面光流弱，里程少算很多）：
        做完的是入口方向和现在朝向最接近的那张；人站在它跟前，另一张在侧面，
        横向隔着 tan(角度) 之差，纵深差按两张画框的高度比估。"""
        if len(self.entry_cards) < 2 or not self.pose.valid:
            return None
        order = sorted(self.entry_cards, key=lambda c: abs(c['ang'] - self.pose.heading))
        done, rest = order[0], order[1]
        side = math.tan(math.radians(rest['ang'])) - math.tan(math.radians(done['ang']))
        ahead = done['h'] / max(rest['h'], 1) - 1 + 0.12   # 以到第一张牌的距离为 1；0.12 是停下时离牌还有的那一小段
        want = math.degrees(math.atan2(side, ahead))
        return (want - self.pose.heading + 180) % 360 - 180

    def interact_area(self):
        """交互组（事件 / 奖励 / 遭遇）：往前走到看见事件标签 → 转过去正对 → 直走到出现 F → 按 F。
        不再左右平移去「对齐」。area_state 记已经交互过几次：一个区域最多两个事件，做完后找门。"""
        done = self.area_state
        self.area_state += 1
        aim = self.next_card_bearing() if done == 1 else None
        if aim is not None:
            # 入口就看到两张事件牌：两张都做完门才开，直接去下一张，不先去撞门
            CUS_LOGGER.info(f"交互区域：入口看到过 {len(self.entry_cards)} 张事件牌，转 {aim:+.0f}° 去下一张")
            self.turn(aim)
            for extra in (0, -35, 70):
                if extra:
                    self.turn(extra)
                time.sleep(0.3)
                self.get_screen()
                if cards.find_cards(np.array(self.screen)):
                    break
            else:
                self.turn(-35)
        elif done >= 1:
            # 多数房间只有一个事件：先找门。远处的门读不到字，就朝光晕走近再读；
            # 门没激活（还有事件没做）会走不进去，这时再去找第二个事件
            if self.portal_opening_days(static=1, soft=True) is not False:
                return 1
            CUS_LOGGER.info("交互区域：门还进不去，找第二个事件")

        def on_prompt(text):
            if any(w in text for w in EVENT_WORDS) and '区域' not in text:
                keyops.keyUp('move_w')
                self.press('interact')
                CUS_LOGGER.info(f"交互区域：按 F「{text}」")
                return True
            return False

        target = '事件' if done == 0 else '事件2'
        res = self.route_to(target, self.event_box)
        if res not in ('gone', 'changed'):
            res = self.walk_to('事件', self.event_box, on_prompt,
                               timeout=self.nav['event_seek_seconds'] if done == 0 else 15 if aim is not None else 8,
                               aim_tol=110, max_detour=3)
        if res not in ('done', 'gone', 'changed'):
            # 正前方没走到：原地扫视找标签，看到就朝它再走一段
            heading = 0
            for a in SCAN_FRONT[1:] + SCAN_BACK:
                if self._stop:
                    break
                self.turn(a - heading)
                heading = a
                time.sleep(0.25)
                self.get_screen()
                if self.event_box():
                    res = self.walk_to('事件', self.event_box, on_prompt, timeout=12, aim_tol=110, max_detour=2)
                    break
            else:
                self.turn(-heading)
        if res == 'done' or (res == 'gone' and time.time() - self.f_last[0] < 4):
            self.route_learn(target)
        if res == 'done':
            time.sleep(1.0)
            return 1
        if res in ('gone', 'changed'):
            return 1
        CUS_LOGGER.info(f"交互区域：第 {done + 1} 个事件没找到（{res}），转去找门")
        if done == 0:
            self.dump_flight("找不到事件")
        self.area_state = 2
        self.portal_opening_days(static=1)
        return 1

    def labeled_portal_around(self):
        """原地转一圈，看有没有带「区域-xx」字样的门（光晕兜底不算），转完回到原朝向。"""
        found = False
        heading = 0
        for a in SCAN_FRONT + SCAN_BACK:
            self.turn(a - heading)
            heading = a
            time.sleep(0.25)
            if self.find_portal().get('type'):
                found = True
                break
        self.turn(-heading)
        return found

    def plane4_survey(self):
        """第三位面首领打完、第一次看到「退出」门时：原地每 45° 存一张全图，然后停止运行，等人看图找第四位面的入口。
        只在 PLANE4_HOLD 文件存在时生效；返回 True 表示已经进入停下状态，调用方不要再走向退出门。"""
        if getattr(self, 'plane4_hold', False):
            return True
        area = getattr(self, 'area_text', '') or ''
        if not (os.path.exists(PLANE4_HOLD) and '第三位面' in area and '首领' in area):
            return False
        self.plane4_hold = True
        stamp = time.strftime("%m%d_%H%M%S")
        for k in range(8):
            cv.imencode('.png', np.array(self.get_screen()))[1].tofile(os.path.join(UNKNOWN_DIR, f"plane4_{stamp}_{k * 45:03d}.png"))
            self.turn(45)
            time.sleep(0.5)
        CUS_LOGGER.warning(f"第三位面首领房间看到退出门：已存 8 张 plane4_{stamp}_*.png，停止运行等待调查第四位面入口"
                           f"（删掉 unknowable_hold_plane4 再重启即可照常退出）")
        self._stop = True
        return True

    def in_plane3_boss(self):
        area = getattr(self, 'area_text', '') or ''
        return '第三位面' in area and '首领' in area

    def seek_fallacy(self):
        """第三位面首领房间看到退出门、没看到「区域-谬误」门（洋红色，通往第四位面，11-15 调查图里两门相隔约 80°）：
        原地每 45° 看一眼，每个区域只找一圈。看到返回 True（谬误门已在画面里），没看到返回 False（照常走退出门）。"""
        self.fallacy = 'scanned'
        for k in range(8):
            self.turn(45)
            time.sleep(0.5)
            if any('谬' in t['raw_text'] for t in self.ts.find_with_box([0, 0, 1920, 540], forward=1, mode=2)):
                self.fallacy = 'seen'
                CUS_LOGGER.info(f"第三位面首领房间：转 {(k + 1) * 45}° 看到谬误门（第四位面入口）")
                return True
        CUS_LOGGER.warning("第三位面首领房间：转了一圈没看到谬误门，照常走退出门")
        return False

    def find_portal(self, type=None):
        """差分宇宙的传送门识别加上不可知域的「精英」「交易」门。"""
        prefer_portal = {'奖励': 3, '事件': 3, '战斗': 2, '遭遇': 2, '商店': 1, '财富': 1,
                         '交易': 1, '精英': 2, '首领': 4, '休整': 4, '谬误': 10}   # 谬误门通往第四位面，压过退出门（9）
        tm = time.time()
        text = self.ts.find_with_box([0, 0, 1920, 540], forward=1, mode=2)
        portal = {'score': 0, 'nums': 0, 'type': ''}
        for i in text:
            raw = i['raw_text']
            if '退出' in raw and '模拟' in raw and i['box'][1] < 600:
                if self.plane4_survey():
                    return {'score': 0, 'nums': 0, 'type': ''}
                if self.in_plane3_boss():
                    if self.fallacy != 'skip' and any('谬' in t['raw_text'] for t in text):
                        self.fallacy = 'seen'
                    if self.fallacy == 'seen':
                        continue   # 本区域看到过谬误门：不理退出门，去第四位面
                    if self.fallacy is None and self.seek_fallacy():
                        return self.find_portal(type)
                # 最后一个首领打完后的「退出「模拟宇宙」」门
                i.update({'score': 9, 'type': '退出', 'nums': portal['nums'] + 1})
                if i['score'] > portal['score']:
                    portal = i
                continue
            if ('区' in raw or '域' in raw) and (i['box'][0] > 400 or i['box'][1] > 60):
                portal_type = self.get_text_type(raw, prefer_portal)
                if portal_type == '谬误':
                    if self.fallacy == 'skip':   # 额外轮次计数为 0，进不了第四位面
                        continue
                    self.fallacy = 'seen'
                if '冒' in raw or '险' in raw:
                    portal['nums'] += 1
                elif portal_type is not None:
                    i.update({'score': prefer_portal[portal_type] + 10 * (portal_type == type),
                              'type': portal_type, 'nums': portal['nums'] + 1})
                    if i['score'] > portal['score']:
                        portal = i
                    else:
                        portal['nums'] = i['nums']
        ocr_time = time.time() - tm
        self.ocr_time_list = self.ocr_time_list[-5:] + [ocr_time]
        if portal['score'] == 0:
            box = self.portal_blob()
            if box:
                # 远处的门看不到「区域-xx」字样：按门的光晕颜色先走过去，近了再读字。
                # nums 记 0：事件区域用 nums 判断「已经出现门＝事件做完」，光晕不能算。
                portal = {'score': 1, 'nums': 0, 'type': '', 'raw_text': '', 'box': box}
        CUS_LOGGER.debug(f"识别时间:{int(ocr_time * 1000)}ms, portal:{portal.get('type')} {portal.get('box')}")
        return portal

    def bias_to_degrees(self, bias):
        zero = bisect.bisect_left(config.angles, 0)
        return bisect.bisect_left(config.angles, bias) - zero

    def portal_opening_days(self, aimed=0, static=0, deep=0, need_label=False, soft=False):
        """找门并走进去：原地扫视（正前 → 左右 → 背后），看到「区域-xx」字样就转过去直走；
        一圈都没有字样才用门的光晕兜底。走不到就换个位置再扫，三轮都不行才暂离。"""
        if deep == 0:
            self.portal_cnt += 1
            pre = self.route_to('门', lambda: self.find_portal()['score'])
            if pre == 'changed':
                # 按航线走进门里直接被传送了
                self.entered_by_portal = self.at_spawn = True
                self.init_floor()
                return
            if pre == 'gone':
                return
        self.note("扫视找门")
        portal, glow, heading = None, None, 0
        for a in SCAN_FRONT + SCAN_BACK:
            if self._stop:
                return
            self.get_screen()
            if self.area_status('') == 'gone':
                # 界面被打开了（盲按 F 碰到了装置、弹窗等）：交给主循环处理，别对着界面转圈
                CUS_LOGGER.info("找门途中离开了位面画面，交给主循环")
                return
            self.turn(a - heading)
            heading = a
            time.sleep(0.25)
            found = self.find_portal()
            if found.get('type'):
                portal = found
                break
            if found['score'] and glow is None and (not need_label or self.portal_blob(blue_only=True)):
                glow = a
        if portal is None and need_label and glow is None:
            self.turn(-heading)
            return False
        if portal is None and glow is not None:
            self.turn(glow - heading)
            heading = glow
            time.sleep(0.25)
            found = self.find_portal()
            portal = found if found['score'] else None
        if portal is not None:
            res = self.walk_to_portal(portal)
            if res == 'entered':
                self.init_floor()
                return
            if res == 'interrupted':
                return
        if soft:
            return False
        if deep < 2:
            if portal is None:
                # 一圈都没看到门：回到进区朝向往前走一段（带撞墙检测），路上看到门就停
                self.turn(-heading)
                self.walk_to('前方找门', lambda: None, lambda text: False, timeout=2.5, max_detour=1,
                             on_frame=lambda: 'seen' if self.find_portal()['score'] else '')
            else:
                self.press('move_s', 1.0)
            return self.portal_opening_days(deep=deep + 1)
        self.close_and_exit()

    def walk_to_portal(self, portal, timeout=30):
        """朝门直走到出现「区域-xx」交互并按 F。返回 'entered' 进了新区域 / 'interrupted' 被战斗或界面打断 / 'failed'。
        每帧核对左上角区域编号：走进门里没按 F 就被传送、或按了 F 没反应，都按实际区域为准。"""
        kind = portal.get('type') or ''
        start_key = self.area_key()

        def locate():
            found = self.find_portal(kind or None)
            return found['box'] if found['score'] else None

        def on_prompt(text):
            if any(w in text for w in LOOT_WORDS):
                # 财富区域：门在宝箱正上方，开了宝箱（+300 碎片）门才激活
                keyops.keyUp('move_w')
                self.press('interact')
                CUS_LOGGER.info(f"财富区域：开启「{text}」")
                time.sleep(3)
                self.motion_reset()
                return False
            if not any(w in text for w in PORTAL_WORDS):
                return False
            keyops.keyUp('move_w')
            self.press('interact')
            time.sleep(0.3)
            self.press('interact')
            after = self.wait_area_change(start_key)
            if after == 'same':
                CUS_LOGGER.warning(f"按了 F「{text}」但区域没变，继续走")
                self.motion_reset()
                return False
            CUS_LOGGER.info(f"已进入传送门 {kind or text}")
            return True

        res = self.walk_to(f"门{kind}", locate, on_prompt, timeout=timeout, aim_tol=120,
                           first_box=portal.get('box'), lost_limit=10)
        if res == 'gone':
            res = {'changed': 'changed', 'same': 'failed', 'gone': 'interrupted'}[self.wait_area_change(start_key)]
        if res in ('done', 'changed'):
            self.route_learn('门')
            self.entered_by_portal = self.at_spawn = True
            return 'entered'
        return res if res == 'interrupted' else 'failed'

    def wait_area_change(self, start_key, timeout=8):
        """等左上角区域编号换掉：'changed' 换了 / 'same' 一直没变（F 没生效）/ 'gone' 到时间区域名还没回来（战斗、结算）。"""
        tm = time.time()
        time.sleep(0.8)
        same_since = None
        was_gone = False
        while time.time() - tm < timeout and not self._stop:
            self.get_screen()
            status = self.area_status(start_key)
            if status == 'changed' and self.confirm_status(start_key, 'changed'):
                return 'changed'
            if status == 'gone':
                was_gone, same_since = True, None
            elif status == 'same':
                if was_gone and not start_key:
                    return 'changed'  # 起点编号没读出来：黑屏过又回到位面画面，当作换了
                same_since = same_since or time.time()
                if time.time() - same_since > 3:
                    return 'same'
            time.sleep(0.3)
        return 'gone' if was_gone else 'same'

    def portal_blob(self, orange_only=False, blue_only=False):
        """在画面上半部找传送门的光晕（蓝色普通门 / 橙色精英、首领门），返回最大的框。"""
        img = np.array(self.get_screen())[:760]
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        best = None
        # 阈值按实拍样本调过：门的光晕饱和度、亮度都很高，地面/天空的蓝色达不到
        ranges = (((100, 190, 235), (122, 255, 255)), ((4, 190, 235), (22, 255, 255)))
        for lo, hi in ranges[1:] if orange_only else ranges[:1] if blue_only else ranges:
            mask = cv.inRange(hsv, lo, hi)
            mask[:, 1640:] = 0  # 右侧队伍栏
            mask[:170, :470] = 0  # 左上角区域名、碎片数
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv.contourArea(c)
                x, y, w, h = cv.boundingRect(c)
                if area < 2000 or h < 0.7 * w:
                    continue
                if best is None or area > best[0]:
                    best = (area, [x, y, x + w, y + h])
        return best[1] if best else None

    def close_and_exit(self, click=True):
        """卡住时暂离再继续（同一存档、回到区域起点），绝不结束并结算。"""
        self.dump_flight("寻路卡住")
        self.area_stuck += 1
        for name in self.route_used:
            rooms.RoomLibrary.fail(self.room, name)   # 按航线走还是卡住了：这条航线记一次失败，两次就作废
        if self.route_used:
            self.room_lib.save()
        self.respawn_pending = True
        area = self.area_text
        self.stuck_count = self.stuck_count + 1 if area and area == self.stuck_area else 1
        self.stuck_area = area
        give_up = self.stuck_count >= STUCK_LIMIT
        if give_up:
            # 暂离重进回到同一区域起点，地形不变，再试也是一样：结束本局换一张地图
            CUS_LOGGER.warning(f"同一区域连续卡住{self.stuck_count}次（{area}），结束并结算后重开本关")
            self.stuck_count = 0
            self.stuck_area = ''
        else:
            CUS_LOGGER.warning(f"寻路卡住（第{self.stuck_count}次），暂离后重新进入")
        self.init_floor()
        self.entered_by_portal = False
        self.press('esc')
        time.sleep(2.5)
        self.ts.forward(self.get_screen())
        wanted = '结束' if give_up else '暂离'
        for item in self.ts.find_with_box([1240, 860, 1800, 1020], redundancy=0):
            if wanted in item['raw_text']:
                self.click_box(item['box'])
                time.sleep(2.5)
                if give_up:
                    self.confirm_center()
                time.sleep(4)
                return
        # 不在菜单里就把菜单关掉，交给主循环
        self.press('esc')
        time.sleep(1)

    def confirm_center(self):
        """结束并结算后的二次确认弹窗：点中间区域里的「确认」。"""
        self.ts.forward(self.get_screen())
        for item in self.ts.find_with_box([500, 350, 1420, 800], redundancy=0):
            if '确认' in item['raw_text']:
                self.click_box(item['box'])
                CUS_LOGGER.info("已确认结束并结算")
                if os.path.exists(SETTLE_FILE):
                    os.remove(SETTLE_FILE)
                return True
        return False

    # ---------- 权杖选择 ----------
    def card_is_new(self, cx):
        """卡片右上角的书本图标（金色圆形）= 图鉴新条目。"""
        x, y = cx + 175, 265
        patch = np.array(self.screen)[y - 16:y + 16, x - 16:x + 16].astype(int)
        # 截图是 BGR。卡面背景是灰色（r≈b），图标是棕金色：数偏暖的像素
        warm = ((patch[..., 2] - patch[..., 0]) > 20).sum()
        return warm > 200

    def card_tags(self, cx):
        return merge_text(self.ts.find_with_box([cx - 200, 795, cx + 200, 832], redundancy=0))

    TRAITS = ['速度', '终结技', '击破', '追加攻击', '持续伤害', '充能']
    STATE_FILE = os.path.join("user", "core_settings", "unknowable_state.json")

    def load_state(self):
        try:
            with open(self.STATE_FILE, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {}

    def save_state(self, **values):
        state = self.load_state()
        state.update(values)
        with open(self.STATE_FILE, "w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False, indent=2)

    def remaining_traits(self):
        return self.load_state().get("traits_left", [t for t in self.TRAITS if t != '充能'])

    def pick_scepter(self):
        """权杖：本局锁定一个「秩序归演」还没完成的特性，所有权杖都选带这个特性的；其次选图鉴新条目。"""
        self.get_screen()
        self.ts.forward(self.screen)
        tendency = getattr(config, 'team', '终结技')
        cards = []
        for cx in self.card_centers():
            name = merge_text(self.ts.find_with_box([cx - 200, 450, cx + 200, 520], redundancy=0))
            if name:
                cards.append((cx, name, self.card_is_new(cx), self.card_tags(cx)))
        if not cards:
            self.select_and_confirm(CARD_XS[1])
            return
        remaining = self.remaining_traits()
        if self.run_trait is None and remaining:
            counts = {t: sum(t in c[3] for c in cards) for t in remaining}
            best_trait = max(remaining, key=lambda t: (counts[t], -remaining.index(t)))
            if counts[best_trait]:
                self.run_trait = best_trait
                self.save_state(run_trait=best_trait)
                CUS_LOGGER.info(f"本局锁定权杖特性：{best_trait}")
        best, best_score = cards[0][0], -99
        for cx, name, new, tags in cards:
            score = (20 if self.run_trait and self.run_trait in tags else 0) + (10 if new else 0)                 + (3 if tendency and tendency in tags else 0)
            CUS_LOGGER.info(f"权杖候选 {name} 新={new} 特性={tags} 分={score}")
            if score > best_score:
                best, best_score = cx, score
        self.select_and_confirm(best, [c[0] for c in cards])

    # 替换权杖界面：权杖数量满了又拿到新权杖。上面一行是新权杖，下面是已有的权杖（左侧数字是算力）。
    REPLACE_ROWS_Y = (534, 671, 808)

    def scepter_tags_on_panel(self):
        self.ts.forward(self.get_screen())
        return merge_text(self.ts.find_with_box([1270, 812, 1670, 858], redundancy=0))

    def replace_scepter(self):
        """新权杖带本局锁定特性：替换掉「不带特性、算力最低」的旧权杖；
        新权杖不带特性而旧权杖都带：放弃新权杖（+100 碎片），保住「全部权杖含有【X】」。"""
        time.sleep(0.8)
        trait = self.run_trait
        new_tags = self.scepter_tags_on_panel()
        rows = []
        for y in self.REPLACE_ROWS_Y:
            power = merge_text(self.ts.find_with_box([110, y - 40, 290, y + 40], redundancy=0))
            digits = ''.join(ch for ch in power if ch.isdigit())
            if not digits:
                continue
            self.click_point(352, y)
            time.sleep(0.8)
            tags = self.scepter_tags_on_panel()
            rows.append((y, int(digits), tags))
        new_has = bool(trait) and trait in new_tags
        CUS_LOGGER.info(f"替换权杖：新权杖特性={new_tags} 本局特性={trait} 旧权杖={[(p, t) for _, p, t in rows]}")
        if not rows:
            self.click_point(740, 978)
            return
        def keep_key(r):
            return (bool(trait) and trait in r[2], r[1])

        victim = min(rows, key=keep_key)
        victim_has = bool(trait) and trait in victim[2]
        if victim_has and not new_has:
            CUS_LOGGER.info("替换权杖：旧权杖都带本局特性、新权杖不带，放弃新权杖")
            self.click_point(740, 978)
        else:
            CUS_LOGGER.info(f"替换权杖：替换算力 {victim[1]} 的旧权杖（特性 {victim[2]}）")
            self.click_point(352, victim[0])
            time.sleep(0.8)
            self.click_point(1165, 978)
        time.sleep(2)

    def read_trait_progress(self):
        """在「重整模型」第一页读两类还没完成的任务：
        「全部权杖均含有【X】特性时通关」（秩序归演）和「难题等级≥1 时使用【X】演算倾向并通关第四位面」（倾向任务，各 80 星琼）。"""
        left, tend_left = [], []
        seen = seen_tend = 0
        for page in range(2):
            self.ts.forward(self.get_screen())
            rows = self.ts.find_with_box([460, 120, 1660, 1020], redundancy=0)
            status = self.ts.find_with_box([1660, 120, 1870, 1020], redundancy=0)
            for row in rows:
                raw = row['raw_text']
                if '全部权杖' in raw:
                    names, found = self.TRAITS, left
                    seen += 1
                elif '演算倾向' in raw and '难题' in raw:   # 关卡页也有「演算倾向」按钮，要带上难题条件才算任务行
                    names, found = self.TENDENCY_X, tend_left
                    seen_tend += 1
                else:
                    continue
                name = next((t for t in names if t in raw), None)
                y = (row['box'][1] + row['box'][3]) // 2
                st = ''.join(i['raw_text'] for i in status if abs((i['box'][1] + i['box'][3]) // 2 - y) < 60)
                if name and '进行' in st and name not in found:
                    found.append(name)
            if page == 0:
                key_mouse_manager.scroll_point(1150, 600, -10)
                key_mouse_manager.wait()
                time.sleep(1)
        if seen_tend:
            self.save_state(tendency_left=tend_left)
            CUS_LOGGER.info(f"第四位面倾向任务还差：{tend_left}")
        else:
            CUS_LOGGER.warning("第四位面倾向任务一行都没读到，沿用上次记录")
        if not seen:
            # 一行任务都没读到（界面被挡、没加载完）：保留上次的记录，不能当成「全部完成」
            CUS_LOGGER.warning("秩序归演任务一行都没读到，沿用上次记录")
            return
        self.save_state(traits_left=left)
        CUS_LOGGER.info(f"秩序归演还差的权杖特性：{left}")

    # 组件、奇物等「三选一」界面：先点新条目，没有就点中间，再确认
    def card_centers(self):
        """按卡片底部的特性标签找卡片位置：卡片数量不固定（2 张时在 725/1193，点正中间会落空）。"""
        items = self.ts.find_with_box([250, 790, 1700, 850], redundancy=0)
        xs = sorted((i['box'][0] + i['box'][2]) // 2 for i in items if i['raw_text'].strip())
        groups = []
        for x in xs:
            if groups and x - groups[-1][-1] < 220:
                groups[-1].append(x)
            else:
                groups.append([x])
        if not groups:
            # 奇物卡片底部没有特性标签：改用卡片正文的文字位置（只有一张卡时在正中间 960）
            for x in sorted((i['box'][0] + i['box'][2]) // 2
                            for i in self.ts.find_with_box([250, 260, 1700, 790], redundancy=0) if i['raw_text'].strip()):
                if groups and x - groups[-1][-1] < 220:
                    groups[-1].append(x)
                else:
                    groups.append([x])
        centers = [sum(g) // len(g) for g in groups]
        for layout in CARD_LAYOUTS:
            if len(layout) == len(centers) and all(abs(a - b) < 150 for a, b in zip(layout, centers)):
                return list(layout)
        return centers or list(CARD_XS)

    def pick_generic(self):
        """组件 / 奇物：图鉴新条目优先，其次带本局锁定特性的，否则第一张。"""
        self.get_screen()
        self.ts.forward(self.screen)
        centers = self.card_centers()
        target = None
        for cx in centers:
            if self.card_is_new(cx):
                target = cx
                break
        if target is None and self.run_trait:
            target = next((cx for cx in centers if self.run_trait in self.card_tags(cx)), None)
        if target is None:
            target = centers[0]
        CUS_LOGGER.debug(f"组件卡片位置 {centers}，选 {target}")
        self.select_and_confirm(target, centers)

    # 右下角确认按钮的药丸区域：高亮（可点）亮度约 200，低亮（不可点）约 45
    CONFIRM_BOX = (1580, 955, 1840, 990)

    def button_lit(self, box=None):
        """按钮亮不亮。界面通用常识：高亮 = 可点，低亮 = 不可点；低亮的按钮点多少次都没用，要先满足前置条件（比如先选中一张卡）。"""
        x1, y1, x2, y2 = box or self.CONFIRM_BOX
        val = float(cv.cvtColor(np.array(self.get_screen())[y1:y2, x1:x2], cv.COLOR_BGR2GRAY).mean())
        self.last_button_val = val
        return val > 110

    def select_and_confirm(self, target, centers=()):
        """点卡片 → 确认按钮变亮了才点确认；还是暗的说明没选中，换个位置再点卡片。返回有没有点确认。"""
        tried = []
        for cx in [target, *centers, 960, 725, 1193, 490, 1430]:
            if any(abs(cx - t) < 60 for t in tried):
                continue
            tried.append(cx)
            self.click_point(cx, 400)
            time.sleep(0.6)
            if self.button_lit():
                if len(tried) > 1:
                    CUS_LOGGER.info(f"选卡：点 {tried[:-1]} 后确认按钮还是暗的，改点 {cx} 才变亮")
                self.dim_fail = 0
                if not self.confirm_bottom_right():
                    self.click_point(1708, 972)
                time.sleep(1.5)
                return True
        self.dim_fail = getattr(self, 'dim_fail', 0) + 1
        CUS_LOGGER.warning(f"选卡：点遍 {tried} 确认按钮都是暗的（亮度 {self.last_button_val:.0f}，第 {self.dim_fail} 次）")
        if self.dim_fail == 2:
            cv.imwrite(os.path.join(UNKNOWN_DIR, time.strftime("dim_%m%d_%H%M%S") + ".png"),
                       np.array(self.screen))
        if self.dim_fail >= 2:
            # 可能是没见过的按钮样式：照旧点一次确认，避免因为误判卡死
            if not self.confirm_bottom_right():
                self.click_point(1708, 972)
            time.sleep(1.5)
        return False

    def confirm_bottom_right(self):
        for item in self.ts.find_with_box([1500, 920, 1900, 1020], forward=1, redundancy=0):
            if '确认' in item['raw_text'] or '确定' in item['raw_text']:
                self.click_box(item['box'])
                return True
        return False

    # ---------- 入口 ----------
    # 五关全通（思维边界 100%）后主界面只剩「混沌涟漪·00」一个节点
    STAGES = ['消逝的帝国', '未来的预演', '冰冷的技术', '知识的中心', '谁人的陵寝', '混沌涟漪']

    def pick_stage(self):
        """不可知域主界面：先把红点奖励领了，再点编号最大的关卡（最新解锁）。"""
        if self.need_claim:
            self.need_claim = False
            self.claim_rewards()
            return
        self.ts.forward(self.get_screen())
        best = None
        for item in self.ts.find_with_box([380, 100, 1800, 1000], redundancy=0):
            for idx, name in enumerate(self.STAGES):
                if name[:3] in item['raw_text'] and (best is None or idx > best[0]):
                    best = (idx, item['box'])
        if best is None and self.click_text_in(['分形档案'], [1400, 850, 1920, 1080]):
            # 主界面停在「庸众集」页（伟大谎言 / 记住他们 / 发条世界…，右下角「分形档案»」）：切回 01~05 的关卡列表
            CUS_LOGGER.info("主界面停在庸众集页：点「分形档案」切回关卡列表")
            time.sleep(2)
            return
        if best is None:
            self.stage_miss = getattr(self, 'stage_miss', 0) + 1
            if self.stage_miss in (1, 5):
                cv.imwrite(os.path.join(UNKNOWN_DIR, time.strftime("main_%m%d_%H%M%S") + ".png"), np.array(self.screen))
            if self.stage_miss >= 20:
                CUS_LOGGER.error("主界面连续 20 次没找到关卡名，停止运行，等待排查")
                self._stop = True
                return
            if self.stage_miss <= 5:
                # 刚领完奖、界面还在切换时常读不到字（11-15 20:07 第一次没读到就点进了 01，白打一局）：先等，不点
                CUS_LOGGER.warning(f"主界面没找到关卡名（第 {self.stage_miss} 次），等 2 秒再读")
                time.sleep(2)
                return
            CUS_LOGGER.warning(f"主界面没找到关卡名（第 {self.stage_miss} 次），点 01")
            self.click_point(490, 215)
        else:
            self.stage_miss = 0
            idx, box = best
            self.cur_stage = self.STAGES[idx]
            CUS_LOGGER.info(f"选择关卡 {idx + 1:02d} {self.STAGES[idx]}")
            # 关卡图标在标题左边约 190 像素、略低
            self.click_point(box[0] - 190, (box[1] + box[3]) // 2 + 15)
        time.sleep(2)

    def click_text_in(self, words, box):
        self.ts.forward(self.get_screen())
        for item in self.ts.find_with_box(box, redundancy=0):
            if any(w in item['raw_text'] for w in words):
                self.click_box(item['box'])
                return True
        return False

    def claim_list(self, box, rounds=8):
        """反复点「领取」，每次领完点空白处关掉获得物品弹窗。"""
        got = 0
        for _ in range(rounds):
            if not self.click_text_in(['领取'], box):
                break
            got += 1
            time.sleep(1.5)
            self.click_point(960, 1000)
            time.sleep(1.2)
        return got

    def claim_rewards(self):
        CUS_LOGGER.info("领取不可知域奖励")
        # 开拓秘闻 → 奖励进度
        self.click_point(157, 360)
        time.sleep(2.5)
        self.click_point(210, 988)
        time.sleep(2)
        n = self.claim_list([1350, 300, 1580, 870])
        self.click_point(1532, 252)
        time.sleep(1)
        self.press('esc')
        time.sleep(2)
        # 重整模型三个分页，列表较长时往下滚一次再领
        self.click_point(120, 950)
        time.sleep(2.5)
        for tab_y in (170, 270, 375):
            self.click_point(225, tab_y)
            time.sleep(1.2)
            n += self.claim_list([1660, 120, 1870, 1020])
            self.log_page(f"重整模型分页 y={tab_y}", [460, 120, 1870, 1020])
            if tab_y == 270:
                self.read_remap_progress()
            key_mouse_manager.scroll_point(1150, 600, -10)
            key_mouse_manager.wait()
            time.sleep(1)
            n += self.claim_list([1660, 120, 1870, 1020])
            if tab_y == 170:
                key_mouse_manager.scroll_point(1150, 600, 10)
                key_mouse_manager.wait()
                time.sleep(1)
                self.read_trait_progress()
        self.press('esc')
        time.sleep(2)
        # 仿真记录：收集奖励
        self.click_point(157, 265)
        time.sleep(2.5)
        n += self.claim_list([0, 0, 1920, 1080], rounds=4)
        self.log_page("仿真记录", [0, 0, 1920, 1080])
        self.press('esc')
        time.sleep(2)
        # 思维边界：猜想够就一直激活
        self.click_point(157, 175)
        time.sleep(2.5)
        for _ in range(6):
            self.click_point(1575, 982)
            time.sleep(2)
        self.press('esc')
        time.sleep(2)
        CUS_LOGGER.info(f"奖励领取完成，领了 {n} 项")

    def read_remap_progress(self):
        """「重整模型-复维映射」第一屏：「难题等级大于等于 3 时通关第四位面」（50 星琼）是否还在进行中，存 plane4_n3_left。
        还没完成时开局打 3 级（3 级没有「额外轮次降低」），完成后回到 1 级。读不到这一行就沿用上次记录。"""
        self.ts.forward(self.get_screen())
        status = self.ts.find_with_box([1660, 120, 1870, 1020], redundancy=0)
        for row in self.ts.find_with_box([460, 120, 1660, 1020], redundancy=0):
            raw = row['raw_text']
            if '第四位面' in raw and '演算倾向' not in raw:
                y = (row['box'][1] + row['box'][3]) // 2
                st = ''.join(i['raw_text'] for i in status if abs((i['box'][1] + i['box'][3]) // 2 - y) < 60)
                self.save_state(plane4_n3_left='进行' in st)
                CUS_LOGGER.info(f"难题≥3 通关第四位面：{'还没完成' if '进行' in st else '已完成'}")
                return
        CUS_LOGGER.warning("复维映射页没读到「通关第四位面」那一行，沿用上次记录")

    def log_page(self, name, box):
        """领完奖后记一行页面文字（进行中 / 可领取的数量 + 前 160 字），用来核对有没有漏领、还剩哪些任务。"""
        self.ts.forward(self.get_screen())
        text = merge_text(self.ts.find_with_box(box, redundancy=0))
        CUS_LOGGER.info(f"{name}：进行中 {text.count('进行中')}，领取 {text.count('领取')}，文字：{text[:160]}")

    TEAM_SLOTS = (1148, 1240, 1332, 1424)

    def team_templates(self):
        team_dir = os.path.join("user", "team")
        out = []
        if os.path.isdir(team_dir):
            for name in sorted(os.listdir(team_dir)):
                if name.lower().endswith(".png"):
                    img = cv.imdecode(np.fromfile(os.path.join(team_dir, name), np.uint8), cv.IMREAD_COLOR)
                    if img is not None:
                        out.append((name, img))
        return out

    def empty_slots(self):
        """数关卡页底部空着的队伍位：空位上是白色图钉图标（折磨难度底色偏红，不能按颜色判断）。"""
        gray = cv.cvtColor(np.array(self.get_screen()), cv.COLOR_BGR2GRAY)
        if self.empty_slot_tpl is None:
            self.empty_slot_tpl = cv.cvtColor(cv.imread(EMPTY_SLOT_TPL), cv.COLOR_BGR2GRAY)
        count = 0
        for x in self.TEAM_SLOTS:
            res = cv.matchTemplate(gray[930:1005, x - 40:x + 40], self.empty_slot_tpl, cv.TM_CCOEFF_NORMED)
            count += res.max() > 0.8
        return int(count)

    def start_stage(self):
        """关卡页：队伍为空就按 user/team 头像模板配队，然后启动。"""
        if os.path.exists(HOLD_FILE):
            CUS_LOGGER.warning("关卡页暂停：等待调查演算倾向和配队（删掉 unknowable_hold_start 再重启即可继续）")
            cv.imwrite(os.path.join(UNKNOWN_DIR, "stage_page.png"), np.array(self.get_screen()))
            self._stop = True
            return
        self.run_trait = None
        self.at_spawn = True   # 新开一局：第一个区域一定从出生点开始
        self.save_state(run_trait=None)
        self.ts.forward(self.get_screen())
        panel = merge_text(self.ts.find_with_box([60, 380, 1000, 760], redundancy=0))
        CUS_LOGGER.info(f"关卡页左侧面板文字：{panel[:160]}")
        self.save_sample("关卡页")
        # 难题 3 级会让首领少 1 回合：11-15 21:56 第四位面首领剩 17% 血时回合用完，判负不算通关第四位面。
        # 「≥3 级通关第四位面」（50）最多试 2 次进第四位面，之后一律打 1 级，先拿 4 个倾向任务（各 80，只要求 ≥1 级）。
        st = self.load_state()
        target = 3 if st.get('plane4_n3_left', True) and st.get('n3_tries', 0) < 2 else self.NANDU_TARGET
        self.save_state(run_nandu=target)
        self.set_nandu(target)
        self.set_tendency()
        self.ensure_team()
        self.click_point(1680, 968)
        time.sleep(6)

    # 演算倾向选择框：4 个选项的中心 x（y=567），确认 (963,980)
    TENDENCY_X = {'终结技': 478, '持续伤害': 802, '追加攻击': 1126, '击破': 1450}

    # 难题等级（只有 05 谁人的陵寝有）：减号 / 加号的位置。1~3 级首通奖励已领（11-15 核对关卡页样本），
    # 4 级以上没有星琼、敌人更硬，5 级还有「额外轮次降低 1」（额外轮次计数为 0 就进不了第四位面，11-15 19:22 那局就是这样）；
    # 第四位面倾向任务只要求难题 ≥1，所以固定打 1 级
    NANDU_MINUS = (248, 547)
    NANDU_PLUS = (823, 547)
    NANDU_TARGET = 1

    def read_nandu(self):
        """读面板右上角的「难题等级 N」大字（OCR 读成「难题等级5」，样本 10 张全对）；没有面板或读不到返回 None。
        不能按「+N」说明条判断：数字会和百分比粘在一起，而且 5 级和 4 级的说明都有「攻击力提高」。"""
        self.ts.forward(self.get_screen())
        for item in self.ts.find_with_box([400, 100, 1000, 330], redundancy=0):
            m = re.search(r'难题等级\D{0,2}(\d)', item['raw_text'])
            if m:
                return int(m.group(1))
        return None

    def set_nandu(self, target):
        """按读到的等级点减号 / 加号，直到等于 target；读不到时先减到底再加 target 次（最低档是 0 还是 1 都能保证 ≥ target）。"""
        level = self.read_nandu()
        if level is None and self.cur_stage != '谁人的陵寝':
            return   # 没有难题面板的关卡（01~04、混沌涟漪）
        for _ in range(8):
            if level is None or level == target:
                break
            self.click_point(*(self.NANDU_MINUS if level > target else self.NANDU_PLUS))
            time.sleep(0.8)
            new = self.read_nandu()
            if new == level and level < target:   # 加号到头（下一级没解锁）
                break
            level = new   # 减号没生效就继续点（最多 8 次），读数不变也不提前放弃
        if level is None:
            for _ in range(6):
                self.click_point(*self.NANDU_MINUS)
                time.sleep(0.4)
            for _ in range(target):
                self.click_point(*self.NANDU_PLUS)
                time.sleep(0.4)
            level = self.read_nandu()
        CUS_LOGGER.info(f"难题等级：目标 {target}，当前读数 {level}")

    def set_tendency(self):
        """演算倾向优先级：秩序归演和第四位面倾向任务都还差的特性 → 只有秩序归演差的 → 只有倾向任务差的 → 四种轮换（换着刷仿真记录）。
        选中的倾向同时锁定为本局权杖特性。"""
        traits = [t for t in self.remaining_traits() if t in self.TENDENCY_X]
        tends = [t for t in self.load_state().get('tendency_left', list(self.TENDENCY_X)) if t in self.TENDENCY_X]
        order = list(self.TENDENCY_X)
        last = self.load_state().get('last_tendency')
        rotate = order[(order.index(last) + 1) % len(order)] if last in order else order[0]
        want = next(iter([t for t in traits if t in tends] or traits or tends), rotate)
        self.save_state(last_tendency=want)
        self.click_point(1300, 845)
        time.sleep(1.5)
        self.click_point(self.TENDENCY_X[want], 567)
        time.sleep(0.6)
        self.click_point(963, 980)
        time.sleep(1.5)
        self.ts.forward(self.get_screen())
        now = merge_text(self.ts.find_with_box([1160, 820, 1480, 870], redundancy=0))
        CUS_LOGGER.info(f"演算倾向：想要 {want}，当前显示 {now}")
        if want in now:
            self.run_trait = want
            self.save_state(run_trait=want)

    # 配队面板「角色列表」的卡片中心（1920x1080），一页 4 列 x 5 行
    GRID_X = (96, 230, 364, 499)
    GRID_Y = (195, 353, 511, 669, 827)

    @staticmethod
    def card_selected(scr, cx, cy):
        """已上阵的卡片有白色描边：上沿和左沿亮度约 208，没选中的不到 90。"""
        g = cv.cvtColor(scr, cv.COLOR_BGR2GRAY)
        return g[cy - 68:cy - 62, cx - 50:cx + 50].mean() > 150 and g[cy - 55:cy + 60, cx - 60:cx - 56].mean() > 150

    @staticmethod
    def card_match(scr, cx, cy, tpls):
        roi = scr[max(cy - 70, 0):cy + 60, max(cx - 70, 0):cx + 70]
        best = 0
        for _, tpl in tpls:
            if roi.shape[0] >= tpl.shape[0] and roi.shape[1] >= tpl.shape[1]:
                best = max(best, cv.minMaxLoc(cv.matchTemplate(roi, tpl, cv.TM_CCOEFF_NORMED))[1])
        return best >= 0.85

    def list_page(self):
        return cv.resize(cv.cvtColor(np.array(self.get_screen())[110:950, 30:570], cv.COLOR_BGR2GRAY), (68, 105))

    def scroll_roster(self, direct):
        """滚动角色列表；返回画面是否变了（到底就不变）。滚轮无效时改用拖动。"""
        before = self.list_page().astype(int)
        key_mouse_manager.scroll_point(300, 560, direct)
        key_mouse_manager.wait()
        time.sleep(0.9)
        if np.abs(self.list_page().astype(int) - before).mean() > 3:
            return True
        y1, y2 = (820, 300) if direct < 0 else (300, 820)
        key_mouse_manager.drag_point(300, y1, 300, y2, 0.6)
        key_mouse_manager.wait()
        time.sleep(0.9)
        return np.abs(self.list_page().astype(int) - before).mean() > 3

    def ensure_team(self):
        """固定队伍（user/team 的头像模板，现为 银狼LV.999/爻光/阿哈/真珠）：打开配队面板，
        取消不在名单里的已上阵角色，再逐页找名单里的角色点上。"""
        tpls = self.team_templates()
        if not tpls:
            return
        self.click_point(self.TEAM_SLOTS[0], 968)
        time.sleep(2)
        self.click_point(167, 88)  # 角色列表标签
        time.sleep(1)
        for _ in range(6):  # 先回到列表顶部
            if not self.scroll_roster(10):
                break
        found = set()
        for page in range(8):
            scr = np.array(self.get_screen())
            for cy in self.GRID_Y:
                for cx in self.GRID_X:
                    if self.card_selected(scr, cx, cy) and not self.card_match(scr, cx, cy, tpls):
                        CUS_LOGGER.info(f"配队：取消不在名单里的角色 ({cx},{cy})")
                        self.click_point(cx, cy)
                        time.sleep(0.6)
            scr = np.array(self.get_screen())
            region = scr[100:960, 20:580]
            for name, tpl in tpls:
                if name in found:
                    continue
                _, v, _, loc = cv.minMaxLoc(cv.matchTemplate(region, tpl, cv.TM_CCOEFF_NORMED))
                if v < 0.85:
                    continue
                cx, cy = 20 + loc[0] + tpl.shape[1] // 2, 100 + loc[1] + tpl.shape[0] // 2
                gx = min(self.GRID_X, key=lambda x: abs(x - cx))
                gy = min(self.GRID_Y, key=lambda y: abs(y - cy - 5))
                if not self.card_selected(np.array(self.get_screen()), gx, gy):
                    self.click_point(gx, gy)
                    time.sleep(0.6)
                found.add(name)
                CUS_LOGGER.info(f"配队：{name} 已上阵（{v:.2f}）")
            if len(found) == len(tpls):
                break
            if not self.scroll_roster(-5):
                break
        missing = [n for n, _ in tpls if n not in found]
        if missing:
            CUS_LOGGER.warning(f"配队：没找到 {missing}")
        self.dump_flight("配队") if missing else None

    # 造物调试台（休整区台子上的装置，挡在首领门前）。主界面：组件覆写 (1457,348) / 组件合成 (1457,576) / 权杖改装 (1457,797)
    BENCH_GOAL = 20  # 覆写 / 合成 / 改装各累计 20 次有奖励

    def page_title(self):
        self.ts.forward(self.get_screen())
        return merge_text(self.ts.find_with_box([60, 45, 320, 110], redundancy=0))

    def fragments(self):
        """调试台各页右上角的宇宙碎片数；读不出来返回 -1。"""
        digits = ''.join(ch for ch in self.ts.ocr_one_row(self.get_screen(), [1690, 46, 1800, 86]) if ch.isdigit())
        return int(digits) if digits else -1

    def bench(self):
        """先做便宜的组件覆写（凑满 20 次），再用剩下的碎片做权杖改装（提升算能）。做完关掉，本区域不再盲按 F。"""
        self.f_count['?'] = self.f_count['造物调试台'] = 99
        if os.path.exists(BENCH_HOLD):
            CUS_LOGGER.warning("造物调试台已打开：按 unknowable_hold_bench 停下等待调查（删掉该文件后重启继续）")
            self._stop = True
            return
        state = self.load_state()
        try:
            if state.get('bench_overwrite', 0) < self.BENCH_GOAL:
                self.bench_overwrite(state.get('bench_overwrite', 0))
            self.bench_refit(self.load_state().get('bench_refit', 0))
            if not state.get('bench_synth_seen'):
                self.bench_peek_synth()
        except ValueError:
            raise
        except Exception as e:
            CUS_LOGGER.error(f"造物调试台操作出错：{e}")
        # 一路 Esc 回到位面画面
        for _ in range(5):
            self.get_screen()
            text = self.in_run_text()
            if '位面' in text or '区域' in text:
                break
            self.press('esc')
            time.sleep(1.5)
        # 装置挡在门的正前方：先退一步向右绕开，再找门
        self.press('move_s', 0.5)
        self.press('move_d', 1.0)

    def bench_overwrite(self, done):
        """组件覆写：把当前选中的组件换成同星级随机组件，8 碎片一次，点一下就生效，没有弹窗。"""
        self.click_point(1457, 348)
        time.sleep(2)
        for _ in range(3):  # 第一次进来有两页教程
            if '覆写' in self.page_title():
                break
            if not self.run_static(action_list=['教程翻页', '教程关闭']):
                self.click_point(1840, 508)
            time.sleep(1.2)
        if '覆写' not in self.page_title():
            CUS_LOGGER.warning("造物调试台：没进到组件覆写页")
            return
        before = self.fragments()
        times = min(self.BENCH_GOAL - done, before // 8 if before >= 0 else 0)
        for _ in range(times):
            self.click_point(1327, 984)
            time.sleep(0.9)
        after = self.fragments()
        made = (before - after) // 8 if before >= 0 and after >= 0 else 0
        self.save_state(bench_overwrite=done + max(made, 0))
        CUS_LOGGER.info(f"造物调试台：组件覆写 {made} 次（碎片 {before} → {after}），累计 {done + max(made, 0)}")
        self.press('esc')
        time.sleep(1.5)

    def bench_refit(self, done):
        """权杖改装：卡片在 x=151 / 372（y≈330），卡片下方是所需碎片；改装预览 (1661,984) → 改装 (1694,1006) → 点空白关闭。"""
        self.click_point(1457, 797)
        time.sleep(2)
        if '改装' not in self.page_title():
            CUS_LOGGER.warning("造物调试台：没进到权杖改装页")
            return
        made = 0
        for _ in range(4):
            have = self.fragments()
            target = None
            for cx in (151, 372, 593):
                digits = ''.join(ch for ch in self.ts.ocr_one_row(self.get_screen(), [cx - 85, 412, cx + 85, 452]) if ch.isdigit())
                if digits and 0 < int(digits) <= have:
                    target = (cx, int(digits))
                    break
            if target is None:
                break
            self.click_point(target[0], 330)
            time.sleep(0.8)
            for _ in range(2):  # 「改装预览」有时要点两下才打开
                self.click_point(1661, 984)
                time.sleep(1.2)
                self.ts.forward(self.get_screen())
                if '预览' in merge_text(self.ts.find_with_box([760, 50, 1160, 140], redundancy=0)):
                    break
            else:
                break
            self.click_point(1694, 1006)
            time.sleep(2.5)
            self.click_point(960, 1015)
            time.sleep(2)
            left = self.fragments()
            if left < 0 or left >= have:
                # 没扣碎片：没改成，退回改装页
                for _ in range(2):
                    if '改装' in self.page_title():
                        break
                    self.press('esc')
                    time.sleep(1.2)
                break
            made += 1
            CUS_LOGGER.info(f"造物调试台：权杖改装 1 次，花费 {target[1]}（碎片 {have} → {left}）")
        if made:
            self.save_state(bench_refit=done + made)
        CUS_LOGGER.info(f"造物调试台：本次权杖改装 {made} 次，累计 {done + made}")
        for _ in range(3):
            title = self.page_title()
            if '调试台' in title:
                break
            self.press('esc')
            time.sleep(1.5)

    def bench_peek_synth(self):
        """组件合成（4 个同星级组件合成更高星级）界面还没摸清：进去拍一张存档就退出。"""
        if '调试台' not in self.page_title():
            return
        self.click_point(1457, 576)
        time.sleep(2)
        for _ in range(3):
            if '合成' in self.page_title():
                break
            if not self.run_static(action_list=['教程翻页', '教程关闭']):
                self.click_point(960, 888)
            time.sleep(1.2)
        cv.imwrite(os.path.join(UNKNOWN_DIR, "bench_synth.png"), np.array(self.get_screen()))
        self.save_state(bench_synth_seen=1)
        CUS_LOGGER.info("造物调试台：组件合成界面已存档 bench_synth.png")
        self.press('esc')
        time.sleep(1.5)

    def expansion_card(self):
        """扩展装置：中间卡片是不可知域就进入，否则往下翻一张。"""
        text = merge_text(self.ts.find_with_box([800, 530, 1100, 590], redundancy=0))
        if '不可知' in text:
            self.click_point(1270, 570)
        else:
            self.click_point(1170, 920)
        time.sleep(2)

    def run_finished(self):
        self.runs_done += 1
        self.need_claim = True
        CUS_LOGGER.info(f"不可知域完成一局，本次共 {self.runs_done} 局")
        self.end_of_uni()


def create_engine(*, script=False):
    config.read()
    return UnknowableDomain(int(config.debug_mode), int(config.max_run), int(config.speed_mode))
