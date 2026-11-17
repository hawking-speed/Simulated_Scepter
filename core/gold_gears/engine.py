"""模拟宇宙「黄金与机械」（游戏内部名 RogueNous）。

和寰宇蝗灾共用棋盘系统：三个位面、从左往右走格子、觐见装置投骰、倒计时、第二位面选首领。
所以直接继承 AnyFateUniverse，只改开局前后的部分：

- 入口：扩展装置里选「黄金与机械」卡片 → 主界面 → 自定义拼装（选骰子）→ 开始探索。
- 跑图计划：config 的 runs 按顺序列出每局用的骰子、难度、命途、各位面首领战的认知值目标；
  通关（打完第三位面结算）才算这一项完成，状态记在 user/core_settings/gold_gears_state.json。
- 位面名和蝗灾不同：认不到位面名时按顺序推断，并把识别到的文字写进日志，方便补关键词。
- 不可知域有暂离存档时黄金与机械开不了局：此时绝不点「快捷结束并结算」，直接停内核。
- 认知值（自我认知）还没做：目前只在日志里记目标区间，供第二阶段使用。
"""
import json
import os
import re
import time
from pathlib import Path

import cv2 as cv
import numpy as np

from core.any_fate.engine import AnyFateUniverse
from core.any_fate.text_key import FATES
from route import PATHS
from tool.GLOBAL import key_mouse_manager
from tool.log import CUS_LOGGER
from tool.mask import MASK
from tool.public_ocr import load_actions, merge_text
from tool.storage import config_path, load_module_settings, resource_path
from tool.utils.image_tool import find_image_by_name

# 被不可知域存档挡住时写这个标记文件，调度方（另一个会话 / 计划任务）看到就知道要先处理不可知域
_GG_TOOLS_DIR = Path(
    os.environ.get("SIMS_GG_TOOLS_DIR") or (Path(__file__).resolve().parents[3] / "tools")
)
BLOCKED_FLAG = str(_GG_TOOLS_DIR / "GG_BLOCKED.txt")
DONE_FLAG = str(_GG_TOOLS_DIR / "GG_DONE.txt")

# 骰面装配界面：顶部四个类别图标（信标 / 区域 / 知识 / 异质），类别下最多 3 张骰子卡片竖排
DICE_CATEGORY_TABS = {"信标": (257, 154), "区域": (384, 154), "知识": (506, 154), "异质": (631, 154)}
DICE_CARD_BOX = [470, 280, 820, 860]      # 卡片上骰子名和「未解锁 / 暂无记录 / 最高通关记录」所在区域
DICE_TITLE_BOX = [780, 135, 1300, 175]    # 右侧详情标题 = 当前选中的骰子
DICE_CLOSE = (1862, 66)

# 黄金与机械主界面
GG_CUSTOM_DICE = (602, 512)   # 自定义拼装
GG_START = (958, 812)         # 开始探索


def in_box(b, box, pad=10):
    """文字框 b 是否落在区域 box 里（允许 pad 像素误差）。"""
    return b[0] >= box[0] - pad and b[1] >= box[1] - pad and b[2] <= box[2] + pad and b[3] <= box[3] + pad


class GoldGearsUniverse(AnyFateUniverse):
    """黄金与机械：按跑图计划逐局换骰子，打通全部骰子测试 / 秘闻 / 难度首通。"""

    count_line = 3   # count.txt 第 4 行；0 铁血、1 弹指、2 任意命途
    cognition_val = None
    cognition_rec = None
    _ban_attempts = 0
    _node_rerolls = 0
    _bless_shop_tries = 0
    _bless_shop_bought = 0
    _actual_conundrum = None

    def __init__(self):
        super().__init__()
        self.gg = load_module_settings(__file__)
        self.opt.update(self.gg)
        # 父类按 any_fate.json 装的是 insect.json，换成本模块的 nous.json
        self.default_json_path = str(resource_path(__file__, "actions/nous.json"))
        self.default_json = load_actions(self.default_json_path)
        # 寰宇蝗灾的命途轮换不用在这里
        self.fate_rotation = []
        # 黄金与机械高价值非认知事件关键词前置，覆盖通用列表
        gg_event_prior = ["作弊", "获得2个随机奇物", "获得1个祝福", "获得1个奇物", "赞成"]
        self.event_prior = gg_event_prior + [k for k in self.event_prior if k not in gg_event_prior]
        self.dice_ready = False           # 本局骰子是否已在骰面装配里选好
        self.locked_dice = set()          # 本次运行中发现还没解锁的骰子
        self.dice_attempts = 0            # 骰面装配里连续没选中的次数
        self.plane_seq_time = 0           # 最近一次「按顺序推断位面」的时间
        self.logged_screens = set()       # 只记录一次的界面文字（难度页、主界面等）
        self.run_cfg = self.current_run()
        self.apply_run_fate()
        self.log_plan()

    # ---------- 跑图计划 ----------
    def state_path(self):
        return config_path(__file__).with_name("gold_gears_state.json")

    def _save_active_conundrum(self, total):
        st = self.load_state()
        st["active_conundrum"] = total
        self.save_state(st)

    def load_state(self):
        try:
            return json.loads(self.state_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def save_state(self, state):
        try:
            self.state_path().write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as e:
            CUS_LOGGER.warning(f"黄金与机械状态保存失败：{e}")

    def run_key(self, run):
        """计划项唯一键：优先使用可选字段 key，默认退回骰子名以兼容历史进度。"""
        return (run or {}).get("key") or (run or {}).get("die") or ""

    def current_run(self):
        """计划里第一项「没完成、本次运行也没发现未解锁」的局；全做完返回 None。"""
        done = set(self.load_state().get("done", []))
        for run in self.gg.get("runs") or []:
            die = run.get("die")
            key = self.run_key(run)
            if die and key not in done and die not in self.locked_dice:
                return run
        return None

    def run_difficulty(self):
        run = self.run_cfg or {}
        try:
            d = min(6, max(1, int(run.get("difficulty", self.gg.get("difficulty", 4)))))
        except (TypeError, ValueError):
            d = 4
        # 这颗骰子在某难度打输过（放弃结算），之后按降过的难度打，免得同一局反复输
        lowered = (self.load_state().get("lowered") or {}).get(self.run_key(run))
        return min(d, int(lowered)) if lowered else d

    # 实机 2026-11-16：难度 V 第二位面首领一场打 4 分钟才输，父类两次失败间隔超过 180 秒就把计数清零，
    # 于是无限重新挑战（连输 3 次仍在重试）。黄金与机械只重试 1 次，计数窗口放宽到 20 分钟。
    BATTLE_RETRY_LIMIT = 1

    def battle_failed(self):
        now = time.time()
        last = getattr(self, "_battle_fail_time", 0)
        if now - last < 5:
            return
        if now - last > 1200:
            self._battle_fail_count = 0
        self._battle_fail_time = now
        self._battle_fail_count = getattr(self, "_battle_fail_count", 0) + 1
        if self._battle_fail_count <= self.BATTLE_RETRY_LIMIT:
            CUS_LOGGER.warning(f"战斗失败，第{self._battle_fail_count}次重新挑战")
            key_mouse_manager.click_point(1205, 990)
            return
        CUS_LOGGER.warning(f"连续{self._battle_fail_count}次战斗失败，终止战斗并结算本局（本局记为未通关）")
        self._battle_fail_count = 0
        self._gg_gave_up = True
        key_mouse_manager.click_point(714, 988)

    def run_conundrum(self):
        run = self.run_cfg or {}
        try:
            return min(12, max(0, int(run.get("conundrum", self.gg.get("conundrum", 0)))))
        except (TypeError, ValueError):
            return 0

    def apply_run_fate(self):
        fate = (self.run_cfg or {}).get("fate") or self.gg.get("fate") or self.fate
        if fate in FATES:
            # 无条件刷新：父类 __init__ 套用过 any_fate.json 的祝福优先级，必须用 gold_gears 配置覆盖
            self.set_fate(fate)
            CUS_LOGGER.info(f"黄金与机械祝福策略：命途 {self.fate}，优先 {self.tk.prior_bless[:6]}…，副命途 {self.tk.secondary}")

    def log_plan(self):
        state = self.load_state()
        runs = self.gg.get("runs") or []
        done = state.get("done", [])
        CUS_LOGGER.info(f"黄金与机械跑图计划：共 {len(runs)} 局，已完成 {len(done)}：{done}")
        if self.run_cfg:
            r = self.run_cfg
            CUS_LOGGER.info(f"本局：骰子【{r.get('die')}】（{r.get('category')}）难度 {self.run_difficulty()} "
                            f"难题 {self.run_conundrum()} 命途 {self.fate}；认知值目标 {r.get('cognition', {})}；{r.get('note', '')}")
        else:
            CUS_LOGGER.info("跑图计划已全部完成")

    def plan_done(self):
        return self.run_cfg is None

    # ---------- 入口 ----------
    def gg_pick_card(self):
        """扩展装置卡片列表：把「黄金与机械」转到正中再点进去（4.7 默认选中上次/进行中的宇宙）。"""
        rows = [r for r in self.ocr_rows([700, 120, 1300, 1010])
                if isinstance(r, dict) and "黄金与机械" in (r.get("raw_text") or "")]
        if not rows:
            # 列表里看不到：往下拖，把上面的卡片拉下来（实机：点非正中卡片不会转动，只有拖动有效）
            CUS_LOGGER.warning("扩展装置里没认到「黄金与机械」，往下拖列表")
            key_mouse_manager.drag_point(1100, 300, 1100, 700)
            key_mouse_manager.wait()
            return
        box = rows[0]["box"]
        cy = (box[1] + box[3]) / 2
        if 520 <= cy <= 600:
            CUS_LOGGER.info("「黄金与机械」已在正中，进入")
            key_mouse_manager.click_point(1190, 580)
        elif cy < 520:
            CUS_LOGGER.info(f"「黄金与机械」在 y={cy:.0f}，往下拖到正中")
            key_mouse_manager.drag_point(1100, 300, 1100, 700)
        else:
            CUS_LOGGER.info(f"「黄金与机械」在 y={cy:.0f}，往上拖到正中")
            key_mouse_manager.drag_point(1100, 700, 1100, 300)
        key_mouse_manager.wait()

    def _all_secrets_unlocked(self):
        """在黄金与机械主界面读取解锁进度，识别是否已达 28/28。"""
        txt = "".join(r.get("raw_text") or "" for r in self.ocr_rows([1430, 565, 1550, 665])).replace(" ", "")
        return bool(re.search(r"(?<!\d)28/28", txt))

    def gg_main(self):
        """黄金与机械主界面：点「开始探索」。骰子在开局流程的骰面装配页里选（主界面的自定义拼装只能预览）。"""
        if "main" not in self.logged_screens:
            self.logged_screens.add("main")
            self.log_screen_text("主界面", [0, 0, 1920, 1080])
        if self.run_cfg and self.run_cfg.get("optional_for_secrets") and self._all_secrets_unlocked():
            state = self.load_state()
            done = state.setdefault("done", [])
            for r in self.gg.get("runs") or []:
                k = self.run_key(r)
                if r.get("optional_for_secrets") and k and k not in done:
                    done.append(k)
            self.save_state(state)
            self.run_cfg = self.current_run()
            CUS_LOGGER.info("主界面秘闻已达 28/28，自动跳过后续可选秘闻收尾局")
        if self.plan_done():
            self.finish_plan()
            return
        CUS_LOGGER.info(f"开始探索（本局骰子【{self.run_cfg.get('die')}】）")
        key_mouse_manager.click_point(*GG_START)
        key_mouse_manager.wait()

    def gg_dice_assembly(self):
        """骰面装配界面：切到目标类别、点目标骰子、核对右侧标题，然后关掉界面。"""
        if self.plan_done():
            key_mouse_manager.click_point(*DICE_CLOSE)
            return
        die = self.run_cfg.get("die")
        category = self.run_cfg.get("category")
        # 一次 OCR 同时拿右侧标题和左侧卡片
        rows = self.ocr_rows([DICE_CARD_BOX[0], DICE_TITLE_BOX[1], DICE_TITLE_BOX[2], DICE_CARD_BOX[3]])
        title = merge_text([r for r in rows if in_box(r["box"], DICE_TITLE_BOX)])
        cards = [r for r in rows if in_box(r["box"], DICE_CARD_BOX)]
        target = next((r for r in cards if die in (r.get("raw_text") or "")), None)
        # 先查未解锁：切类别时第一张卡会自动选中，标题对上了也可能是锁着的（实测「公司时刻」）
        if target is not None:
            tb = target["box"]
            # 卡片上骰子名下方约 80 像素是「未解锁 / 暂无记录 / 最高通关记录」
            if any(30 <= (r["box"][1] - tb[1]) <= 130 and "未解锁" in (r.get("raw_text") or "") for r in cards):
                CUS_LOGGER.warning(f"骰子【{die}】未解锁，本次运行先跳过，换计划里的下一局")
                self.locked_dice.add(die)
                self.run_cfg = self.current_run()
                self.apply_run_fate()
                self.log_plan()
                return
        if die and die in title:
            self.dice_ready = True
            self.dice_attempts = 0
            # 实机 2026-11-15：开局流程（难度→命途+配队→骰面装配）里这一页右下有「确认」，点了才真正装上并开局；
            # 主界面「自定义拼装」进来的同名界面只能预览，右上角 X 关掉
            confirm = [r for r in self.ocr_rows([1500, 940, 1850, 1010])
                       if isinstance(r, dict) and (r.get("raw_text") or "").strip() == "确认"]
            if confirm:
                b = confirm[0]["box"]
                CUS_LOGGER.info(f"骰面装配已选中【{die}】，确认并开局")
                key_mouse_manager.click_point((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            else:
                CUS_LOGGER.info(f"骰面装配已选中【{die}】（预览界面），关掉")
                key_mouse_manager.click_point(*DICE_CLOSE)
            key_mouse_manager.wait()
            return
        self.dice_attempts += 1
        if self.dice_attempts > 6:
            CUS_LOGGER.error(f"骰面装配里连续 6 次没选中【{die}】，停止内核等待人工检查")
            self.log_screen_text("骰面装配", [0, 0, 1920, 1080])
            self.stop()
            return
        if target is None:
            # 当前不是目标类别（或没认到）：切到目标类别，下一轮再找
            CUS_LOGGER.info(f"切到「{category}」类找骰子【{die}】（当前卡片：{[r.get('raw_text') for r in cards]}）")
            if category in DICE_CATEGORY_TABS:
                key_mouse_manager.click_point(*DICE_CATEGORY_TABS[category])
                key_mouse_manager.wait()
            return
        tb = target["box"]
        key_mouse_manager.click_point((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2, box=tb)
        key_mouse_manager.wait()
        # 有的版本选中后要再点「装配 / 使用」类按钮，认到就点（低亮按钮点了也没事，下一轮会再核对标题）
        for r in self.ocr_rows([1450, 930, 1900, 1020]):
            text = (r.get("raw_text") or "") if isinstance(r, dict) else ""
            if any(k in text for k in ("装配", "使用", "选择")) and "编辑" not in text:
                CUS_LOGGER.info(f"点击「{text}」")
                b = r["box"]
                key_mouse_manager.click_point((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, box=b)
                key_mouse_manager.wait()
                break

    def gg_blocked_by_save(self):
        """其他模拟宇宙（不可知域）有进行中的存档：绝不点「快捷结束并结算」，关提示、写标记、停内核。"""
        CUS_LOGGER.error("不可知域有进行中的存档，黄金与机械开不了局；为保护不可知域进度，停止内核")
        try:
            os.makedirs(os.path.dirname(BLOCKED_FLAG), exist_ok=True)
            with open(BLOCKED_FLAG, "w", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S") + " 不可知域有暂离存档，黄金与机械内核已停止\n")
        except OSError:
            pass
        key_mouse_manager.press("esc")
        key_mouse_manager.wait()
        self.stop()

    def finish_plan(self):
        CUS_LOGGER.info("黄金与机械跑图计划全部完成")
        try:
            os.makedirs(os.path.dirname(DONE_FLAG), exist_ok=True)
            with open(DONE_FLAG, "w", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S") + " 跑图计划全部完成\n")
        except OSError:
            pass
        if self.gg.get("stop_when_plan_done", True):
            self.stop()

    # ---------- 开局 ----------
    def read_conundrum_state(self):
        """读取难度 V 左侧难题面板的总等级与分项等级，返回 (总等级, 属性难题, 追加难题) 或 None。"""
        text = "".join(r.get("raw_text") or "" for r in self.ocr_rows([250, 120, 880, 680]))
        m_total = re.search(r"难题等级\s*(\d+)", text)
        if not m_total:
            return None
        m_attr = re.search(r"属性难题\s*\+?\s*(\d+)", text)
        m_extra = re.search(r"追加难题\s*\+?\s*(\d+)", text)
        total = int(m_total.group(1))
        attr = int(m_attr.group(1)) if m_attr else total
        extra = int(m_extra.group(1)) if m_extra else 0
        return total, attr, extra

    def adjust_conundrum(self, target):
        """在难度 V 面板上将难题等级调到 target（0~12），调不动时立即退出避免死循环。"""
        prev = None
        stale = 0
        for _ in range(14):
            state = self.read_conundrum_state()
            if state is None:
                return
            total, attr, extra = state
            self._actual_conundrum = total
            if total == target:
                CUS_LOGGER.info(f"难题等级已调至 {total}（属性+{attr}, 追加+{extra}）")
                self._save_active_conundrum(total)
                return
            if state == prev:
                stale += 1
                if stale >= 2:
                    CUS_LOGGER.warning(f"难题等级停在 {total}（目标 {target}），不再继续调整")
                    self._save_active_conundrum(total)
                    return
            else:
                stale = 0
            prev = state
            if total > target:
                pt = (260, 625) if extra > 0 else (260, 360)
            else:
                pt = (838, 360) if attr <= extra else (838, 625)
            key_mouse_manager.click_point(*pt)
            key_mouse_manager.wait()
            time.sleep(0.45)

    def begin_universe(self):
        """难度页（首次通关奖励）：按本局计划选难度和难题等级，然后开始。"""
        self.area = ""
        self._ban_attempts = 0
        btn = self.start_buttons()
        con = any("继续进度" in (r.get("raw_text") or "") for r in btn)
        if not con:
            if "difficulty" not in self.logged_screens:
                self.logged_screens.add("difficulty")
                self.log_screen_text("难度页", [0, 100, 1920, 1080])
            difficulty = self.run_difficulty()
            # 实机 2026-11-15：难度列表 I–V 在 x=127，y≈178/290/394/501/610，和寰宇蝗灾的公式一致
            key_mouse_manager.click_point(127, 179 + 108 * (difficulty - 1))
            CUS_LOGGER.info(f"选择难度 {difficulty}")
            self.apply_run_fate()
            key_mouse_manager.wait()
            time.sleep(1)
            if difficulty == 5:
                self.adjust_conundrum(self.run_conundrum())
            btn = self.start_buttons()  # 换难度后折磨面板可能收起，按钮位置以新画面为准
        else:
            CUS_LOGGER.info("继续进度")
        if btn:
            b = btn[0]["box"]
            key_mouse_manager.click_point((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        else:
            # 实机 2026-11-15：黄金与机械的「下载初始角色」在 (1542,949)，和寰宇蝗灾的 (1712,971) 不同
            key_mouse_manager.click_point(1542, 949)

    def start_buttons(self):
        """难度页右下角的开始按钮：新开是「下载初始角色」，有存档是「继续进度」。"""
        return [r for r in self.ocr_rows([1250, 900, 1900, 1010])
                if isinstance(r, dict) and any(k in (r.get("raw_text") or "") for k in ("继续进度", "下载初始角色"))]

    def select_fate(self):
        """命途页：黄金与机械的命途数量和寰宇蝗灾不同，按文字定位点击，认不到再用父类的固定位置。"""
        # 实机 2026-11-15：8 个命途名在 y≈702（记忆 160 … 智识 1756），点名字进详情页
        for r in self.ocr_rows([100, 660, 1820, 745]):
            if isinstance(r, dict) and self.fate in (r.get("raw_text") or ""):
                b = r["box"]
                CUS_LOGGER.info(f"选择命途 {self.fate}")
                key_mouse_manager.click_point((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                return
        if "fate" not in self.logged_screens:
            self.logged_screens.add("fate")
            self.log_screen_text("命途页", [0, 0, 1920, 1080])
        CUS_LOGGER.warning(f"命途页没找到「{self.fate}」")

    @staticmethod
    def team_templates():
        """黄金与机械用自己的头像模板 user/team_goldgears（user/team 是不可知域的队）；没有就退回父类。"""
        folder = os.path.join(PATHS["user"], "team_goldgears")
        if not os.path.isdir(folder):
            return AnyFateUniverse.team_templates()
        res = []
        for name in sorted(os.listdir(folder)):
            if not name.lower().endswith((".png", ".jpg")):
                continue
            img = cv.imdecode(np.fromfile(os.path.join(folder, name), dtype=np.uint8), cv.IMREAD_COLOR)
            if img is not None:
                res.append((name, img))
        return res

    def gg_fill_team(self, already_open=False):
        """队伍为空：点第一个队伍槽打开左侧「角色列表」，按 user/team_goldgears 模板依次点头像，再点确认收起。

        实机 2026-11-15/16：队伍槽在 y≈973，x≈683/767/859/950；角色列表 3 列（x 164/298/432），
        点头像直接填进当前槽并跳到下一个槽；头像模板匹配度 0.93–0.95。
        """
        CUS_LOGGER.info("队伍为空，打开角色列表配队")
        opened = already_open
        if not opened:
            for _ in range(3):
                key_mouse_manager.click_point(683, 973)
                key_mouse_manager.wait()
                time.sleep(1.0)
                if any("角色列表" in (r.get("raw_text") or "") for r in self.ocr_rows([200, 60, 400, 115])):
                    opened = True
                    break
        if not opened:
            CUS_LOGGER.warning("配队：未能打开角色列表，跳过本轮点击")
            return
        screen = self.get_screen()
        rx1, ry1, rx2, ry2 = self.TEAM_ROSTER_BOX
        picked = []
        for name, tpl in self.team_templates():
            _, value, _, loc = cv.minMaxLoc(cv.matchTemplate(screen[ry1:ry2, rx1:rx2], tpl, cv.TM_CCOEFF_NORMED))
            if value < 0.85:
                CUS_LOGGER.warning(f"配队：角色列表里没找到 {name}（匹配度{value:.2f}）")
                continue
            key_mouse_manager.click_point(rx1 + loc[0] + tpl.shape[1] // 2, ry1 + loc[1] + tpl.shape[0] // 2)
            key_mouse_manager.wait()
            time.sleep(0.6)
            picked.append(name)
        if not picked:
            CUS_LOGGER.warning("配队：一个模板都没匹配上，按位置选列表前 4 个角色")
            for cx, cy in ((164, 188), (298, 188), (432, 188), (164, 346)):
                key_mouse_manager.click_point(cx, cy)
                key_mouse_manager.wait()
                time.sleep(0.6)
        CUS_LOGGER.info(f"配队：选了 {picked}")
        key_mouse_manager.click_point(1688, 974)  # 收起角色列表，回到命途详情，下一轮再确认
        key_mouse_manager.wait()

    def forbidden_fates(self):
        """返回本局要禁用的命途列表（最多 2 个，且不包含当前主命途）。"""
        raw = (self.run_cfg or {}).get("forbidden_fates", self.gg.get("forbidden_fates", []))
        return [f for f in (raw or []) if f in FATES and f != self.fate][:2]

    def _apply_forbidden_panel(self, want):
        """在已打开的禁用命途左侧面板里勾选目标禁用命途，并按 Esc 收起面板。"""
        slot_text = "".join(r.get("raw_text") or "" for r in self.ocr_rows([180, 970, 630, 1020]))
        current = [f for f in FATES if f in slot_text]
        panel_rows = self.ocr_rows([150, 160, 680, 700])
        # 先取消不在目标里的已禁命途，再点缺的目标命途
        for fate_name in [f for f in current if f not in want] + [f for f in want if f not in current]:
            hit = next((r for r in panel_rows if fate_name in (r.get("raw_text") or "")), None)
            if hit is not None:
                b = hit["box"]
                CUS_LOGGER.info(f"禁用命途面板点击「{fate_name}」")
                key_mouse_manager.click_point((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                key_mouse_manager.wait()
                time.sleep(0.4)
        key_mouse_manager.press("esc")
        key_mouse_manager.wait()
        time.sleep(0.5)

    def gg_fate_confirm(self):
        """命途详情页：确认主命途并按配置禁用最多两个无关命途，之后点确认。"""
        if any("角色列表" in (r.get("raw_text") or "") for r in self.ocr_rows([200, 60, 400, 115])):
            self.gg_fill_team(already_open=True)
            return
        want = self.forbidden_fates()
        # 禁用面板打开时会遮住左列主命途名（把面板里的「巡猎」露在 [100,600,400,760]），必须先处理并关掉面板
        if any("禁用命途" in (r.get("raw_text") or "") for r in self.ocr_rows([230, 55, 420, 115])):
            if want and self._ban_attempts < 3:
                self._ban_attempts += 1
                self._apply_forbidden_panel(want)
            else:
                key_mouse_manager.press("esc")
                key_mouse_manager.wait()
                time.sleep(0.5)
        text = merge_text(self.ocr_rows([100, 600, 400, 760]))
        if not text:
            # 淡入/黑屏过渡帧可能短暂空识别，不能当成「选错命途」按 Esc 退回
            time.sleep(0.5)
            return
        if self.fate not in text:
            CUS_LOGGER.info(f"命途详情是「{text}」，不是「{self.fate}」，返回重选")
            key_mouse_manager.press("esc")
            key_mouse_manager.wait()
            return
        if want and self._ban_attempts < 3:
            slot_text = "".join(r.get("raw_text") or "" for r in self.ocr_rows([180, 970, 630, 1020]))
            current = [f for f in FATES if f in slot_text]
            if set(current) != set(want):
                self._ban_attempts += 1
                CUS_LOGGER.info(f"打开禁用命途面板（当前槽位：{current or '空'}，目标：{want}）")
                key_mouse_manager.click_point(267, 995)
                key_mouse_manager.wait()
                time.sleep(0.6)
                self._apply_forbidden_panel(want)
        CUS_LOGGER.info(f"命途详情是「{self.fate}」（禁用：{want}），确认")
        key_mouse_manager.click_point(1689, 974)
        key_mouse_manager.wait()
        time.sleep(1.5)
        # 实机 2026-11-15：队伍为空时弹「请至少下载一位角色」
        if "下载一位角色" in merge_text(self.ocr_rows([600, 500, 1320, 580])):
            time.sleep(0.4)
            self.gg_fill_team()
        key_mouse_manager.wait()

    # ---------- 选首领 ----------
    # 实机 2026-11-15：第一位面首领格进去先选首领：两张卡（上卡 y≈240–490，下卡 y≈560–810），
    # 每张写「对第三位面首领的影响：首领获得 X 属性弱点」；点卡片后右下「请选择首领」(1687,979) 变亮。
    TEAM_ELEMENTS = ("量子", "雷", "风")  # 黄泉 雷 / 遐蝶、缇宝 量子 / 风堇 风
    # 第二位面选第三位面首领的【回响推演】敌方增益：越靠前对黄泉/遐蝶队威胁越小（避开丰饶回血、存护厚盾、毁灭自爆）
    ENEMY_ECHO_ORDER = ("虚无", "记忆", "欢愉", "繁育", "智识", "巡猎", "毁灭", "存护", "丰饶")

    def _echo_rank(self, card_text):
        """解析卡片上的敌方回响推演命途威胁排序，越小越安全；未识别返回 99。"""
        if "推演" not in card_text:
            return 99
        for rank, fate in enumerate(self.ENEMY_ECHO_ORDER):
            if re.search(rf"推演\D{{0,3}}{fate}", card_text) or fate in card_text:
                return rank
        return 98

    def select_head(self):
        rows = self.ocr_rows([880, 200, 1620, 840])
        cards = {0: "", 1: ""}
        for r in rows:
            if isinstance(r, dict):
                cards[0 if (r["box"][1] + r["box"][3]) / 2 < 520 else 1] += r.get("raw_text") or ""
        pick = 0
        # 第二位面「推演：命途」选敌方最弱；第一位面「获得 X 属性弱点」优先队伍属性；第三位面优先虫群（弱量子）或削弱首领词条
        ranks = (self._echo_rank(cards[0]), self._echo_rank(cards[1]))
        elem_hit = [idx for idx in (0, 1) if any(f"获得{e}" in cards[idx] for e in self.TEAM_ELEMENTS)]
        swarm_hit = [idx for idx in (0, 1) if "虫群" in cards[idx]]
        debuff_hit = [idx for idx in (0, 1) if any(k in cards[idx] for k in ("延后", "降低", "减少", "易伤", "受损", "受到的伤害"))]
        if min(ranks) < 99:
            pick = 0 if ranks[0] <= ranks[1] else 1
        elif elem_hit:
            pick = elem_hit[0]
        elif swarm_hit:
            pick = swarm_hit[0]
        elif debuff_hit:
            pick = debuff_hit[0]
        CUS_LOGGER.info(f"选首领：上卡「{cards[0][:40]}」下卡「{cards[1][:40]}」→ 选{'上' if pick == 0 else '下'}卡")
        # 实机：卡片描述里的【关键词】可点，点中会弹说明框而不是选中卡片，所以点左侧头像区
        key_mouse_manager.click_point(1000, 400 if pick == 0 else 720)
        key_mouse_manager.wait()
        time.sleep(0.6)
        btn = merge_text(self.ocr_rows([1560, 955, 1820, 1005]))
        if "确认" in btn or ("选择" in btn and "请" not in btn):
            self.head_fail = 0
            key_mouse_manager.click_point(1687, 979)
            key_mouse_manager.wait()
        else:
            # 按钮文字没变也点一次（亮暗靠下一轮再核对），连续多次失败就停
            self.head_fail = getattr(self, "head_fail", 0) + 1
            CUS_LOGGER.warning(f"选首领后按钮是「{btn}」（第 {self.head_fail} 次）")
            key_mouse_manager.click_point(1687, 979)
            key_mouse_manager.wait()
            if self.head_fail >= 6:
                self.log_screen_text("选首领", [0, 0, 1920, 1080])
                self.stop()

    def handle_rift_ui(self):
        """命途狭间/终章剧情对话：只识别右侧选项栏（x>=1220），避免误点居中字幕。"""
        last = getattr(self, "_rift_ui_time", 0)
        if time.time() - last < 1.2:
            return True
        self._rift_ui_time = time.time()
        rows = self.ts.find_with_box(box=[1220, 200, 1900, 980], forward=True, re_screen=True)
        rows = [r for r in rows if isinstance(r, dict) and len((r.get("raw_text") or "").strip()) >= 2]
        texts = [r["raw_text"] for r in rows]
        self.save_rift_screen("ui")
        leave = [r for r in rows if "离开" in r["raw_text"]]
        if leave:
            box = leave[-1]["box"]
            CUS_LOGGER.info(f"命途狭间：点击「离开」 {texts}")
            key_mouse_manager.click_point((box[0] + box[2]) // 2, (box[1] + box[3]) // 2, box=box)
        elif rows:
            pick = max(rows, key=lambda r: r["box"][3])
            box = pick["box"]
            CUS_LOGGER.info(f"命途狭间：点击选项「{pick['raw_text']}」 全部{texts}")
            key_mouse_manager.click_point((box[0] + box[2]) // 2, (box[1] + box[3]) // 2, box=box)
        else:
            CUS_LOGGER.info("命途狭间：没认到右侧选项，点击推进对话")
            key_mouse_manager.click_point(960, 900)
        key_mouse_manager.wait()
        self.last_interact_time = time.time()
        return True

    # ---------- 区域名 ----------
    # 实机 2026-11-15：黄金与机械的「自我认知」区域父类不认识，会落到「区域名没认出」分支原地转圈直到暂离；
    # 它的小地图交互点就是普通事件图标（mini_event 1.2 倍匹配 0.95），按事件区域寻路即可。
    @property
    def area(self):
        return self.__dict__.get("_gg_area", "")

    @area.setter
    def area(self, value):
        if value and "自我认知" in value and "事件" not in value:
            value = value + "事件"
        self.__dict__["_gg_area"] = value

    # ---------- 六边形棋盘寻路 ----------
    # 实机 2026-11-15：黄金与机械的棋盘是尖顶六边形格，不是寰宇蝗灾的左右节点图，父类 try_analysis_map 会抛 NoBossError。
    # 选路界面棋盘格心落在固定格点 (HEX_OX + 57*i, HEX_OY + 96*j)，i+j 为偶数；相邻格 (i±2,j)、(i±1,j±1)。
    # 只能走相邻格：点相邻格后右下「当前选中」显示类型、按钮变「确认移动」；点不相邻的格提示「不符合移动条件」。
    HEX_OX, HEX_OY, HEX_DX, HEX_DY = 1100, 520, 57, 96
    HEX_BOX = (900, 250, 1860, 880)
    HEX_AVOID = ("双倍", "精英", "冒险")
    HEX_HIGH_AVOID = ("双倍", "冒险", "交易")
    HEX_BLESS_TILES = ("战斗", "精英", "奖励", "事件")

    def read_cognition(self, img):
        """读取棋盘顶部的自我认知当前值（-40~40）与浅绿推荐区间，并更新缓存。"""
        rows = self.ocr_rows([1040, 170, 1740, 212])
        val = None
        for r in rows:
            m = re.search(r"-?\d+", r.get("raw_text") or "")
            if not m:
                continue
            v = abs(int(m.group(0)))
            if v > 40:
                continue
            cx = (r["box"][0] + r["box"][2]) / 2
            # 刻度条左端 -40 在 x≈1092，中点 0 在 x≈1403，右端 +40 在 x≈1710；用位置纠正 OCR 漏负号
            if cx < 1385:
                val = -v if v else 0
            elif cx > 1420:
                val = v
            else:
                val = 0 if v <= 2 else int(m.group(0))
            break
        if val is None:
            # 整图 OCR 常漏检小气泡中的单个数字，退回青色气泡重心反推坐标对应的认知值
            roi = img[175:210, 1040:1740]
            hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
            mask = (hsv[..., 0] >= 80) & (hsv[..., 0] <= 100) & (hsv[..., 1] > 100) & (hsv[..., 2] > 150)
            _, xs = np.where(mask)
            if len(xs) > 20:
                cx = float(xs.mean()) + 1040
                est = int(round((cx - 1403) / 305.0 * 40))
                val = 0 if abs(est) <= 2 else max(-40, min(40, est))
        if val is not None:
            self.cognition_val = val
        strip = img[219:224, 1092:1688]
        mean_g = strip[:, :, 1].mean(axis=0)
        mean_r = strip[:, :, 2].mean(axis=0)
        bright = np.where((mean_g > 210) & (mean_r > 150))[0]
        if len(bright) > 10:
            lo = int(round(-40 + (bright[0] / 596.0) * 80))
            hi = int(round(-40 + (bright[-1] / 596.0) * 80))
            self.cognition_rec = [lo, hi]
        return self.cognition_val, self.cognition_rec

    def target_cognition(self):
        """返回当前位面目标认知区间 [lo, hi]：优先游戏内浅绿推荐条，否则用跑图计划配置。"""
        if self.cognition_rec:
            return self.cognition_rec
        return ((self.run_cfg or {}).get("cognition") or {}).get(str(self.plane_floor))

    def cognition_in_target(self):
        target = self.target_cognition()
        if not target or self.cognition_val is None:
            return True
        return target[0] <= self.cognition_val <= target[1]

    def hex_scan(self, img):
        """逐格点统计：图标亮度 B（有格子时 >0.2，空格约 0）、金色边框 G（当前所在格）、暗红 R（首领格底色）。"""
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        yy, xx = np.mgrid[-55:56, -55:56]
        dist = np.sqrt(xx ** 2 + yy ** 2)
        inner, ring = dist <= 28, (dist >= 40) & (dist <= 58)
        x1, y1, x2, y2 = self.HEX_BOX
        cells = {}
        for j in range(-4, 5):
            for i in range(-6, 18):
                if (i + j) % 2:
                    continue
                x, y = self.HEX_OX + self.HEX_DX * i, self.HEX_OY + self.HEX_DY * j
                if not (x1 + 45 <= x <= x2 - 45 and y1 + 45 <= y <= y2 - 45):
                    continue
                p = hsv[y - 55:y + 56, x - 55:x + 56]
                h, s, v = p[..., 0], p[..., 1], p[..., 2]
                b = float((v[inner] > 110).mean())
                g = float(((h[ring] >= 12) & (h[ring] <= 28) & (s[ring] > 40) & (v[ring] > 200)).mean())
                r = float((((h[inner] <= 8) | (h[inner] >= 165)) & (s[inner] > 120) & (v[inner] > 90)).mean())
                cells[(i, j)] = {"x": x, "y": y, "B": b, "G": g, "R": r}
        return cells

    def hex_boss_score(self, img, x, y):
        tpls = getattr(self, "_boss_tpls", None)
        if tpls is None:
            tpls = [cv.imread(resource_path(__file__, p)) for p in ("tpl/boss.png", "tpl/boss3.png")]
            tpls = self._boss_tpls = [t for t in tpls if t is not None]
        win = img[y - 60:y + 61, x - 60:x + 61]
        return max((float(cv.minMaxLoc(cv.matchTemplate(win, t, cv.TM_CCOEFF_NORMED))[1]) for t in tpls), default=0.0)

    @staticmethod
    def _hex_neighbors(k):
        i, j = k
        return ((i + 2, j), (i + 1, j - 1), (i + 1, j + 1), (i - 1, j - 1), (i - 1, j + 1), (i - 2, j))

    @classmethod
    def _bfs_tree(cls, pool, start, banned=()):
        prev, dist, queue = {start: None}, {start: 0}, [start]
        while queue:
            k = queue.pop(0)
            for n in cls._hex_neighbors(k):
                if n in pool and n not in prev and n not in banned:
                    prev[n] = k
                    dist[n] = dist[k] + 1
                    queue.append(n)
        return prev, dist

    def hex_plan(self, img):
        """返回 (当前格, 首领格, 下一步格, 格子表)；认不出时对应项为 None。"""
        cells = self.hex_scan(img)
        # 「往哪走？」气泡贴在当前格左上，亮度会被当成格子，先用 OCR 位置剔除
        bubble = [r for r in self.ocr_rows(list(self.HEX_BOX)) if isinstance(r, dict) and "往哪" in (r.get("raw_text") or "")]
        for r in bubble:
            bx1, by1, bx2, by2 = r["box"]
            for k, c in cells.items():
                if bx1 - 50 <= c["x"] <= bx2 + 50 and by1 - 50 <= c["y"] <= by2 + 50:
                    c["B"] = 0.0
        tiles = {k: c for k, c in cells.items() if c["B"] > 0.08}
        cur = max(tiles, key=lambda k: tiles[k]["G"], default=None)
        if cur is not None and tiles[cur]["G"] < 0.08:
            cur = None
        if cur is None and bubble:
            # 兜底：当前格在气泡右下 (+64,+57)
            bx1, by1, bx2, by2 = bubble[0]["box"]
            cx, cy = (bx1 + bx2) / 2 + 64, (by1 + by2) / 2 + 57
            cur = min(cells, key=lambda k: abs(cells[k]["x"] - cx) + abs(cells[k]["y"] - cy))
            tiles[cur] = cells[cur]
        # 实格亮度 >0.35；背景空格和首领红晕只有 0.17~0.29，选首领与寻路优先仅走实格
        walkable = {k: c for k, c in tiles.items() if c["B"] > 0.35 or k == cur}
        boss_pool = {k: c for k, c in walkable.items() if k != cur} or tiles
        boss, best = None, 0.0
        for k, c in boss_pool.items():
            if k == cur:
                continue
            sc = self.hex_boss_score(img, c["x"], c["y"])
            c["boss"] = sc
            if sc > best:
                boss, best = k, sc
        if best < 0.6:
            red = [k for k in boss_pool if k != cur and boss_pool[k]["R"] > 0.12]
            boss = max(red, key=lambda k: boss_pool[k]["R"]) if red else None
        nxt = None
        path_pool = walkable
        if cur is not None and boss is not None:
            prev, _ = self._bfs_tree(walkable, cur)
            if boss not in prev:
                prev, _ = self._bfs_tree(tiles, cur)
                path_pool = tiles
            if boss in prev:
                k = boss
                while prev[k] != cur and prev[k] is not None:
                    k = prev[k]
                nxt = k
        # 和 nxt 一样近、以及多 1 步的其他相邻格，select_go 用来避开「冒险/双倍/精英」或凑认知值
        self._hex_alts = []
        self._hex_near_alts = []
        if nxt is not None and boss is not None and boss in path_pool:
            _, dist = self._bfs_tree(path_pool, boss, banned=(cur,))
            adj = self._hex_neighbors(cur)
            self._hex_alts = [n for n in adj if n != nxt and n in dist and dist[n] == dist.get(nxt)]
            self._hex_near_alts = [n for n in adj if n != nxt and n in dist and dist[n] == dist.get(nxt, 0) + 1]
        if nxt is None and cur is not None:
            # 首领没认到或不连通：往右边最远的实邻格走（首领一般在棋盘远端），并记录现场
            adj = [n for n in self._hex_neighbors(cur) if n in walkable] or [n for n in self._hex_neighbors(cur) if n in tiles]
            nxt = max(adj, key=lambda n: tiles[n]["x"], default=None)
        return cur, boss, nxt, tiles

    def select_go(self):
        """选路界面：按六边形棋盘找去首领的最短路，避开冒险/双倍/精英并兼顾自我认知，点下一格确认移动。"""
        key_mouse_manager.clean()
        key_mouse_manager.keyUp("move_w")
        key_mouse_manager.wait()
        if self.click_text(text="点击空白处关闭", box=[876, 1008, 1047, 1035], click=False, allow_fail=True):
            key_mouse_manager.wait()
            return
        # 棋盘是逐格渲染出来的（实机：刚进界面只扫到 7 格），等相邻两次扫描的格子集合一致再规划
        # 实机还见过切场景的黑屏帧（格子集合为空也「稳定」），空集不算稳定
        prev = None
        img = self.get_screen()
        for _ in range(8):
            img = self.get_screen()
            keys = {k for k, c in self.hex_scan(img).items() if c["B"] > 0.08}
            if keys and keys == prev:
                break
            prev = keys
            time.sleep(0.6)
        if not prev:
            return
        cog, rec = self.read_cognition(img)
        cur, boss, nxt, tiles = self.hex_plan(img)
        brief = " ".join(f"{k}:B{c['B']:.2f}" + (f"G{c['G']:.2f}" if c["G"] > 0.05 else "") +
                         (f"首{c.get('boss', 0):.2f}" if c.get("boss", 0) > 0.4 else "") for k, c in sorted(tiles.items()))
        CUS_LOGGER.info(f"棋盘：当前 {cur} 首领 {boss} 下一步 {nxt} 认知 {cog} 目标 {self.target_cognition()}｜{brief}")
        if nxt is None:
            self.hex_fail = getattr(self, "hex_fail", 0) + 1
            if tiles:
                self.save_hex_board(img, "noboss" if boss is None else "nopath")
            CUS_LOGGER.warning(f"棋盘上没找到可走的相邻格（第 {self.hex_fail} 次），下一轮再看")
            if self.hex_fail >= 10:
                CUS_LOGGER.error("连续 10 次找不到可走的格子，停止内核等待人工检查")
                self.stop()
            return
        need_cog = not self.cognition_in_target()
        high_diff = self.run_difficulty() >= 5 or self.run_conundrum() > 0
        avoid = self.HEX_HIGH_AVOID if high_diff else self.HEX_AVOID

        def _is_good(ktext):
            if any(w in ktext for w in avoid):
                return False
            if high_diff and "首领" not in ktext and not any(w in ktext for w in self.HEX_BLESS_TILES):
                return False
            return True

        same = [nxt] + list(getattr(self, "_hex_alts", []))
        near = list(getattr(self, "_hex_near_alts", []))
        candidates = []
        tried = set()
        last = None
        for idx, n in enumerate(same + near):
            has_cog = any("自我认知" in c[1] for c in candidates)
            if idx >= len(same) and any(_is_good(c[1]) for c in candidates) and (not need_cog or has_cog):
                break
            tried.add(n)
            kind, btn = self.hex_select(tiles[n])
            last = n
            CUS_LOGGER.info(f"选中格 {n}：{kind}（按钮「{btn}」）")
            if "确认移动" not in btn:
                continue
            candidates.append((idx < len(same), kind, n))
            is_good = _is_good(kind)
            if is_good and (not need_cog or "自我认知" in kind):
                break
            if idx < len(same) and is_good and not high_diff and idx == len(same) - 1 and (not need_cog or has_cog):
                break
        if not candidates and cur is not None:
            extras = [n for n in self._hex_neighbors(cur) if n in tiles and n not in tried]
            extras.sort(key=lambda n: (-tiles[n]["B"], -tiles[n]["x"]))
            for n in extras[:4]:
                kind, btn = self.hex_select(tiles[n])
                last = n
                CUS_LOGGER.info(f"备选格 {n}：{kind}（按钮「{btn}」）")
                if "确认移动" in btn:
                    candidates.append((False, kind, n))
                    break
        chosen = None
        if candidates:
            # 优先级：不踩避让格 > 认知未达标时踩自我认知格 > 高难优先战斗/精英/奖励/事件拿祝福 > 同距最短
            def rank(item):
                is_same, ktext, _ = item
                return (
                    0 if not any(w in ktext for w in avoid) else (1 if "冒险" not in ktext else 2),
                    0 if (need_cog and "自我认知" in ktext) else 1,
                    0 if (high_diff and ("首领" in ktext or any(w in ktext for w in self.HEX_BLESS_TILES))) else 1,
                    0 if is_same else 1,
                )
            chosen = min(candidates, key=rank)[2]
        if chosen is not None and chosen != last:
            kind, btn = self.hex_select(tiles[chosen])
            CUS_LOGGER.info(f"改选 {chosen}：{kind}（按钮「{btn}」）")
        elif chosen is not None:
            btn = "确认移动"
        nxt = chosen if chosen is not None else nxt
        if "确认移动" in btn:
            key_mouse_manager.click_point(1687, 984)
            key_mouse_manager.wait()
            self.hex_fail = 0
            self.area = ""
            self.new_node = True
            self.fail_count = 0
            self.head_fail = 0
            self._node_rerolls = 0
            self._bless_shop_tries = 0
            self._bless_shop_bought = 0
        else:
            self.hex_fail = getattr(self, "hex_fail", 0) + 1
            CUS_LOGGER.warning(f"点了 {nxt} 但没出现「确认移动」（第 {self.hex_fail} 次）")
            if self.hex_fail >= 5:
                self.save_hex_board(img, "nomove")
                self.stop()

    def init_map(self):
        new_node = self.new_node
        super().init_map()
        if new_node:
            self._node_rerolls = 0
            self._bless_shop_tries = 0
            self._bless_shop_bought = 0

    def calculated_roll(self):
        """投骰界面：位面未知时先开地图认位面；需要调认知或掷出抹除且富余重投时重投 1 次，否则确认效果。"""
        if self.plane_floor == -1:
            self.click_target(find_image_by_name("inmap"), 0.9, flag=False, click=True)
            key_mouse_manager.wait()
            return
        face_text = "".join(r.get("raw_text", "") for r in self.ocr_rows([1180, 420, 1740, 580]))
        counts_text = "".join(r.get("raw_text", "") for r in self.ocr_rows([1180, 690, 1760, 755]))
        redo_m = re.search(r"重投次数\D*(\d+)", counts_text)
        cheat_m = re.search(r"作弊次数\D*(\d+)", counts_text)
        redo = int(redo_m.group(1)) if redo_m else 0
        cheat = int(cheat_m.group(1)) if cheat_m else 0
        if face_text or counts_text:
            CUS_LOGGER.info(f"投骰界面：骰面「{face_text[:60]}」｜重投 {redo} 作弊 {cheat}")
        if not self.check("fast_roll", 1674, 100, threshold=0.9):
            self.click_text(text="快速投掷", box=[1700, 80, 1823, 117])
        target = self.target_cognition()
        cur = self.cognition_val
        need_neg = bool(target and cur is not None and cur > target[1])
        need_pos = bool(target and cur is not None and cur < target[0])
        helpful_cog = ("认知" in face_text) or (need_neg and "感性" in face_text) or (need_pos and "理性" in face_text)
        wrong_cog = (need_pos and "感性" in face_text) or (need_neg and "理性" in face_text)
        reroll_for_bad = ("抹除" in face_text or wrong_cog) and redo >= 1
        reroll_for_cog = not self.cognition_in_target() and not helpful_cog and redo >= 2
        if self._node_rerolls < 1 and (reroll_for_cog or reroll_for_bad):
            if self.click_text(text="重投", box=[1599, 760, 1657, 795], allow_fail=True):
                self._node_rerolls += 1
                reason = "避开抹除/反向认知" if reroll_for_bad else "争取认知调整"
                CUS_LOGGER.info(f"投骰重投第 {self._node_rerolls} 次（{reason}，剩余重投 {max(0, redo - 1)}）")
                key_mouse_manager.wait()
                return
        self.click_text(text="确认效果", box=[1584, 961, 1687, 994])
        self.init_map()
        self.mini_state = 1

    def gg_bless_shop(self):
        """祝福商店：按 prior_bless 优先级挑选祝福购买；最多买 2 个或尝试 3 次后退出，不卡死。"""
        if self.click_text(text="点击空白", box=[760, 700, 1160, 1030], allow_fail=True):
            key_mouse_manager.wait()
            return
        if "祝福商店" not in self.logged_screens:
            self.logged_screens.add("祝福商店")
            self.log_screen_text("祝福商店", [0, 0, 1920, 1080])
        if self._bless_shop_tries >= 3 or self._bless_shop_bought >= 2:
            CUS_LOGGER.info(f"退出祝福商店（尝试 {self._bless_shop_tries} 次，已购 {self._bless_shop_bought} 个）")
            key_mouse_manager.press("esc")
            key_mouse_manager.wait()
            return
        self._bless_shop_tries += 1
        tk = getattr(self, "tk", None)
        skip_words = {"命途回响", "回响交错", "回响构音"}
        prior = [w for w in (getattr(tk, "prior_bless", None) or []) if w and w not in skip_words]
        extra = [w for grp in (getattr(tk, "blesses", None) or []) for w in grp if w and w not in skip_words]
        rows = self.ocr_rows([150, 180, 1750, 780])
        sold_boxes = [r["box"] for r in rows if any(k in (r.get("raw_text") or "") for k in ("售罄", "已购", "已拥有"))]
        cands = []
        for r in rows:
            txt = (r.get("raw_text") or "").strip()
            b = r.get("box")
            if not txt or not b or any(k in txt for k in ("售罄", "已购", "已拥有", "碎片", "刷新")):
                continue
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if any(abs((sb[0] + sb[2]) / 2 - cx) < 140 and -40 <= sb[1] - cy <= 260 for sb in sold_boxes):
                continue
            rank = next((i for i, k in enumerate(prior) if k in txt), None)
            if rank is None and any(k in txt for k in extra):
                rank = len(prior)
            if rank is not None:
                cands.append((rank, cy, cx, txt))
        if not cands:
            CUS_LOGGER.info("祝福商店未识别到可购祝福，按 Esc 退出")
            key_mouse_manager.press("esc")
            key_mouse_manager.wait()
            return
        _, cy, cx, card_txt = min(cands)
        key_mouse_manager.click_point(cx, cy)
        key_mouse_manager.wait()
        time.sleep(0.4)
        btn_rows = self.ocr_rows([1450, 910, 1820, 1020])
        buy = next(
            (r for r in btn_rows
             if "无法" not in (r.get("raw_text") or "") and "不足" not in (r.get("raw_text") or "")
             and any(w in (r.get("raw_text") or "") for w in ("购买", "确认", "兑换"))),
            None,
        )
        if not buy:
            CUS_LOGGER.info(f"祝福商店无可用购买按钮（识别：{[r.get('raw_text') for r in btn_rows]}），退出")
            key_mouse_manager.press("esc")
            key_mouse_manager.wait()
            return
        bb = buy["box"]
        screen = self.get_screen()
        roi = screen[max(0, bb[1] - 15):min(1080, bb[3] + 15), max(0, bb[0] - 50):min(1920, bb[2] + 50)]
        bright_ratio = float((roi.mean(axis=2) > 190).mean()) if roi.size else 0.0
        if bright_ratio < 0.15:
            CUS_LOGGER.info(f"祝福商店「{buy.get('raw_text')}」按钮灰暗（亮度占比 {bright_ratio:.2f}），碎片不足退出")
            key_mouse_manager.press("esc")
            key_mouse_manager.wait()
            return
        key_mouse_manager.click_point((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)
        key_mouse_manager.wait()
        self._bless_shop_bought += 1
        CUS_LOGGER.info(f"祝福商店购买「{card_txt}」（第 {self._bless_shop_bought} 个）")
        time.sleep(0.6)
        self.click_text(text="点击空白", box=[760, 700, 1160, 1030], allow_fail=True)
        key_mouse_manager.wait()

    def try_analysis_map(self, mode=1, path_mode=2, image=None, matches=None):
        """父类按蝗灾节点图识别，黄金与机械的六边形棋盘一定认不出并抛 NoBossError（实机：地图更新页）。
        寻路在 select_go 里按六边形格做，这里不再分析。"""
        self.expectation_weight = 0
        return True

    def hex_select(self, c):
        """点一个格子，返回（「当前选中」的格子类型, 右下按钮文字）。"""
        key_mouse_manager.click_point(c["x"], c["y"])
        key_mouse_manager.wait()
        time.sleep(0.6)
        kind = merge_text(self.ocr_rows([1480, 885, 1760, 930]))
        btn = merge_text(self.ocr_rows([1600, 960, 1780, 1005]))
        return kind, btn

    def save_hex_board(self, img, tag):
        folder = os.path.join(PATHS["temp"], "gg_boards")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{time.strftime('%m%d_%H%M%S')}_{tag}.png")
        cv.imwrite(path, img)
        CUS_LOGGER.info(f"棋盘截图已存 {path}")

    # ---------- 位面 ----------
    def detect_plane(self, box):
        """先按关键词认位面名；认不到时，在位面总览页按顺序推断（每个位面只推断一次）。"""
        key_mouse_manager.keyUp("move_w")
        old_plane = self.plane_floor
        overview = box[1] == 0
        # 实机 2026-11-15：黄金与机械总览页左上没有位面名，位面名（如「疑问」）在左侧菱形徽章下方 (468,480) 附近
        text = merge_text(self.ocr_rows([300, 330, 650, 520] if overview else box))
        for plane, words in (self.gg.get("plane_keywords") or {}).items():
            if any(w and w in text for w in words):
                self.plane_floor = int(plane)
                if self.plane_floor != old_plane:
                    self.cognition_rec = None
                CUS_LOGGER.info(f"位面名「{text}」→ 第 {self.plane_floor} 位面")
                return True
        CUS_LOGGER.warning(f"位面名没认出：「{text}」")
        # 只有「进入位面」总览页（框在左上角）才按顺序推断；地图更新页沿用当前位面
        if not overview:
            if self.plane_floor > 0:
                return True
            if not text:
                return False
            # 中途重启后不知道是第几位面：先按第 1 位面算（通关判定可能偏差），免得在地图更新页死循环
            self.plane_floor = 1
            if old_plane != 1:
                self.cognition_rec = None
            CUS_LOGGER.warning(f"地图更新页位面未知（「{text}」），先按第 1 位面处理")
            return True
        if time.time() - self.plane_seq_time < 90 and self.plane_floor > 0:
            return True
        self.plane_floor = 1 if self.plane_floor < 1 else min(3, self.plane_floor + 1)
        if self.plane_floor != old_plane:
            self.cognition_rec = None
        self.plane_seq_time = time.time()
        CUS_LOGGER.info(f"按顺序推断为第 {self.plane_floor} 位面（请把位面名补进 plane_keywords）")
        return True

    def on_plane_detected(self):
        target = self.target_cognition()
        if target:
            CUS_LOGGER.info(f"第 {self.plane_floor} 位面首领战认知值目标 {target}（当前：{self.cognition_val}）")
        if self.gg.get("save_boards", False):
            self.save_board()
        return True

    # ---------- 自我认知事件调值 ----------
    @staticmethod
    def parse_cognition_delta(text, plane_floor=2):
        """从选项及其说明文字中解析认知变化值，非认知调整选项返回 None。"""
        is_cog = any(k in text for k in ("自我认知", "我选择『", "我是有机", "我是无机", "提升『智力』", "提升『魅力』"))
        if not is_cog:
            return None
        m = re.search(r"(提高|提升|增加|降低|减少)\D{0,6}(\d+)", text)
        if m:
            num = int(m.group(2))
            return num if m.group(1) in ("提高", "提升", "增加") else -num
        step = 40 if plane_floor == 3 else 20
        if any(k in text for k in ("我是有机", "我选择『善』", "提升『智力』")):
            return step
        if any(k in text for k in ("我是无机", "我选择『恶』", "提升『魅力』")):
            return -step
        if any(k in text for k in ("不调整", "保持", "放弃")):
            return 0
        return None

    def click_event_prior(self, event_name, rows):
        """优先根据目标认知区间选择自我认知事件选项，否则沿用通用优选项逻辑。"""
        parsed = []
        for row in rows:
            full = " ".join(line.get("raw_text", "") for line in (row.get("lines") or [row]))
            delta = self.parse_cognition_delta(full, self.plane_floor)
            if delta is not None:
                parsed.append((row, delta, full))
        if parsed:
            target = self.target_cognition() or [0, 40]
            cur = self.cognition_val if self.cognition_val is not None else 0
            lo, hi = target[0], target[1]
            mid = (lo + hi) / 2

            def dist_to(val):
                return 0 if lo <= val <= hi else (lo - val if val < lo else val - hi)

            def score(item):
                _, delta, _ = item
                nv = max(-40, min(40, cur + delta))
                return (dist_to(nv), abs(nv - mid))

            best_row, best_delta, _ = min(parsed, key=score)
            if dist_to(max(-40, min(40, cur + best_delta))) > dist_to(cur):
                parsed_ids = {id(r) for r, _, _ in parsed}
                neutral = [r for r in rows if id(r) not in parsed_ids]
                if neutral:
                    CUS_LOGGER.info(f"自我认知当前 {cur}，目标 {target}，认知选项均会远离目标，改走非认知选项")
                    hit = super().click_event_prior(event_name, neutral)
                    if hit:
                        return hit
                    if self.click_event_option(event_name, neutral[-1], from_prior=False):
                        return neutral[-1]
            nv = max(-40, min(40, cur + best_delta))
            CUS_LOGGER.info(f"自我认知调整：当前 {cur}，目标 {target}，选「{best_row['raw_text']}」（{best_delta:+d} → {nv}）")
            if self.click_event_option(event_name, best_row, from_prior=True):
                self.cognition_val = nv
                return best_row
            return None
        return super().click_event_prior(event_name, rows)

    # ---------- 卡死兜底与冒险 ----------
    def leave_universe(self, *, need_press_esc=False):
        """同一节点连续超时暂离达到 3 次时直接结束并结算本局，避免无限卡死。"""
        if not self.need_end and self.need_away and not self.away_keep_status and self.fail_count >= 2:
            CUS_LOGGER.error(
                f"第 {self.plane_floor} 位面「{self.area}」已连续暂离 {self.fail_count} 次，第 3 次触发自动结束并结算"
            )
            self.need_end = True
            self.need_away = False
            self._gg_gave_up = True
        if self.need_end:
            if need_press_esc:
                key_mouse_manager.clean()
                key_mouse_manager.wait()
                key_mouse_manager.keyUp("move_w")
                key_mouse_manager.press("esc")
                key_mouse_manager.wait()
                deadline = time.time() + 3
                while time.time() < deadline:
                    if self.click_text(text="暂离", box=[1321, 787, 1383, 821], click=False, allow_fail=True):
                        break
                    key_mouse_manager.wait()
                else:
                    CUS_LOGGER.debug("未识别到暂离文本，保留结束请求等待重试")
                    return False
            self.update_state("exit")
            end_rows = [r for r in self.ocr_rows([1260, 920, 1540, 995]) if "结束" in (r.get("raw_text") or "")]
            if end_rows:
                b = end_rows[0]["box"]
                key_mouse_manager.click_point((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            else:
                key_mouse_manager.click_point(1403, 957)
            key_mouse_manager.wait()
            time.sleep(0.8)
            ok_rows = [r for r in self.ocr_rows([1050, 620, 1320, 720]) if "确认" in (r.get("raw_text") or "")]
            if ok_rows:
                b = ok_rows[0]["box"]
                key_mouse_manager.click_point((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            else:
                key_mouse_manager.click_point(1187, 671)
            key_mouse_manager.wait()
            self.away_keep_status = False
            return True
        return super().leave_universe(need_press_esc=need_press_esc)

    def get_adventure(self):
        """冒险格：先原地检查是否有可按 F 交互的机关，避免开局直接冲离机关；无法完成则由暂离兜底结束。"""
        key_mouse_manager.keyUp("move_w")
        key_mouse_manager.wait()
        if self.check("interact", mask=MASK["interact_icon"], threshold=0.96, fresh=True):
            CUS_LOGGER.info("冒险格检测到可交互机关，按 F 交互")
            key_mouse_manager.press("interact")
            key_mouse_manager.wait()
            time.sleep(1.5)
            # 若弹出罗盘类「停止」按钮（共 3 次），连续点停混过结算，避免卡死
            for step in range(3):
                stop_rows = [r for r in self.ocr_rows([1450, 880, 1850, 1020]) if "停止" in (r.get("raw_text") or "")]
                if not stop_rows:
                    break
                b = stop_rows[0]["box"]
                CUS_LOGGER.info(f"冒险小游戏点击「停止」第 {step + 1} 次")
                key_mouse_manager.click_point((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                key_mouse_manager.wait()
                time.sleep(1.2)
            return
        super().get_adventure()

    def save_board(self):
        try:
            folder = os.path.join(PATHS["temp"], "gg_boards")
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f"{time.strftime('%m%d_%H%M%S')}_p{self.plane_floor}.png")
            cv.imwrite(path, self.get_screen())
            CUS_LOGGER.debug(f"已保存位面总览：{path}")
        except Exception as e:
            CUS_LOGGER.warning(f"保存位面总览失败：{e}")

    # ---------- 结算 ----------
    def end_of_university(self):
        self.log_screen_text("结算页", [0, 0, 1920, 1080])
        settle_text = merge_text(self.ocr_rows([350, 480, 620, 860]))
        gave_up = getattr(self, "_gg_gave_up", False)
        self._gg_gave_up = False
        if "探索中断" in settle_text:
            success = False
        elif not gave_up and (self.plane_floor == 3 or "第三位面" in settle_text or "探索完成" in settle_text):
            success = True
        else:
            success = self.plane_floor == 3 and not gave_up
        state = self.load_state()
        run = self.run_cfg or {}
        rkey = self.run_key(run)
        target_con = self.run_conundrum()
        actual_con = self._actual_conundrum if self._actual_conundrum is not None else state.get("active_conundrum", 0)
        if not success and rkey:
            lower = max(3, self.run_difficulty() - 1)
            if gave_up and not target_con:
                state.setdefault("lowered", {})[rkey] = lower
                CUS_LOGGER.warning(f"骰子【{run.get('die')}】这局打输或卡死放弃，下次改用难度 {lower}")
        entry = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "die": run.get("die"),
            "difficulty": self.run_difficulty(),
            "conundrum": actual_con,
            "fate": self.fate,
            "plane": self.plane_floor,
            "success": success,
            "minutes": int((time.time() - getattr(self, "run_start_time", time.time())) / 60),
        }
        state.setdefault("history", []).append(entry)
        if success and rkey:
            if target_con and actual_con < target_con:
                CUS_LOGGER.info(f"本局通过难题 {actual_con}（目标难题 {target_con}），暂不完结本项以继续解锁更高难题")
            elif rkey not in state.setdefault("done", []):
                state["done"].append(rkey)
        state.pop("active_conundrum", None)
        self.save_state(state)
        CUS_LOGGER.info(f"黄金与机械本局结算：{'通关' if success else '未通关'} {entry}")
        super().end_of_university()
        # 下一局重新选骰子、位面从头数、刷新认知和命途配置
        self.plane_floor = -1
        self.plane_seq_time = 0
        self.dice_ready = False
        self.dice_attempts = 0
        self._ban_attempts = 0
        self.head_fail = 0
        self._actual_conundrum = None
        self.cognition_val = None
        self.cognition_rec = None
        self._node_rerolls = 0
        self._bless_shop_tries = 0
        self._bless_shop_bought = 0
        self.run_cfg = self.current_run()
        self.apply_run_fate()
        self.log_plan()

    def settle_university(self):
        super().settle_university()
        if not self._stop and self.plan_done():
            self.finish_plan()

    # ---------- 工具 ----------
    def ocr_rows(self, box):
        """重新截整屏做一次 OCR，返回落在 box 里的文字块（坐标为窗口像素）。

        不用 find_with_box 的 forward 模式：那会先过滤掉非白色文字，
        卡片上灰色的「未解锁」「暂无记录」之类的小字就认不到了。
        """
        self.ts.forward(self.get_screen())
        return [r for r in self.ts.find_with_box(box=box, redundancy=10) if isinstance(r, dict)]

    def log_screen_text(self, label, box):
        """把整块界面的文字和坐标记进日志（每种界面只记一次），用来补触发器坐标。"""
        try:
            rows = self.ocr_rows(box)
            items = [f"{r.get('raw_text')}@({(r['box'][0] + r['box'][2]) // 2},{(r['box'][1] + r['box'][3]) // 2})"
                     for r in rows if isinstance(r, dict)]
            CUS_LOGGER.info(f"[界面记录:{label}] " + " | ".join(items)[:1500])
        except Exception as e:
            CUS_LOGGER.warning(f"记录{label}文字失败：{e}")


def create_engine(*, script=False):
    return GoldGearsUniverse()
