"""Shared Star Rail window discovery and validation helpers."""

from __future__ import annotations

import ctypes
from contextlib import contextmanager
from dataclasses import dataclass

import win32con
import win32gui


@contextmanager
def _physical_pixel_context():
    """Temporarily disable DPI virtualization for Win32 geometry calls."""
    setter = getattr(ctypes.windll.user32, "SetThreadDpiAwarenessContext", None)
    previous = None
    if setter is not None:
        setter.argtypes = (ctypes.c_void_p,)
        setter.restype = ctypes.c_void_p
        try:
            # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 == (HANDLE)-4
            previous = setter(ctypes.c_void_p(-4))
        except OSError:
            previous = None
    try:
        yield
    finally:
        if setter is not None and previous:
            try:
                setter(previous)
            except OSError:
                pass


def _get_client_rect(hwnd: int) -> tuple[int, int, int, int]:
    with _physical_pixel_context():
        return win32gui.GetClientRect(hwnd)


@dataclass(frozen=True)
class GameWindow:
    hwnd: int
    kind: str
    title: str
    class_name: str
    client_width: int
    client_height: int


def get_window_kind(hwnd: int) -> str | None:
    """Return ``local``/``cloud`` only for real game host windows.

    Matching the class is intentional: Windows 11 creates an Explorer-owned
    ``Windows.Internal.Shell.TabProxyWindow`` with the exact cloud-game title.
    Its client area can be 152x0 and must never be treated as the game.
    """
    if not hwnd or not win32gui.IsWindow(hwnd):
        return None
    title = win32gui.GetWindowText(hwnd)
    class_name = win32gui.GetClassName(hwnd)
    if class_name == "UnityWndClass" and title == "崩坏：星穹铁道":
        return "local"
    if class_name == "Chrome_WidgetWin_1" and title.startswith("云·星穹铁道"):
        return "cloud"
    return None


def is_usable_game_window(hwnd: int) -> bool:
    if get_window_kind(hwnd) is None:
        return False
    if not win32gui.IsWindowVisible(hwnd) or win32gui.IsIconic(hwnd):
        return False
    left, top, right, bottom = _get_client_rect(hwnd)
    return right > left and bottom > top


def inspect_game_window(hwnd: int) -> GameWindow | None:
    kind = get_window_kind(hwnd)
    if kind is None:
        return None
    left, top, right, bottom = _get_client_rect(hwnd)
    return GameWindow(
        hwnd=hwnd,
        kind=kind,
        title=win32gui.GetWindowText(hwnd),
        class_name=win32gui.GetClassName(hwnd),
        client_width=max(0, right - left),
        client_height=max(0, bottom - top),
    )


def is_supported_resolution(kind: str, width: int, height: int) -> bool:
    """Validate local strictly and cloud with a small browser-frame tolerance."""
    if kind == "local":
        return width == 1920 and height == 1080
    if kind == "cloud":
        return (
            abs(width - 1920) <= 8
            and abs(height - 1080) <= 8
        )
    return False


def get_client_screen_rect(hwnd: int) -> tuple[int, int, int, int]:
    """Return the client rectangle in physical screen coordinates."""
    with _physical_pixel_context():
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        screen_left, screen_top = win32gui.ClientToScreen(hwnd, (left, top))
        screen_right, screen_bottom = win32gui.ClientToScreen(hwnd, (right, bottom))
    return screen_left, screen_top, screen_right, screen_bottom


def _candidate_score(window: GameWindow, z_order: int) -> tuple[int, int, int]:
    supported = int(
        is_supported_resolution(
            window.kind,
            window.client_width,
            window.client_height,
        )
    )
    area = window.client_width * window.client_height
    return supported, area, -z_order


def find_game_window(prefer_foreground: bool = True) -> GameWindow | None:
    """Find the real local or cloud game host, excluding shell proxy windows."""
    if prefer_foreground:
        foreground = win32gui.GetForegroundWindow()
        if is_usable_game_window(foreground):
            return inspect_game_window(foreground)

    candidates: list[GameWindow] = []

    def callback(hwnd: int, _extra: object) -> bool:
        if is_usable_game_window(hwnd):
            window = inspect_game_window(hwnd)
            if window is not None:
                candidates.append(window)
        return True

    win32gui.EnumWindows(callback, None)
    if not candidates:
        return None
    return max(
        enumerate(candidates),
        key=lambda item: _candidate_score(item[1], item[0]),
    )[1]


def set_game_foreground() -> GameWindow | None:
    window = find_game_window(prefer_foreground=True)
    if window is None:
        return None
    try:
        if win32gui.IsIconic(window.hwnd):
            win32gui.ShowWindow(window.hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(window.hwnd)
    except Exception:
        # Windows can deny focus stealing. The caller will keep waiting until
        # the user activates the selected game window.
        pass
    return window


def get_foreground_game_window() -> GameWindow | None:
    hwnd = win32gui.GetForegroundWindow()
    if not is_usable_game_window(hwnd):
        return None
    return inspect_game_window(hwnd)


def canonical_game_title(hwnd: int) -> str | None:
    kind = get_window_kind(hwnd)
    if kind == "local":
        return "崩坏：星穹铁道"
    if kind == "cloud":
        return "云·星穹铁道"
    return None


def fix_local_window_size(hwnd: int, width: int = 1920, height: int = 1080) -> str:
    """Try to bring a local game window to a width x height client area.

    游戏有时会无视注册表以全屏/错误尺寸启动（实测 4.7 掉线重启后出现过 2560x1600 全屏）。
    全屏（窗口与客户区同尺寸、无边框）时先发 Alt+Enter 切回窗口模式，再按边框厚度把客户区设为目标尺寸。
    调用方需已是管理员进程（游戏以管理员运行，UIPI 会拦截低权限进程的按键与 SetWindowPos）。

    Returns:
        描述本次做了什么的文本，便于写日志。
    """
    import time

    import win32api

    actions = []
    with _physical_pixel_context():
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        _, _, client_w, client_h = win32gui.GetClientRect(hwnd)
        if (right - left, bottom - top) == (client_w, client_h):
            try:
                win32gui.SetForegroundWindow(hwnd)
            except Exception:
                pass
            win32api.keybd_event(win32con.VK_MENU, 0x38, 0, 0)
            win32api.keybd_event(win32con.VK_RETURN, 0x1C, 0, 0)
            win32api.keybd_event(win32con.VK_RETURN, 0x1C, win32con.KEYEVENTF_KEYUP, 0)
            win32api.keybd_event(win32con.VK_MENU, 0x38, win32con.KEYEVENTF_KEYUP, 0)
            actions.append("全屏→Alt+Enter")
            time.sleep(2.5)
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            _, _, client_w, client_h = win32gui.GetClientRect(hwnd)
        border_w = (right - left) - client_w
        border_h = (bottom - top) - client_h
        win32gui.SetWindowPos(
            hwnd, 0, max(0, left), max(0, top), width + border_w, height + border_h,
            win32con.SWP_NOZORDER | win32con.SWP_NOACTIVATE,
        )
        actions.append(f"客户区 {client_w}x{client_h}→{width}x{height}")
    time.sleep(1.5)
    return "，".join(actions)


def ensure_game_window_size_async(timeout: float = 600, logger=None) -> None:
    """拉起游戏后在后台线程里盯住窗口 timeout 秒：期间只要客户区不是 1920x1080 就纠正。

    4.7 实测游戏常无视注册表与 -screen-* 参数以 2560x1600 全屏启动，只能启动后纠正。
    """
    import threading
    import time

    def worker():
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(5)
            window = find_game_window(prefer_foreground=False)
            if window is None or window.kind != "local":
                continue
            if is_supported_resolution(window.kind, window.client_width, window.client_height):
                # 启动画面是 1920x1080，进入加载后游戏还可能自己切成全屏，所以不提前退出，盯满 timeout
                continue
            try:
                done = fix_local_window_size(window.hwnd)
                if logger:
                    logger.warning(f"游戏窗口为 {window.client_width} × {window.client_height}，已自动纠正：{done}")
            except Exception as error:
                if logger:
                    logger.warning(f"纠正游戏窗口尺寸失败，稍后重试：{error}")

    threading.Thread(target=worker, name="纠正游戏窗口", daemon=True).start()
