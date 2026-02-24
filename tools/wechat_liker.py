#!/usr/bin/env python3
"""微信朋友圈自动点赞工具 - 通过模板匹配 + 模拟鼠标操作实现"""

import argparse
import glob
import logging
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyautogui

# --- 配置 ---
BASE_DIR = Path.home() / ".wechat-liker"
TEMPLATES_DIR = BASE_DIR / "templates"
LOGS_DIR = BASE_DIR / "logs"
SCREENSHOT_PATH = BASE_DIR / "current_screen.png"

TEMPLATE_NAMES = {
    "dots_button": "操作按钮 (帖子旁边的 '...' 按钮)",
    "liked_state": "已赞状态 (已经点过赞的标识)",
    "cancel_button": "取消按钮 (已赞帖子弹出菜单中的 '取消')",
}

# pyautogui 安全设置
pyautogui.FAILSAFE = True  # 鼠标移到左上角可中断
pyautogui.PAUSE = 0.1

# --- 日志 ---
def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)

log = setup_logging()

# --- 贝塞尔曲线鼠标移动 ---
def bezier_point(t, p0, p1, p2, p3):
    """三阶贝塞尔曲线插值"""
    u = 1 - t
    return u**3 * p0 + 3 * u**2 * t * p1 + 3 * u * t**2 * p2 + t**3 * p3


def human_move(x, y, duration=None):
    """用贝塞尔曲线模拟人类鼠标移动"""
    start_x, start_y = pyautogui.position()
    if duration is None:
        dist = ((x - start_x) ** 2 + (y - start_y) ** 2) ** 0.5
        duration = max(0.3, min(1.2, dist / 800))

    # 随机控制点，让轨迹有弧度
    cx1 = start_x + (x - start_x) * random.uniform(0.2, 0.4) + random.randint(-30, 30)
    cy1 = start_y + (y - start_y) * random.uniform(0.0, 0.3) + random.randint(-30, 30)
    cx2 = start_x + (x - start_x) * random.uniform(0.6, 0.8) + random.randint(-20, 20)
    cy2 = start_y + (y - start_y) * random.uniform(0.7, 1.0) + random.randint(-20, 20)

    steps = max(20, int(duration * 60))
    for i in range(steps + 1):
        t = i / steps
        # 缓入缓出
        t = t * t * (3 - 2 * t)
        mx = bezier_point(t, start_x, cx1, cx2, x)
        my = bezier_point(t, start_y, cy1, cy2, y)
        pyautogui.moveTo(int(mx), int(my), _pause=False)
        time.sleep(duration / steps)


def human_click(x, y):
    """移动到目标位置并点击，带随机偏移"""
    offset_x = random.randint(-3, 3)
    offset_y = random.randint(-3, 3)
    human_move(x + offset_x, y + offset_y)
    time.sleep(random.uniform(0.05, 0.15))
    pyautogui.click()


def random_delay(low=3.0, high=8.0):
    """正态分布随机延迟"""
    mean = (low + high) / 2
    std = (high - low) / 6
    delay = max(low, min(high, random.gauss(mean, std)))
    log.info(f"等待 {delay:.1f} 秒...")
    time.sleep(delay)


def reading_pause():
    """模拟阅读停顿"""
    delay = random.uniform(2.0, 5.0)
    log.info(f"模拟阅读 {delay:.1f} 秒...")
    time.sleep(delay)


# --- 截屏 ---
def take_screenshot():
    """使用 macOS screencapture 截屏"""
    SCREENSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["screencapture", "-x", str(SCREENSHOT_PATH)],
        check=True, capture_output=True,
    )
    img = cv2.imread(str(SCREENSHOT_PATH))
    if img is None:
        raise RuntimeError("截屏失败，请检查屏幕录制权限")
    return img


# --- 模板匹配 ---
def load_template(name):
    """加载模板图片"""
    path = TEMPLATES_DIR / f"{name}.png"
    if not path.exists():
        return None
    img = cv2.imread(str(path))
    return img


def load_templates(name):
    """加载某个名称的所有模板图片（支持 name.png, name_2.png）"""
    templates = []
    main = load_template(name)
    if main is not None:
        templates.append(main)
    alt = TEMPLATES_DIR / f"{name}_2.png"
    if alt.exists():
        img = cv2.imread(str(alt))
        if img is not None:
            templates.append(img)
    return templates


def find_matches(screen, template, threshold=0.8):
    """在截屏中查找所有匹配位置，返回中心坐标列表"""
    if template is None or screen is None:
        return []
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    h, w = template.shape[:2]

    # 合并相近的匹配点（NMS）
    points = []
    for pt in zip(*locations[::-1]):
        cx, cy = pt[0] + w // 2, pt[1] + h // 2
        too_close = False
        for ex, ey in points:
            if abs(cx - ex) < w * 0.5 and abs(cy - ey) < h * 0.5:
                too_close = True
                break
        if not too_close:
            points.append((cx, cy))
    return points


def find_matches_multi(screen, templates, threshold=0.8):
    """用多个模板在截屏中查找匹配位置，合并去重后返回"""
    all_points = []
    for tpl in templates:
        pts = find_matches(screen, tpl, threshold=threshold)
        for cx, cy in pts:
            too_close = False
            for ex, ey in all_points:
                if abs(cx - ex) < 30 and abs(cy - ey) < 30:
                    too_close = True
                    break
            if not too_close:
                all_points.append((cx, cy))
    return all_points


# --- 模板采集 (--setup) ---
def setup_templates():
    """引导用户截取模板图片"""
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    print("\n=== 微信朋友圈点赞工具 - 模板采集 ===\n")
    print("请先打开微信朋友圈页面。\n")
    print("接下来会依次采集 3 个模板图片。")
    print("每次采集时，会启动 macOS 截图工具，请用鼠标框选对应的 UI 元素。\n")

    for name, desc in TEMPLATE_NAMES.items():
        target = TEMPLATES_DIR / f"{name}.png"
        if target.exists():
            resp = input(f"模板 [{desc}] 已存在，是否重新采集？(y/N): ").strip().lower()
            if resp != "y":
                print(f"  跳过 {name}")
                continue

        print(f"\n>>> 请截取: {desc}")
        print("    系统会打开截图工具，请框选该元素区域。")
        input("    准备好后按回车...")

        tmp_path = str(target)
        subprocess.run(["screencapture", "-i", tmp_path], check=True)

        if target.exists() and target.stat().st_size > 0:
            img = cv2.imread(tmp_path)
            if img is not None:
                h, w = img.shape[:2]
                print(f"  ✓ 已保存: {target} ({w}x{h})")
            else:
                print(f"  ✗ 图片读取失败，请重试")
        else:
            print(f"  ✗ 截图被取消，请重试")

    # 额外采集第二个已赞模板
    liked_alt = TEMPLATES_DIR / "liked_state_2.png"
    if liked_alt.exists():
        resp = input(f"\n第二个已赞模板已存在，是否重新采集？(y/N): ").strip().lower()
        if resp == "y":
            print("\n>>> 请截取: 第二个已赞状态标识 (不同样式的已赞图标)")
            print("    系统会打开截图工具，请框选该元素区域。")
            input("    准备好后按回车...")
            subprocess.run(["screencapture", "-i", str(liked_alt)], check=True)
            if liked_alt.exists() and liked_alt.stat().st_size > 0:
                img = cv2.imread(str(liked_alt))
                if img is not None:
                    h, w = img.shape[:2]
                    print(f"  ✓ 已保存: {liked_alt} ({w}x{h})")
                else:
                    print(f"  ✗ 图片读取失败，请重试")
            else:
                print(f"  ✗ 截图被取消")
        else:
            print(f"  跳过第二个已赞模板")
    else:
        resp = input(f"\n是否采集第二个已赞模板？可提高识别率 (y/N): ").strip().lower()
        if resp == "y":
            print("\n>>> 请截取: 第二个已赞状态标识 (不同样式的已赞图标)")
            print("    系统会打开截图工具，请框选该元素区域。")
            input("    准备好后按回车...")
            subprocess.run(["screencapture", "-i", str(liked_alt)], check=True)
            if liked_alt.exists() and liked_alt.stat().st_size > 0:
                img = cv2.imread(str(liked_alt))
                if img is not None:
                    h, w = img.shape[:2]
                    print(f"  ✓ 已保存: {liked_alt} ({w}x{h})")
                else:
                    print(f"  ✗ 图片读取失败，请重试")
            else:
                print(f"  ✗ 截图被取消")

    # 验证
    print("\n=== 模板检查 ===")
    all_ok = True
    for name, desc in TEMPLATE_NAMES.items():
        path = TEMPLATES_DIR / f"{name}.png"
        if path.exists():
            print(f"  ✓ {name}: {desc}")
        else:
            print(f"  ✗ {name}: 缺失 - {desc}")
            all_ok = False

    liked_alt_path = TEMPLATES_DIR / "liked_state_2.png"
    if liked_alt_path.exists():
        print(f"  ✓ liked_state_2: 第二个已赞状态标识")
    else:
        print(f"  - liked_state_2: 未采集 (可选)")

    if all_ok:
        print("\n模板采集完成！可以用 --dry-run 测试识别效果。\n")
    else:
        print("\n部分模板缺失，请重新运行 --setup 补充。\n")


# --- Retina 屏幕坐标转换 ---
def get_screen_scale():
    """获取 Retina 缩放比例"""
    screen = take_screenshot()
    # screencapture 输出的是实际像素，pyautogui 用的是逻辑坐标
    logical_w, _ = pyautogui.size()
    actual_w = screen.shape[1]
    return actual_w / logical_w


# --- 核心点赞逻辑 ---
def run_liker(count=15, like_rate=0.4, dry_run=False):
    """主流程"""
    log.info(f"启动点赞: count={count}, like_rate={like_rate}, dry_run={dry_run}")

    # 加载模板
    dots_tpl = load_template("dots_button")
    liked_tpls = load_templates("liked_state")
    cancel_tpl = load_template("cancel_button")

    if dots_tpl is None:
        log.error("缺少 dots_button 模板，请先运行 --setup")
        return

    scale = get_screen_scale()
    log.info(f"屏幕缩放比例: {scale}")

    liked_count = 0
    processed = 0
    max_scrolls = count * 3  # 防止无限滚动
    scroll_count = 0
    handled_y = set()  # 记录已处理过的帖子 Y 坐标（像素级）
    total_scrolled = 0  # 累计滚动的像素量

    while processed < count and scroll_count < max_scrolls:
        screen = take_screenshot()

        # 查找 "..." 按钮
        dots_matches = find_matches(screen, dots_tpl, threshold=0.75)
        log.info(f"找到 {len(dots_matches)} 个操作按钮")

        if not dots_matches:
            log.info("当前屏幕未找到操作按钮，向下滚动")
            scroll_amount = scroll_down(scale)
            total_scrolled += scroll_amount
            scroll_count += 1
            reading_pause()
            continue

        # 检查已赞状态，过滤掉已赞的
        liked_positions = []
        if liked_tpls:
            liked_positions = find_matches_multi(screen, liked_tpls, threshold=0.75)
            log.info(f"找到 {len(liked_positions)} 个已赞标识")

        for dx, dy in dots_matches:
            if processed >= count:
                break

            # 用绝对 Y 坐标（屏幕 Y + 累计滚动量）判断是否处理过
            abs_y = dy + total_scrolled
            already_handled = False
            for hy in handled_y:
                if abs(abs_y - hy) < 100:
                    already_handled = True
                    break
            if already_handled:
                log.info(f"  跳过已处理帖子 ({dx}, {dy})")
                continue

            # 检查附近是否已赞
            is_liked = False
            for lx, ly in liked_positions:
                if abs(dy - ly) < 60:  # 同一行附近
                    is_liked = True
                    break
            if is_liked:
                log.info(f"  跳过已赞帖子 ({dx}, {dy})")
                handled_y.add(abs_y)
                continue

            processed += 1
            handled_y.add(abs_y)

            # 随机决定是否点赞
            if random.random() > like_rate:
                log.info(f"  路过不赞 #{processed} ({dx}, {dy})")
                continue

            if dry_run:
                log.info(f"  [DRY-RUN] 会点赞 #{processed} ({dx}, {dy})")
                liked_count += 1
                continue

            # 执行点赞
            success = do_like(dx, dy, scale, cancel_tpl)
            if success:
                liked_count += 1
                log.info(f"  点赞成功 #{liked_count}")
            else:
                log.warning(f"  点赞失败，跳过")

            random_delay(3.0, 8.0)

        # 滚动到下一屏
        scroll_px = scroll_down(scale)
        total_scrolled += scroll_px
        scroll_count += 1
        reading_pause()

    log.info(f"完成: 处理 {processed} 条，点赞 {liked_count} 条")


# --- 执行单次点赞 ---
def do_like(dots_x, dots_y, scale, cancel_tpl):
    """点击 ... 按钮，检查是否已赞，未赞则点赞"""
    # 转换为逻辑坐标
    lx, ly = int(dots_x / scale), int(dots_y / scale)

    # 1. 点击 "..." 按钮
    log.info(f"  点击操作按钮 ({lx}, {ly})")
    human_click(lx, ly)
    time.sleep(random.uniform(1.0, 1.5))

    # 2. 截屏检查菜单内容
    screen2 = take_screenshot()

    # 3. 检查是否出现"取消"按钮（说明已赞过）
    if cancel_tpl is not None:
        cancel_matches = find_matches(screen2, cancel_tpl, threshold=0.65)
        if cancel_matches:
            log.info(f"  检测到取消按钮，该帖已赞过，关闭菜单跳过")
            pyautogui.press("escape")
            time.sleep(0.3)
            return False

    # 4. 直接点击赞按钮（在 ... 左边约 135 像素）
    like_x = lx - 135
    like_y = ly
    log.info(f"  点击赞按钮 ({like_x}, {like_y})")
    human_click(like_x, like_y)
    time.sleep(random.uniform(0.5, 1.0))

    return True


def scroll_down(scale):
    """随机距离向下滚动，返回估算的像素滚动量"""
    scroll_amount = random.randint(8, 15)  # pyautogui scroll units
    log.info(f"向下滚动 {scroll_amount} 格")
    pyautogui.scroll(-scroll_amount)
    time.sleep(random.uniform(0.3, 0.6))
    # 每个 scroll unit 大约对应 40 像素（Retina 下约 80 像素）
    return int(scroll_amount * 40 * scale)


# --- 入口 ---
def main():
    parser = argparse.ArgumentParser(description="微信朋友圈自动点赞工具")
    parser.add_argument("--setup", action="store_true", help="采集模板图片")
    parser.add_argument("--count", type=int, default=15, help="最多处理条数 (默认 15)")
    parser.add_argument("--like-rate", type=float, default=0.4, help="点赞概率 0.0-1.0 (默认 0.4)")
    parser.add_argument("--dry-run", action="store_true", help="只识别不点击，用于测试")
    args = parser.parse_args()

    # 确保目录存在
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if args.setup:
        setup_templates()
        return

    # 限制单次运行上限
    count = min(args.count, 20)
    like_rate = max(0.0, min(1.0, args.like_rate))

    print(f"\n微信朋友圈点赞工具")
    print(f"处理条数: {count}, 点赞概率: {like_rate}")
    if args.dry_run:
        print("*** DRY-RUN 模式：只识别不操作 ***")
    print(f"安全提示: 将鼠标快速移到屏幕左上角可紧急停止")
    print()

    input("请确保微信朋友圈已打开，按回车开始...")
    time.sleep(1.0)

    try:
        run_liker(count=count, like_rate=like_rate, dry_run=args.dry_run)
    except pyautogui.FailSafeException:
        log.warning("检测到鼠标移至左上角，紧急停止！")
    except KeyboardInterrupt:
        log.warning("用户中断 (Ctrl+C)")
    except Exception as e:
        log.error(f"运行出错: {e}", exc_info=True)


if __name__ == "__main__":
    main()
