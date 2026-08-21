#!/usr/bin/env python3
"""逆向小卡分层PNG生成器 - rembg 版"""
import sys, os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

os.environ['NUMBA_DISABLE_JIT'] = '1'

from pathlib import Path
Path.home = lambda: Path('/tmp')
import os.path
original_expanduser = os.path.expanduser
def patched_expanduser(path):
    if path.startswith('~'):
        return path.replace('~', '/tmp')
    return original_expanduser(path)
os.path.expanduser = patched_expanduser

from rembg import remove

CARD_W, CARD_H = 791, 1098
TOP_TEXT_Y = (55, 110)
BOTTOM_TEXT_Y = (950, 1030)
WHITE_INK_COLOR = (5, 1, 1)
REVERSE_COLOR = (231, 27, 26)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

def fit_image_to_card(img):
    w, h = img.size
    ratio = CARD_W / CARD_H
    if w / h > ratio:
        nh = CARD_H; nw = int(h * ratio)
        img = img.resize((nw, nh), Image.LANCZOS)
        l = (nw - CARD_W) // 2
        img = img.crop((l, 0, l + CARD_W, CARD_H))
    else:
        nw = CARD_W; nh = int(w / ratio)
        img = img.resize((nw, nh), Image.LANCZOS)
        t = (nh - CARD_H) // 2
        img = img.crop((0, t, CARD_W, t + CARD_H))
    return img.convert("RGBA")

def get_subject_mask_rembg(img_rgba):
    """用 rembg 抠图"""
    img_rgb = img_rgba.convert("RGB")
    result = remove(img_rgb)
    # 提取 alpha 通道作为蒙版
    alpha = np.array(result.split()[3])
    # 形态学优化
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=2)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    return Image.fromarray(alpha, mode="L")

def get_background_mask(subject_mask):
    arr = 255 - np.array(subject_mask)
    return Image.fromarray(arr, mode="L")

def get_font(size):
    for fp in ["/Library/Fonts/Courier New.ttf", "/System/Library/Fonts/Courier.dfont"]:
        if os.path.exists(fp):
            try: return ImageFont.truetype(fp, size)
            except: continue
    return ImageFont.load_default()

def make_text_mask(text, y_range, font_size=36):
    mask = Image.new("L", (CARD_W, CARD_H), 0)
    if not text.strip(): return mask
    draw = ImageDraw.Draw(mask)
    font = get_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (CARD_W - tw) // 2
    y = (y_range[0] + y_range[1]) // 2 - th // 2
    draw.text((x, y), text, fill=255, font=font)
    arr = cv2.dilate(np.array(mask), np.ones((3, 3), np.uint8), iterations=3)
    return Image.fromarray(arr, mode="L")

def apply_text_cutout(img, tm):
    r, g, b, a = img.split()
    a_arr = np.array(a)
    a_arr[np.array(tm) > 128] = 0
    return Image.merge("RGBA", (r, g, b, Image.fromarray(a_arr, mode="L")))

def build_color_layer(mask, color):
    r, g, b = color
    arr = np.zeros((CARD_H, CARD_W, 4), dtype=np.uint8)
    arr[:, :, 0] = r; arr[:, :, 1] = g; arr[:, :, 2] = b
    arr[:, :, 3] = np.array(mask)
    return Image.fromarray(arr, mode="RGBA")

def main():
    if len(sys.argv) < 2:
        print("用法: python3 generate_layers_rembg.py 图片路径 [顶部文字] [底部文字]")
        sys.exit(1)
    
    img_path = sys.argv[1]
    top_text = sys.argv[2] if len(sys.argv) > 2 else "2026PROMOTIONALCARD"
    bottom_text = sys.argv[3] if len(sys.argv) > 3 else "2026-SSSR-KWIN"
    
    print(f"读取图片: {img_path}")
    img = Image.open(img_path)
    card = fit_image_to_card(img)
    
    print("抠图中 (rembg AI 模型)...")
    subject_mask = get_subject_mask_rembg(card)
    background_mask = get_background_mask(subject_mask)
    
    tm1 = make_text_mask(top_text, TOP_TEXT_Y, 38)
    tm2 = make_text_mask(bottom_text, BOTTOM_TEXT_Y, 30)
    text_mask = Image.fromarray(np.maximum(np.array(tm1), np.array(tm2)), mode="L")
    
    print_layer = apply_text_cutout(card.copy(), text_mask)
    white_layer = apply_text_cutout(build_color_layer(subject_mask, WHITE_INK_COLOR), text_mask)
    reverse_layer = apply_text_cutout(build_color_layer(background_mask, REVERSE_COLOR), text_mask)
    
    base = os.path.splitext(os.path.basename(img_path))[0]
    p1 = os.path.join(OUT_DIR, f"{base}_印刷层.png")
    p2 = os.path.join(OUT_DIR, f"{base}_白墨层.png")
    p3 = os.path.join(OUT_DIR, f"{base}_逆向层.png")
    
    print_layer.save(p1)
    white_layer.save(p2)
    reverse_layer.save(p3)
    
    print(f"\n✅ 已生成:")
    print(f"  {p1}")
    print(f"  {p2}")
    print(f"  {p3}")

if __name__ == "__main__":
    main()
