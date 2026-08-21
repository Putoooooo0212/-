"""
逆向小卡PSD生成器 - rembg 版
"""
import io, os, uuid, base64, subprocess, json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TMP_DIR = os.path.join(BASE_DIR, "_tmp")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

app = FastAPI(title="逆向小卡PSD生成器")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


def fit_image_to_card(img: Image.Image) -> Image.Image:
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


def get_subject_mask(img_rgba: Image.Image) -> Image.Image:
    """用 rembg 抠图"""
    img_rgb = img_rgba.convert("RGB")
    result = remove(img_rgb)
    alpha = np.array(result.split()[3])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=2)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    return Image.fromarray(alpha, mode="L")


def get_background_mask(subject_mask: Image.Image) -> Image.Image:
    arr = 255 - np.array(subject_mask)
    return Image.fromarray(arr, mode="L")


def get_font(size: int):
    for fp in ["/Library/Fonts/Courier New.ttf", "/System/Library/Fonts/Courier.dfont"]:
        if os.path.exists(fp):
            try: return ImageFont.truetype(fp, size)
            except: continue
    return ImageFont.load_default()


def make_text_mask_top(text: str, y_range: tuple, font_size: int = 38) -> Image.Image:
    """顶部文字：镂空描边效果蒙版（白色=保留，黑色=镂空）"""
    mask = Image.new("L", (CARD_W, CARD_H), 0)
    if not text.strip(): return mask
    draw = ImageDraw.Draw(mask)
    font = get_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (CARD_W - tw) // 2
    y = (y_range[0] + y_range[1]) // 2 - th // 2
    stroke_width = 6
    # 背景白色（保留）
    draw.rectangle([0, 0, CARD_W, CARD_H], fill=255)
    # 创建文字蒙版
    temp = Image.new("L", (CARD_W, CARD_H), 0)
    temp_draw = ImageDraw.Draw(temp)
    temp_draw.text((x, y), text, fill=255, font=font)
    temp_arr = np.array(temp)
    # 膨胀 = 描边外圈
    kernel_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stroke_width*2, stroke_width*2))
    dilated = cv2.dilate(temp_arr, kernel_outer, iterations=1)
    # 腐蚀 = 描边内圈
    kernel_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stroke_width, stroke_width))
    eroded = cv2.erode(dilated, kernel_inner, iterations=1)
    # 描边区域 = 膨胀 - 腐蚀（白色=保留）
    stroke_area = dilated - eroded
    # 最终蒙版：描边区域白色，其他区域黑色（镂空）
    final = np.zeros_like(dilated)
    final[stroke_area > 0] = 255
    return Image.fromarray(final, mode="L")


def make_text_mask_bottom(text: str, y_range: tuple, font_size: int = 30) -> Image.Image:
    """底部文字：普通镂空（不加描边效果）"""
    mask = Image.new("L", (CARD_W, CARD_H), 0)
    if not text.strip(): return mask
    draw = ImageDraw.Draw(mask)
    font = get_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (CARD_W - tw) // 2
    y = (y_range[0] + y_range[1]) // 2 - th // 2
    draw.text((x, y), text, fill=255, font=font)
    return Image.fromarray(np.array(mask), mode="L")


def apply_text_cutout(img: Image.Image, tm: Image.Image) -> Image.Image:
    r, g, b, a = img.split()
    a_arr = np.array(a)
    a_arr[np.array(tm) > 128] = 0
    return Image.merge("RGBA", (r, g, b, Image.fromarray(a_arr, mode="L")))


def build_color_layer(mask: Image.Image, color: tuple) -> Image.Image:
    r, g, b = color
    arr = np.zeros((CARD_H, CARD_W, 4), dtype=np.uint8)
    arr[:, :, 0] = r; arr[:, :, 1] = g; arr[:, :, 2] = b
    arr[:, :, 3] = np.array(mask)
    return Image.fromarray(arr, mode="RGBA")


def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def make_text_mask_combined(top_text, bottom_text):
    tm1 = make_text_mask_top(top_text, TOP_TEXT_Y, font_size=38)
    tm2 = make_text_mask_bottom(bottom_text, BOTTOM_TEXT_Y, font_size=30)
    return Image.fromarray(np.maximum(np.array(tm1), np.array(tm2)), mode="L")


@app.get("/")
async def index():
    return HTMLResponse(open(os.path.join(BASE_DIR, "static", "index.html")).read())
@app.post("/preview")
async def preview(image: UploadFile = File(...), top_text: str = Form(""), bottom_text: str = Form("")):
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes))
    card = fit_image_to_card(img)

    try:
        subject = get_subject_mask(img.convert("RGBA"))
    except Exception as e:
        print(f"rembg 失败: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    subject = subject.resize((CARD_W, CARD_H), Image.LANCZOS)
    background = get_background_mask(subject)
    tm = make_text_mask_combined(top_text, bottom_text)

    # 印刷层：照片 + 文字镂空
    print_prev = apply_text_cutout(card.copy(), tm)

    # 白墨层：人物蒙版 + 文字镂空
    wl = build_color_layer(subject, WHITE_INK_COLOR)
    wl = apply_text_cutout(wl, tm)

    # 逆向层：背景蒙版 + 文字镂空
    rl = build_color_layer(background, REVERSE_COLOR)
    rl = apply_text_cutout(rl, tm)

    return {"print": img_to_b64(print_prev), "white": img_to_b64(wl), "reverse": img_to_b64(rl)}


@app.post("/generate")
async def generate(image: UploadFile = File(...), top_text: str = Form(""), bottom_text: str = Form("")):
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes))
    card = fit_image_to_card(img)

    try:
        subject = get_subject_mask(img.convert("RGBA"))
    except Exception as e:
        print(f"rembg 失败，使用 fallback: {e}")
        img_cv = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        img_cv = cv2.resize(img_cv, (CARD_W, CARD_H))
        mask = np.zeros((CARD_H, CARD_W), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        rect = (50, 50, CARD_W - 100, CARD_H - 100)
        cv2.grabCut(img_cv, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
        subject = Image.fromarray(mask, mode='L')

    subject = subject.resize((CARD_W, CARD_H), Image.LANCZOS)
    background = get_background_mask(subject)
    tm = make_text_mask_combined(top_text, bottom_text)

    pl = apply_text_cutout(card.copy(), tm)
    wl = apply_text_cutout(build_color_layer(subject, WHITE_INK_COLOR), tm)
    rl = apply_text_cutout(build_color_layer(background, REVERSE_COLOR), tm)

    try:
        path = create_psd(pl, wl, rl)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    return FileResponse(path, filename=f"逆向小卡_{uuid.uuid4().hex[:6]}.psd")


def create_psd(print_layer, white_layer, reverse_layer) -> str:
    uid = uuid.uuid4().hex[:8]
    out = os.path.join(OUTPUT_DIR, f"逆向小卡_{uid}.psd")
    print_layer.save(os.path.join(TMP_DIR, f"{uid}_p.png"))
    white_layer.save(os.path.join(TMP_DIR, f"{uid}_w.png"))
    reverse_layer.save(os.path.join(TMP_DIR, f"{uid}_r.png"))
    params = json.dumps({"uid": uid, "tmpDir": TMP_DIR, "output": out, "w": CARD_W, "h": CARD_H})
    js = f"""
const fs=require('fs'),path=require('path'),{{writePsd}}=require('ag-psd'),sharp=require('sharp');
async function load(p){{const{{data,info}}=await sharp(p).ensureAlpha().raw().toBuffer({{resolveWithObject:true}});return{{data:new Uint8ClampedArray(data.buffer,data.byteOffset,data.byteLength),width:info.width,height:info.height}}}}
async function main(){{
  const P=JSON.parse(process.argv[2]);
  const[p,w,r]=await Promise.all([load(path.join(P.tmpDir,P.uid+'_p.png')),load(path.join(P.tmpDir,P.uid+'_w.png')),load(path.join(P.tmpDir,P.uid+'_r.png'))]);
  const bg=new Uint8ClampedArray(P.w*P.h*4).fill(255);
  const psd={{width:P.w,height:P.h,children:[
    {{name:'\\u80cc\\u666f',left:0,top:0,hidden:true,imageData:{{data:bg,width:P.w,height:P.h}}}},
    {{name:'\\u5370\\u5237\\u5c42',left:0,top:0,imageData:p}},
    {{name:'\\u767d\\u58a8\\u5c42',left:0,top:0,imageData:w}},
    {{name:'\\u9006\\u5411\\u5c42',left:0,top:0,imageData:r}}
  ]}};
  fs.writeFileSync(P.output,Buffer.from(writePsd(psd,{{generateThumbnail:false}})));
  ['_p.png','_w.png','_r.png'].forEach(s=>{{try{{fs.unlinkSync(path.join(P.tmpDir,P.uid+s))}}catch(e){{}}}});
  console.log('OK');
}}main().catch(e=>{{console.error(e.message);process.exit(1)}});
"""
    js_path = os.path.join(TMP_DIR, f"{uid}.js")
    with open(js_path, "w") as f:
        f.write(js)
    result = subprocess.run(["node", js_path, params], capture_output=True, text=True, cwd=BASE_DIR, timeout=30)
    try: os.remove(js_path)
    except: pass
    if result.returncode != 0:
        return JSONResponse(status_code=500, content={"error": result.stderr.strip()})
    return FileResponse(out, filename=f"逆向小卡_{uid[:6]}.psd")


@app.post("/assemble_psd")
async def assemble_psd(
    print: UploadFile = File(...),
    white: UploadFile = File(...),
    reverse: UploadFile = File(...),
):
    uid = uuid.uuid4().hex[:8]
    out = os.path.join(OUTPUT_DIR, f"逆向小卡_{uid}.psd")

    for name, upload in [("p", print), ("w", white), ("r", reverse)]:
        data = await upload.read()
        with open(os.path.join(TMP_DIR, f"{uid}_{name}.png"), "wb") as f:
            f.write(data)

    params = json.dumps({"uid": uid, "tmpDir": TMP_DIR, "output": out, "w": CARD_W, "h": CARD_H})
    js = f"""
const fs=require('fs'),path=require('path'),{{writePsd}}=require('ag-psd'),sharp=require('sharp');
async function load(p){{const{{data,info}}=await sharp(p).ensureAlpha().raw().toBuffer({{resolveWithObject:true}});return{{data:new Uint8ClampedArray(data.buffer,data.byteOffset,data.byteLength),width:info.width,height:info.height}}}}
async function main(){{
  const P=JSON.parse(process.argv[2]);
  const[p,w,r]=await Promise.all([load(path.join(P.tmpDir,P.uid+'_p.png')),load(path.join(P.tmpDir,P.uid+'_w.png')),load(path.join(P.tmpDir,P.uid+'_r.png'))]);
  const bg=new Uint8ClampedArray(P.w*P.h*4).fill(255);
  const psd={{width:P.w,height:P.h,children:[
    {{name:'\\u80cc\\u666f',left:0,top:0,hidden:true,imageData:{{data:bg,width:P.w,height:P.h}}}},
    {{name:'\\u5370\\u5237\\u5c42',left:0,top:0,imageData:p}},
    {{name:'\\u767d\\u58a8\\u5c42',left:0,top:0,imageData:w}},
    {{name:'\\u9006\\u5411\\u5c42',left:0,top:0,imageData:r}}
  ]}};
  fs.writeFileSync(P.output,Buffer.from(writePsd(psd,{{generateThumbnail:false}})));
  ['_p.png','_w.png','_r.png'].forEach(s=>{{try{{fs.unlinkSync(path.join(P.tmpDir,P.uid+s))}}catch(e){{}}}});
  console.log('OK');
}}main().catch(e=>{{console.error(e.message);process.exit(1)}});
"""
    js_path = os.path.join(TMP_DIR, f"{uid}.js")
    with open(js_path, "w") as f:
        f.write(js)
    result = subprocess.run(["node", js_path, params], capture_output=True, text=True, cwd=BASE_DIR, timeout=30)
    try: os.remove(js_path)
    except: pass
    if result.returncode != 0:
        return JSONResponse(status_code=500, content={"error": result.stderr.strip()})
    return FileResponse(out, filename=f"逆向小卡_{uid[:6]}.psd")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7777))
    uvicorn.run(app, host="0.0.0.0", port=port)


@app.post("/export_png")
async def export_png(image: UploadFile = File(...), top_text: str = Form(""), bottom_text: str = Form("")):
    """导出三层PNG"""
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes))
    card = fit_image_to_card(img)

    try:
        subject = get_subject_mask(img.convert("RGBA"))
    except Exception as e:
        print(f"rembg 失败: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    subject = subject.resize((CARD_W, CARD_H), Image.LANCZOS)
    background = get_background_mask(subject)
    tm = make_text_mask_combined(top_text, bottom_text)

    pl = apply_text_cutout(card.copy(), tm)
    wl = apply_text_cutout(build_color_layer(subject, WHITE_INK_COLOR), tm)
    rl = apply_text_cutout(build_color_layer(background, REVERSE_COLOR), tm)

    return {
        "print": img_to_b64(pl),
        "white": img_to_b64(wl),
        "reverse": img_to_b64(rl),
    }
