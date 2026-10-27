import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

# ────────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="수중 사진 보정",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ────────────────────────────────────────────────
# CSS — 딥 네이비 테마
# ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans KR', sans-serif;
}

.stApp {
    background: linear-gradient(160deg, #04111f 0%, #062035 50%, #071a2e 100%);
    min-height: 100vh;
    color: #d6e8f5;
}

.hero-header {
    text-align: center;
    padding: 2.8rem 1rem 1.2rem;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #4fc3f7 0%, #81d4fa 50%, #b3e5fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
    margin-bottom: 0.35rem;
}
.hero-sub {
    font-size: 1.0rem;
    font-weight: 300;
    color: #7baecf;
    letter-spacing: 0.03em;
}

.stButton > button {
    background: linear-gradient(135deg, #0277bd 0%, #0288d1 50%, #039be5 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2.5rem !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    font-family: 'Noto Sans KR', sans-serif !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 4px 20px rgba(2,119,189,0.4) !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #01579b 0%, #0277bd 50%, #0288d1 100%) !important;
    box-shadow: 0 6px 28px rgba(2,119,189,0.6) !important;
    transform: translateY(-1px) !important;
}

.stDownloadButton > button {
    background: linear-gradient(135deg, #00695c 0%, #00796b 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1.0rem !important;
    font-weight: 600 !important;
    font-family: 'Noto Sans KR', sans-serif !important;
    width: 100% !important;
    padding: 0.75rem !important;
    box-shadow: 0 3px 14px rgba(0,105,92,0.4) !important;
}
.stDownloadButton > button:hover {
    box-shadow: 0 5px 20px rgba(0,105,92,0.6) !important;
    transform: translateY(-1px) !important;
}

.img-label {
    font-size: 0.82rem;
    font-weight: 500;
    color: #7baecf;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    text-align: center;
    margin-bottom: 0.5rem;
}

.analysis-badge {
    background: rgba(2,119,189,0.18);
    border: 1px solid rgba(79,195,247,0.25);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    font-size: 0.88rem;
    color: #7baecf;
    line-height: 1.7;
    margin: 1rem 0;
}
.analysis-badge strong {
    color: #4fc3f7;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1.5px dashed rgba(79,195,247,0.3) !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploader"] label {
    color: #7baecf !important;
}
.stImage img {
    border-radius: 12px;
    border: 1px solid rgba(79,195,247,0.15);
}
footer, header { visibility: hidden; }
#MainMenu { visibility: hidden; }
.block-container { padding-top: 0 !important; }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────

def safe_resize(pil_img, max_px=2400):
    w, h = pil_img.size
    if max(w, h) <= max_px:
        return pil_img
    scale = max_px / max(w, h)
    return pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def pil_to_bytes(pil_img, fmt="JPEG", quality=95):
    buf = io.BytesIO()
    if fmt == "JPEG":
        pil_img.convert("RGB").save(buf, format="JPEG", quality=quality, subsampling=0)
    else:
        pil_img.save(buf, format="PNG")
    return buf.getvalue()


def apply_curves(channel, points):
    if len(points) < 2:
        return channel
    pts = sorted(points, key=lambda p: p[0])
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        if i <= pts[0][0]:
            lut[i] = pts[0][1]
        elif i >= pts[-1][0]:
            lut[i] = pts[-1][1]
        else:
            for j in range(len(pts) - 1):
                x0, y0 = pts[j]
                x1, y1 = pts[j + 1]
                if x0 <= i <= x1:
                    t = (i - x0) / max(x1 - x0, 1)
                    lut[i] = y0 + t * (y1 - y0)
                    break
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return cv2.LUT(channel, lut)


def analyze_image(bgr):
    img_f = bgr.astype(np.float32) / 255.0
    r = img_f[:, :, 2]
    g = img_f[:, :, 1]
    b = img_f[:, :, 0]
    mean_r = float(r.mean())
    mean_g = float(g.mean())
    mean_b = float(b.mean())
    mean_lum = float((0.299 * r + 0.587 * g + 0.114 * b).mean())
    red_deficit = mean_g - mean_r
    rg_ratio = mean_r / max(mean_g, 1e-6)
    bg_ratio = mean_b / max(mean_g, 1e-6)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    lum_img = (0.299 * r + 0.587 * g + 0.114 * b)
    mid_mask = (lum_img > 0.30) & (lum_img < 0.70)
    if mid_mask.sum() > 100:
        mid_r = float(r[mid_mask].mean())
        mid_g = float(g[mid_mask].mean())
    else:
        mid_r, mid_g = mean_r, mean_g
    target_rg = 0.80
    r_correction = min(target_rg / max(rg_ratio, 0.05), 2.0)
    target_bg = 1.00
    b_correction = min(max(target_bg / max(bg_ratio, 0.1), 0.90), 1.30)
    mid_green_excess = mid_g - mid_r
    g_suppress = max(0.90, 1.0 - min(mid_green_excess * 0.6, 0.10))
    exposure_gain = min(max(0.48 / max(mean_lum, 0.08), 0.88), 1.30)
    sat_factor = 0.85 if mean_g > 0.48 else 0.92
    veil_strength = max(0, (mean_g + mean_b) / 2 - mean_r)
    defog = min(max(0.15 + veil_strength * 0.8, 0.10), 0.45)
    sharpen_amount = min(max(0.25 + (1.0 - min(lap_var / 400.0, 1.0)) * 0.30, 0.20), 0.55)
    return {
        "r_correction": round(r_correction, 3),
        "g_suppress": round(g_suppress, 3),
        "b_correction": round(b_correction, 3),
        "exposure_gain": round(exposure_gain, 3),
        "sat_factor": round(sat_factor, 3),
        "defog": round(defog, 3),
        "sharpen_amount": round(sharpen_amount, 3),
        "_rg_ratio": round(rg_ratio, 3),
        "_bg_ratio": round(bg_ratio, 3),
        "_mean_lum": round(mean_lum, 3),
        "_red_deficit": round(red_deficit, 3),
        "_lap_var": round(lap_var, 1),
    }


def dark_channel_defog(bgr, strength=0.40):
    img = bgr.astype(np.float32) / 255.0
    dark = cv2.erode(img.min(axis=2), np.ones((15, 15), np.uint8))
    num_px = max(int(dark.size * 0.001), 1)
    flat = dark.ravel()
    indices = np.argpartition(flat, -num_px)[-num_px:]
    atm_mask = np.zeros_like(flat, dtype=bool)
    atm_mask[indices] = True
    atm_mask = atm_mask.reshape(dark.shape)
    A = float(img[atm_mask].mean())
    A = max(A, 0.55)
    t = 1.0 - strength * dark / max(A, 1e-6)
    t = np.clip(t, 0.30, 1.0)
    t3 = t[:, :, np.newaxis]
    result = (img - A) / t3 + A
    return np.clip(result * 255.0, 0, 255).astype(np.uint8)


def enhance_underwater(pil_img, params):
    p = params
    bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    img_f = bgr.astype(np.float32)
    img_f[:, :, 2] = np.clip(img_f[:, :, 2] * p["r_correction"], 0, 255)
    img_f[:, :, 1] = np.clip(img_f[:, :, 1] * p["g_suppress"], 0, 255)
    img_f[:, :, 0] = np.clip(img_f[:, :, 0] * p["b_correction"], 0, 255)
    bgr = img_f.astype(np.uint8)
    bgr = dark_channel_defog(bgr, p["defog"])
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a_ch, b_ch = lab[:, :, 1], lab[:, :, 2]
    a_shift = (128.0 - float(a_ch.mean())) * 0.40
    b_shift = (128.0 - float(b_ch.mean())) * 0.20
    lab[:, :, 1] = np.clip(a_ch + a_shift, 0, 255)
    lab[:, :, 2] = np.clip(b_ch + b_shift, 0, 255)
    bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    img_f = bgr.astype(np.float32) / 255.0
    img_f = img_f * p["exposure_gain"]
    hi = img_f > 0.88
    img_f[hi] = 0.88 + (img_f[hi] - 0.88) * 0.30
    bgr = (np.clip(img_f, 0, 1) * 255).astype(np.uint8)
    lab2 = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a2, cb2 = cv2.split(lab2)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    bgr = cv2.cvtColor(cv2.merge([clahe.apply(l), a2, cb2]), cv2.COLOR_LAB2BGR)
    amount = p["sharpen_amount"]
    detail_blur = cv2.GaussianBlur(bgr, (0, 0), 0.8)
    sharp = cv2.addWeighted(bgr, 1.0 + amount, detail_blur, -amount, 0)
    bgr = cv2.addWeighted(sharp, 0.70, bgr, 0.30, 0)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * p["sat_factor"], 0, 255)
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    hsv_f = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue, sat, val = hsv_f[:, :, 0], hsv_f[:, :, 1], hsv_f[:, :, 2]
    skin = ((hue < 18) | (hue > 162)) & (sat > 25) & (val > 55)
    hsv_f[:, :, 1][skin] = np.clip(hsv_f[:, :, 1][skin] * 1.10, 0, 255)
    hsv_f[:, :, 2][skin] = np.clip(hsv_f[:, :, 2][skin] * 1.05, 0, 255)
    bgr = cv2.cvtColor(hsv_f.astype(np.uint8), cv2.COLOR_HSV2BGR)
    b_ch2, g_ch, r_ch = cv2.split(bgr)
    r_ch = apply_curves(r_ch, [(0, 2), (128, 128), (255, 255)])
    g_ch = apply_curves(g_ch, [(0, 2), (128, 126), (255, 250)])
    b_ch2 = apply_curves(b_ch2, [(0, 5), (128, 133), (255, 255)])
    bgr = cv2.merge([b_ch2, g_ch, r_ch])
    bgr = cv2.bilateralFilter(bgr, d=5, sigmaColor=20, sigmaSpace=5)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


# ────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
  <div class="hero-title">🌊 수중 사진 자동 보정</div>
  <div class="hero-sub">업로드 → 보정 실행 한 번으로 전문가급 결과물</div>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "사진을 선택하세요 (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible",
)

if uploaded:
    pil_orig = Image.open(uploaded).convert("RGB")
    pil_resized = safe_resize(pil_orig, max_px=2400)

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, _ = st.columns([2, 3])
    with col_btn:
        run = st.button("🎨 보정 실행", use_container_width=True)

    if run:
        with st.spinner("이미지 분석 중…"):
            bgr_for_analysis = cv2.cvtColor(
                np.array(pil_resized), cv2.COLOR_RGB2BGR
            )
            params = analyze_image(bgr_for_analysis)

        with st.spinner("전문가 보정 적용 중…"):
            pil_result = enhance_underwater(pil_resized, params)

        rg = params["_rg_ratio"]
        deficit = params["_red_deficit"]
        depth_label = (
            "얕은 수심 (5m 미만)" if deficit < 0.06
            else "중간 수심 (5–15m)" if deficit < 0.15
            else "깊은 수심 (15m+)"
        )
        lap = params["_lap_var"]
        clarity_label = (
            "선명" if lap > 200 else "약간 흐릿함" if lap > 60 else "많이 흐릿함"
        )

        st.markdown(f"""
        <div class="analysis-badge">
          📊 <strong>자동 분석 결과</strong> —
          {depth_label} · {clarity_label}<br>
          🔴 적색 복원 {params['r_correction']:.2f}x ·
          🟢 색조 중화 {int((1-params['g_suppress'])*100)}% ·
          💧 디포그 {int(params['defog']*100)}% ·
          ✨ 채도 조정 {int((params['sat_factor']-1)*100):+d}%
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="img-label">BEFORE — 원본</div>', unsafe_allow_html=True)
            st.image(pil_resized, use_container_width=True)
        with c2:
            st.markdown('<div class="img-label">AFTER — 보정 완료</div>', unsafe_allow_html=True)
            st.image(pil_result, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        dc1, dc2, _ = st.columns([2, 2, 3])
        stem = uploaded.name.rsplit(".", 1)[0]
        with dc1:
            st.download_button(
                "⬇️ JPG 다운로드",
                data=pil_to_bytes(pil_result, "JPEG", 95),
                file_name=f"{stem}_enhanced.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )
        with dc2:
            st.download_button(
                "⬇️ PNG 다운로드",
                data=pil_to_bytes(pil_result, "PNG"),
                file_name=f"{stem}_enhanced.png",
                mime="image/png",
                use_container_width=True,
            )
