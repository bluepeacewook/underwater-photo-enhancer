"""
수중 사진 자동 보정 시스템 (Underwater Photo Auto-Enhancer)
대상: 스쿠버/프리다이빙 강사 및 교육생
핵심: 픽셀 구조 100% 유지 + 수중 컬러 그레이딩 알고리즘
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import base64

# ────────────────────────────────────────────────────────────
# 페이지 기본 설정
# ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Underwater Auto-Enhancer",
    page_icon="🤿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ────────────────────────────────────────────────────────────
# CSS 스타일링
# ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Google Fonts */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  /* 전역 리셋 */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* 배경 – 수심을 연상시키는 딥 네이비 그라데이션 */
  .stApp {
    background: linear-gradient(160deg, #04111f 0%, #062035 45%, #04111f 100%);
    color: #e8f4fd;
  }

  /* 메인 컨테이너 패딩 */
  .block-container {
    padding: 2rem 3rem 4rem !important;
    max-width: 1280px;
  }

  /* ── 헤더 ── */
  .hero-header {
    text-align: center;
    padding: 2.8rem 0 1.6rem;
    margin-bottom: 0.5rem;
  }
  .hero-badge {
    display: inline-block;
    background: rgba(0,168,230,0.15);
    border: 1px solid rgba(0,168,230,0.4);
    color: #4dc8f0;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    padding: 0.28rem 0.8rem;
    border-radius: 100px;
    margin-bottom: 1.1rem;
  }
  .hero-title {
    font-size: clamp(2rem, 4vw, 3.1rem);
    font-weight: 700;
    color: #ffffff;
    line-height: 1.15;
    margin: 0 0 0.6rem;
    letter-spacing: -0.02em;
  }
  .hero-title span {
    color: #00c8f8;
  }
  .hero-sub {
    font-size: 0.95rem;
    color: #7ab8d4;
    font-weight: 400;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.6;
  }

  /* ── 업로드 영역 ── */
  .upload-zone {
    background: rgba(255,255,255,0.03);
    border: 1.5px dashed rgba(0,200,248,0.3);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    transition: border-color 0.2s;
    margin-bottom: 1.2rem;
  }
  .upload-zone:hover {
    border-color: rgba(0,200,248,0.6);
  }

  /* Streamlit 파일업로더 커스텀 */
  [data-testid="stFileUploader"] {
    background: transparent !important;
  }
  [data-testid="stFileUploadDropzone"] {
    background: rgba(0,200,248,0.04) !important;
    border: 1.5px dashed rgba(0,200,248,0.35) !important;
    border-radius: 14px !important;
    padding: 2.2rem 1rem !important;
  }
  [data-testid="stFileUploadDropzone"]:hover {
    border-color: rgba(0,200,248,0.7) !important;
    background: rgba(0,200,248,0.08) !important;
  }
  [data-testid="stFileUploadDropzone"] p {
    color: #7ab8d4 !important;
  }

  /* ── 이미지 패널 ── */
  .img-panel {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.2rem 1.2rem 1rem;
  }
  .img-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    color: #4dc8f0;
    margin-bottom: 0.7rem;
    text-transform: uppercase;
  }

  /* ── 슬라이더 레이블 ── */
  .param-label {
    font-size: 0.8rem;
    color: #7ab8d4;
    font-weight: 500;
    margin-bottom: 0.1rem;
  }

  /* ── Streamlit 슬라이더 틱 색상 ── */
  [data-testid="stSlider"] .rc-slider-track { background: #00c8f8 !important; }
  [data-testid="stSlider"] .rc-slider-handle { border-color: #00c8f8 !important; }

  /* ── 버튼 ── */
  .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #0094c6 0%, #00c8f8 100%) !important;
    color: #04111f !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.78rem 1.5rem !important;
    letter-spacing: 0.01em !important;
    transition: opacity 0.15s !important;
    box-shadow: 0 4px 20px rgba(0,200,248,0.25) !important;
  }
  .stButton > button:hover {
    opacity: 0.88 !important;
  }

  /* 다운로드 버튼 */
  .stDownloadButton > button {
    width: 100%;
    background: rgba(0,200,248,0.1) !important;
    color: #00c8f8 !important;
    border: 1.5px solid rgba(0,200,248,0.45) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.7rem 1.5rem !important;
  }
  .stDownloadButton > button:hover {
    background: rgba(0,200,248,0.18) !important;
  }

  /* ── 정보 칩 ── */
  .info-chips {
    display: flex;
    gap: 0.55rem;
    flex-wrap: wrap;
    margin-top: 0.8rem;
  }
  .chip {
    background: rgba(0,200,248,0.1);
    border: 1px solid rgba(0,200,248,0.25);
    border-radius: 100px;
    padding: 0.25rem 0.75rem;
    font-size: 0.72rem;
    color: #7ab8d4;
    white-space: nowrap;
  }

  /* ── 구분선 ── */
  hr {
    border-color: rgba(255,255,255,0.07) !important;
    margin: 2rem 0 !important;
  }

  /* ── 섹션 타이틀 ── */
  .section-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: #4dc8f0;
    text-transform: uppercase;
    margin-bottom: 1rem;
  }

  /* ── 알고리즘 설명 박스 ── */
  .algo-box {
    background: rgba(0,168,230,0.06);
    border-left: 3px solid #00c8f8;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    font-size: 0.83rem;
    color: #7ab8d4;
    line-height: 1.65;
  }

  /* expander 스타일 */
  [data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
  }

  /* 스크롤바 */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(0,200,248,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  핵심 알고리즘: 수중 자동 컬러 그레이딩
# ════════════════════════════════════════════════════════════

def enhance_underwater(
    pil_img: Image.Image,
    red_boost: float = 1.4,
    green_suppress: float = 0.88,
    blue_suppress: float = 0.92,
    defog_strength: float = 0.55,
    clahe_clip: float = 2.8,
    saturation_boost: float = 1.18,
    warmth: float = 0.08,
) -> Image.Image:
    """
    수중 사진 자동 보정 파이프라인
    ─────────────────────────────────────────────────────────
    STEP 1. 채널 선택적 레벨 보상
    STEP 2. 그레이 월드 & 화이트발란스
    STEP 3. CLAHE
    STEP 4. 언샤프 마스크
    STEP 5. 채도 미세 조정
    STEP 6. 워밍 커브
    """
    img_rgb = np.array(pil_img.convert("RGB"), dtype=np.float32)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    b, g, r = cv2.split(img_bgr)

    def channel_boost(ch, factor):
        mn, mx = ch.min(), ch.max()
        if mx - mn < 1e-3:
            return ch
        normalized = (ch - mn) / (mx - mn)
        boosted = normalized * factor
        return np.clip(boosted * 255.0, 0, 255)

    r = channel_boost(r, red_boost)
    g_mean = np.mean(g)
    b_mean = np.mean(b)
    global_mean = (np.mean(r) + g_mean + b_mean) / 3.0
    if g_mean > 1e-3:
        g = g * ((global_mean / g_mean) * green_suppress)
    if b_mean > 1e-3:
        b = b * ((global_mean / b_mean) * blue_suppress)
    r = np.clip(r, 0, 255)
    g = np.clip(g, 0, 255)
    b = np.clip(b, 0, 255)
    img_bgr = cv2.merge([b, g, r]).astype(np.float32)

    if defog_strength > 0.01:
        img_u8 = np.clip(img_bgr, 0, 255).astype(np.uint8)
        patch = max(3, int(min(h, w) * 0.03)) | 1
        dark = np.min(img_u8, axis=2)
        dark_eroded = cv2.erode(dark, np.ones((patch, patch), np.uint8))
        flat = dark_eroded.flatten()
        threshold = np.percentile(flat, 99.9)
        bright_mask = dark_eroded >= threshold
        if bright_mask.sum() == 0:
            bright_mask = dark_eroded >= flat.max() * 0.9
        A = np.mean(img_u8.reshape(-1, 3)[bright_mask.flatten()], axis=0)
        A = np.clip(A, 100, 255)
        norm = img_u8.astype(np.float32) / A.astype(np.float32)
        t_map = 1.0 - defog_strength * np.min(norm, axis=2)
        t_map = np.clip(t_map, 0.15, 1.0)
        t_map3 = cv2.GaussianBlur(t_map, (0, 0), max(1, patch // 2))
        t_map3 = np.stack([t_map3] * 3, axis=2)
        img_defogged = (img_bgr - A) / t_map3 + A
        img_bgr = np.clip(img_defogged, 0, 255)

    img_u8 = np.clip(img_bgr, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB)
    l, a, bb_ch = cv2.split(lab)
    tile = max(2, min(8, h // 80, w // 80))
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(tile, tile))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, bb_ch])
    img_bgr = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR).astype(np.float32)

    blur_r = max(1, int(min(h, w) * 0.006)) | 1
    blurred = cv2.GaussianBlur(img_bgr, (blur_r, blur_r), 0)
    img_bgr = cv2.addWeighted(img_bgr, 1.35, blurred, -0.35, 0)
    img_bgr = np.clip(img_bgr, 0, 255)

    img_u8 = img_bgr.astype(np.uint8)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_boost, 0, 255)
    img_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    if warmth > 0.001:
        b2, g2, r2 = cv2.split(img_bgr)
        r2 = np.clip(r2 + warmth * 18.0, 0, 255)
        g2 = np.clip(g2 + warmth * 7.0, 0, 255)
        b2 = np.clip(b2 - warmth * 8.0, 0, 255)
        img_bgr = cv2.merge([b2, g2, r2])

    img_rgb_out = cv2.cvtColor(np.clip(img_bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb_out)


def safe_resize(pil_img: Image.Image, max_px: int = 2048) -> Image.Image:
    w, h = pil_img.size
    if max(w, h) <= max_px:
        return pil_img
    ratio = max_px / max(w, h)
    return pil_img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)


def pil_to_bytes(pil_img: Image.Image, fmt: str = "JPEG", quality: int = 95) -> bytes:
    buf = io.BytesIO()
    if fmt == "JPEG":
        pil_img.save(buf, format="JPEG", quality=quality, subsampling=0)
    else:
        pil_img.save(buf, format="PNG")
    return buf.getvalue()


def pil_to_b64(pil_img: Image.Image) -> str:
    return base64.b64encode(pil_to_bytes(pil_img)).decode()


# UI
st.markdown("""
<div class="hero-header">
  <div class="hero-badge">🤿 UNDERWATER PHOTO AUTO-ENHANCER</div>
  <h1 class="hero-title">수중 사진,<br><span>한 번에 맑게</span></h1>
  <p class="hero-sub">녹색·청록 틴트 제거 · 붉은 채널 복원 · 디포그<br>
  얼굴과 형태는 100% 원본 그대로</p>
</div>
""", unsafe_allow_html=True)

col_ctrl, col_orig, col_result = st.columns([1.05, 1.47, 1.47], gap="large")

with col_ctrl:
    st.markdown('<div class="section-title">📂 이미지 업로드</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "JPG 또는 PNG 파일",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-title" style="margin-top:1.6rem">⚙️ 보정 파라미터</div>', unsafe_allow_html=True)

    with st.expander("🔴 레드 채널 (수심 색소실 복원)", expanded=True):
        st.markdown('<div class="param-label">Red Boost — 붉은 채널 강도</div>', unsafe_allow_html=True)
        red_boost = st.slider("red_boost", 1.0, 2.2, 1.4, 0.05, label_visibility="collapsed", key="rb")
        st.markdown('<div class="param-label">Warmth — 피부톤 황금빛</div>', unsafe_allow_html=True)
        warmth = st.slider("warmth", 0.0, 0.3, 0.08, 0.01, label_visibility="collapsed", key="wm")

    with st.expander("🟢 그린/블루 억제 (틴트 중화)", expanded=True):
        st.markdown('<div class="param-label">Green Suppress</div>', unsafe_allow_html=True)
        green_suppress = st.slider("green_suppress", 0.5, 1.05, 0.88, 0.01, label_visibility="collapsed", key="gs")
        st.markdown('<div class="param-label">Blue Suppress</div>', unsafe_allow_html=True)
        blue_suppress = st.slider("blue_suppress", 0.5, 1.05, 0.92, 0.01, label_visibility="collapsed", key="bs")

    with st.expander("🌫️ 디포그 & 명암대비", expanded=False):
        st.markdown('<div class="param-label">Defog Strength — 안개 제거</div>', unsafe_allow_html=True)
        defog_strength = st.slider("defog_strength", 0.0, 1.0, 0.55, 0.05, label_visibility="collapsed", key="df")
        st.markdown('<div class="param-label">CLAHE Clip — 국소 명암대비</div>', unsafe_allow_html=True)
        clahe_clip = st.slider("clahe_clip", 1.0, 6.0, 2.8, 0.2, label_visibility="collapsed", key="cc")
        st.markdown('<div class="param-label">Saturation Boost</div>', unsafe_allow_html=True)
        saturation_boost = st.slider("saturation_boost", 0.8, 1.8, 1.18, 0.02, label_visibility="collapsed", key="sb")

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🎨 보정 실행", use_container_width=True)

    st.markdown("""
    <div class="algo-box">
      <strong style="color:#00c8f8">처리 파이프라인</strong><br>
      ① 채널 선택적 레벨 보상<br>
      ② 그레이월드 화이트발란스<br>
      ③ 다크채널 디포그<br>
      ④ CLAHE 국소 명암대비<br>
      ⑤ 언샤프 마스크<br>
      ⑥ HSV 채도 스케일<br>
      ⑦ 워밍 커브<br>
      <br>
      <strong style="color:#00c8f8">✅ 픽셀 보존 원칙</strong><br>
      얼굴 · 표정 · 체형 · 장비 픽셀 위치 변경 없음
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-chips" style="margin-top:1rem">
      <span class="chip">AI 생성 없음</span>
      <span class="chip">100% 원본 구조</span>
      <span class="chip">순수 수학 변환</span>
    </div>
    """, unsafe_allow_html=True)


with col_orig:
    st.markdown('<div class="img-panel">', unsafe_allow_html=True)
    st.markdown('<div class="img-label">원본 (Original)</div>', unsafe_allow_html=True)
    if uploaded is not None:
        orig_pil = Image.open(uploaded)
        orig_safe = safe_resize(orig_pil, max_px=2048)
        ow, oh = orig_safe.size
        st.image(orig_safe, use_container_width=True)
        st.markdown(f'<div style="font-size:0.72rem;color:#4a6f88;margin-top:0.5rem">{ow} × {oh}px · {orig_safe.mode}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="height:340px;display:flex;align-items:center;justify-content:center;color:#2a4a60;font-size:0.85rem;flex-direction:column;gap:0.5rem">
          <div style="font-size:2.5rem;opacity:0.4">🌊</div>
          <div>업로드된 사진이 없습니다</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


with col_result:
    st.markdown('<div class="img-panel">', unsafe_allow_html=True)
    st.markdown('<div class="img-label">보정 결과 (Enhanced)</div>', unsafe_allow_html=True)
    if uploaded is not None and run_btn:
        with st.spinner("수중 컬러 그레이딩 처리 중…"):
            try:
                result_pil = enhance_underwater(
                    orig_safe,
                    red_boost=red_boost, green_suppress=green_suppress,
                    blue_suppress=blue_suppress, defog_strength=defog_strength,
                    clahe_clip=clahe_clip, saturation_boost=saturation_boost, warmth=warmth,
                )
                st.session_state["result"] = result_pil
            except Exception as e:
                st.error(f"처리 중 오류 발생: {e}")
                st.session_state.pop("result", None)

    if "result" in st.session_state and st.session_state["result"] is not None:
        res = st.session_state["result"]
        st.image(res, use_container_width=True)
        rw, rh = res.size
        st.markdown(f'<div style="font-size:0.72rem;color:#4a6f88;margin-top:0.5rem">{rw} × {rh}px · 보정 완료</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button("⬇ JPG 다운로드", data=pil_to_bytes(res, "JPEG"), file_name="underwater_enhanced.jpg", mime="image/jpeg", use_container_width=True)
        with dl_col2:
            st.download_button("⬇ PNG 다운로드", data=pil_to_bytes(res, "PNG"), file_name="underwater_enhanced.png", mime="image/png", use_container_width=True)
    elif uploaded is not None:
        st.markdown("""
        <div style="height:340px;display:flex;align-items:center;justify-content:center;color:#2a4a60;font-size:0.85rem;flex-direction:column;gap:0.5rem">
          <div style="font-size:2.5rem;opacity:0.4">✨</div>
          <div>← '보정 실행' 버튼을 눌러주세요</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="height:340px;display:flex;align-items:center;justify-content:center;color:#2a4a60;font-size:0.85rem;flex-direction:column;gap:0.5rem">
          <div style="font-size:2.5rem;opacity:0.4">🤿</div>
          <div>이미지를 먼저 업로드해 주세요</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;font-size:0.72rem;color:#2a4a60;padding-bottom:1rem">
  Underwater Photo Auto-Enhancer &nbsp;·&nbsp; Pixel-accurate color grading &nbsp;·&nbsp; No AI generation — pure math
</div>
""", unsafe_allow_html=True)
