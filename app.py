import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import base64

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

/* 헤더 */
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

/* 업로드 영역 */
.upload-zone {
    background: rgba(255,255,255,0.035);
    border: 1.5px dashed rgba(79,195,247,0.35);
    border-radius: 16px;
    padding: 2rem 1.5rem;
    text-align: center;
    margin: 1.5rem 0;
    transition: border-color 0.3s;
}

/* 실행 버튼 */
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

/* 다운로드 버튼 */
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

/* 이미지 캡션 */
.img-label {
    font-size: 0.82rem;
    font-weight: 500;
    color: #7baecf;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    text-align: center;
    margin-bottom: 0.5rem;
}

/* 분석 뱃지 */
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

/* Streamlit 기본 요소 정리 */
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

def safe_resize(pil_img: Image.Image, max_px: int = 2400) -> Image.Image:
    w, h = pil_img.size
    if max(w, h) <= max_px:
        return pil_img
    scale = max_px / max(w, h)
    return pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def pil_to_bytes(pil_img: Image.Image, fmt: str = "JPEG", quality: int = 95) -> bytes:
    buf = io.BytesIO()
    if fmt == "JPEG":
        pil_img.convert("RGB").save(buf, format="JPEG", quality=quality, subsampling=0)
    else:
        pil_img.save(buf, format="PNG")
    return buf.getvalue()


# ────────────────────────────────────────────────
# 이미지 자동 분석 함수
# 업로드된 사진의 색상·밝기·노이즈 특성을 측정해
# 최적 보정 파라미터를 자동으로 계산합니다.
# ────────────────────────────────────────────────

def analyze_image(bgr: np.ndarray) -> dict:
    """
    수중 사진의 색상·밝기·흐릿함 정도를 분석해 보정 파라미터 딕셔너리 반환.
    """
    # --- float32 변환 ---
    img_f = bgr.astype(np.float32) / 255.0
    r = img_f[:, :, 2]
    g = img_f[:, :, 1]
    b = img_f[:, :, 0]

    mean_r = float(r.mean())
    mean_g = float(g.mean())
    mean_b = float(b.mean())
    mean_lum = float(0.299 * r + 0.587 * g + 0.114 * b).mean() if False else float(
        (0.299 * r + 0.587 * g + 0.114 * b).mean()
    )

    # 전체 밝기
    # 수심별 색 손실 지표: 적색 부족 = (mean_g - mean_r)
    red_deficit = mean_g - mean_r          # 클수록 빨강이 많이 빠진 깊은 곳
    green_excess = mean_g - mean_b         # 클수록 녹조 느낌

    # 흐릿함 추정 (Laplacian variance)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # 액션캠 압축 노이즈 추정 (고주파 에너지)
    noise_estimate = max(0.0, 1.0 - min(laplacian_var / 300.0, 1.0))

    # ── 파라미터 자동 계산 ──

    # 적색 보상: 적색 부족량에 비례 (0.3 ~ 1.0 사이 / 최종 곱 1.15~1.75)
    red_boost = 1.15 + min(max(red_deficit * 3.2, 0.0), 0.60)

    # 녹색 억제: 녹색 과잉에 비례 (0.78~0.97)
    green_suppress = max(0.78, 1.0 - min(green_excess * 1.8, 0.22))

    # 청색 억제 (수중 파란 베일)
    blue_suppress = max(0.82, 1.0 - min(mean_b * 0.25, 0.18))

    # 디포그 강도: 밝기 낮고 흐릿할수록 강하게
    defog_strength = 0.35 + (1.0 - mean_lum) * 0.4 + noise_estimate * 0.15
    defog_strength = min(max(defog_strength, 0.25), 0.75)

    # CLAHE 클립: 저조도일수록 강하게
    clahe_clip = 2.0 + (1.0 - mean_lum) * 3.0
    clahe_clip = min(max(clahe_clip, 1.8), 4.5)

    # 채도 부스트: 색이 죽어있을수록 강하게
    color_flatness = 1.0 - min(abs(mean_r - mean_g) + abs(mean_g - mean_b), 0.4) / 0.4
    saturation_boost = 1.10 + color_flatness * 0.30
    saturation_boost = min(saturation_boost, 1.45)

    # 워밍: 피부톤 복원 (red_deficit이 클수록)
    warmth = 0.04 + min(red_deficit * 0.35, 0.14)

    # 언샤프 마스크: 흐릿할수록 강하게
    unsharp_amount = 0.6 + noise_estimate * 0.5 + (1.0 - min(laplacian_var / 500.0, 1.0)) * 0.4
    unsharp_amount = min(max(unsharp_amount, 0.5), 1.4)

    return {
        "red_boost": round(red_boost, 3),
        "green_suppress": round(green_suppress, 3),
        "blue_suppress": round(blue_suppress, 3),
        "defog_strength": round(defog_strength, 3),
        "clahe_clip": round(clahe_clip, 2),
        "saturation_boost": round(saturation_boost, 3),
        "warmth": round(warmth, 3),
        "unsharp_amount": round(unsharp_amount, 3),
        # 분석 요약 (UI 표시용)
        "_mean_r": round(mean_r, 3),
        "_mean_g": round(mean_g, 3),
        "_mean_b": round(mean_b, 3),
        "_mean_lum": round(mean_lum, 3),
        "_red_deficit": round(red_deficit, 3),
        "_laplacian_var": round(laplacian_var, 1),
    }


# ────────────────────────────────────────────────
# 다크 채널 기반 디포그 (국소 최솟값 + 대기광 추정)
# ────────────────────────────────────────────────

def dark_channel_defog(bgr: np.ndarray, strength: float = 0.55) -> np.ndarray:
    img = bgr.astype(np.float32) / 255.0
    # 다크 채널 (15x15 패치)
    dark = cv2.erode(img.min(axis=2), np.ones((15, 15), np.uint8))
    # 대기광 A: 다크 채널 상위 0.1% 밝기 픽셀의 평균
    num_px = max(int(dark.size * 0.001), 1)
    flat = dark.ravel()
    indices = np.argpartition(flat, -num_px)[-num_px:]
    atm_mask = np.zeros_like(flat, dtype=bool)
    atm_mask[indices] = True
    atm_mask = atm_mask.reshape(dark.shape)
    A = float(img[atm_mask].mean(axis=0).mean())
    A = max(A, 0.6)
    # 투과율
    t = 1.0 - strength * dark / max(A, 1e-6)
    t = np.clip(t, 0.25, 1.0)
    t3 = t[:, :, np.newaxis]
    # 복원
    result = (img - A) / t3 + A
    return np.clip(result * 255.0, 0, 255).astype(np.uint8)


# ────────────────────────────────────────────────
# 메인 보정 파이프라인
# ────────────────────────────────────────────────

def enhance_underwater(pil_img: Image.Image, params: dict) -> Image.Image:
    """
    6단계 수중 보정 파이프라인:
    1. 채널 레벨 보정 (적색 보상 + 녹/청색 억제)
    2. 다크 채널 디포그
    3. CLAHE (L채널 명암 대비)
    4. 언샤프 마스크 (선명도)
    5. HSV 채도 부스트
    6. 워밍 커브 (피부톤)
    """
    p = params
    bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

    # ── STEP 1: 채널 레벨 보정 ──
    img_f = bgr.astype(np.float32)
    img_f[:, :, 2] = np.clip(img_f[:, :, 2] * p["red_boost"], 0, 255)       # R↑
    img_f[:, :, 1] = np.clip(img_f[:, :, 1] * p["green_suppress"], 0, 255)  # G↓
    img_f[:, :, 0] = np.clip(img_f[:, :, 0] * p["blue_suppress"], 0, 255)   # B↓
    bgr = img_f.astype(np.uint8)

    # ── STEP 2: 디포그 ──
    bgr = dark_channel_defog(bgr, p["defog_strength"])

    # ── STEP 3: CLAHE (LAB 명도 채널) ──
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, ch_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=p["clahe_clip"], tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    bgr = cv2.cvtColor(cv2.merge([l_eq, a, ch_b]), cv2.COLOR_LAB2BGR)

    # ── STEP 4: 언샤프 마스크 ──
    amount = p["unsharp_amount"]
    blur = cv2.GaussianBlur(bgr, (0, 0), 1.5)
    bgr = cv2.addWeighted(bgr, 1 + amount, blur, -amount, 0)

    # ── STEP 5: HSV 채도 부스트 ──
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * p["saturation_boost"], 0, 255)
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ── STEP 6: 워밍 커브 (LUT) ──
    warmth = p["warmth"]
    lut_r = np.clip(np.arange(256, dtype=np.float32) * (1 + warmth), 0, 255).astype(np.uint8)
    lut_b = np.clip(np.arange(256, dtype=np.float32) * (1 - warmth * 0.5), 0, 255).astype(np.uint8)
    bgr[:, :, 2] = cv2.LUT(bgr[:, :, 2], lut_r)
    bgr[:, :, 0] = cv2.LUT(bgr[:, :, 0], lut_b)

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

        # ── 분석 결과 요약 뱃지 ──
        lum_pct = int(params["_mean_lum"] * 100)
        deficit = params["_red_deficit"]
        depth_label = (
            "얕은 수심 (5m 미만)" if deficit < 0.06
            else "중간 수심 (5–15m)" if deficit < 0.15
            else "깊은 수심 (15m+)"
        )
        lap = params["_laplacian_var"]
        clarity_label = (
            "선명" if lap > 200 else "약간 흐릿함" if lap > 60 else "많이 흐릿함"
        )

        st.markdown(f"""
        <div class="analysis-badge">
          📊 <strong>자동 분석 결과</strong> —
          밝기 {lum_pct}% · {depth_label} · {clarity_label}<br>
          🔴 적색 보상 {params['red_boost']:.2f}x ·
          🟢 녹색 억제 {int((1-params['green_suppress'])*100)}% ·
          디포그 {int(params['defog_strength']*100)}% ·
          채도 +{int((params['saturation_boost']-1)*100)}%
        </div>
        """, unsafe_allow_html=True)

        # ── 원본 / 보정 비교 ──
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="img-label">BEFORE — 원본</div>', unsafe_allow_html=True)
            st.image(pil_resized, use_container_width=True)
        with c2:
            st.markdown('<div class="img-label">AFTER — 보정 완료</div>', unsafe_allow_html=True)
            st.image(pil_result, use_container_width=True)

        # ── 다운로드 ──
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
    else:
        # 업로드 후 미리보기
        st.image(pil_resized, caption="업로드된 원본 사진", use_container_width=True)
else:
    st.info("📷 JPG 또는 PNG 파일을 위에 드래그하시거나 클릭해서 업로드하세요.")
