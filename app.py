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
    atm_mask = np.zeros_like(dark)
    atm_mask.ravel()[indices] = 1
    atm = np.array([
        img[:, :, c][atm_mask == 1].mean() for c in range(3)
    ], dtype=np.float32)

    # 투과률 t 추정 (BGR의 반색으로 조정)
    omega = 0.95
    t = 1.0 - omega * dark / (atm.max() + 1e-6)
    t = np.clip(t, 0.1, 1.0)

    # 복원: J = (I - A) / t + A, strength로 보간
    result = np.zeros_like(img)
    for c in range(3):
        result[:, :, c] = (img[:, :, c] - atm[c]) / (t + 1e-6) + atm[c]
    result = np.clip(result, 0, 1)

    # strength 바랙으로 원본과 혼합
    blended = img * (1 - strength) + result * strength
    return np.clip(blended * 255.0, 0, 255).astype(np.uint8)


# ────────────────────────────────────────────────
# 6단계 수중 보정 파이프라인
# ────────────────────────────────────────────────

def enhance_underwater(pil_img: Image.Image, params: dict) -> Image.Image:
    """
    수중 사진 보정 6단계 파이프라인.
    인물(얼굴, 표정, 체형, 장비)의 픽셀 일체 변형 없음.
    색상 채널만 수학적으로 보정합니다.
    """
    bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

    # ----- STEP 1: 채널색 레벨 보정 -----
    # 적색 상승, 녹색·청색 억제
    lut = np.arange(256, dtype=np.float32)
    r_lut = np.clip(lut * params["red_boost"],   0, 255).astype(np.uint8)
    g_lut = np.clip(lut * params["green_suppress"], 0, 255).astype(np.uint8)
    b_lut = np.clip(lut * params["blue_suppress"],  0, 255).astype(np.uint8)

    b, g, r = cv2.split(bgr)
    bgr = cv2.merge([
        cv2.LUT(b, b_lut),
        cv2.LUT(g, g_lut),
        cv2.LUT(r, r_lut),
    ])

    # ----- STEP 2: 다크 채널 디포그 -----
    bgr = dark_channel_defog(bgr, strength=params["defog_strength"])

    # ----- STEP 3: CLAHE (LAB L-채널) -----
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=params["clahe_clip"],
        tileGridSize=(8, 8)
    )
    l = clahe.apply(l)
    bgr = cv2.cvtColor(cv2.merge([l, a, b_ch]), cv2.COLOR_LAB2BGR)

    # ----- STEP 4: 언샤프 마스크 (선명도 향상) -----
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=2.5)
    amt = params["unsharp_amount"]
    bgr = cv2.addWeighted(bgr, 1.0 + amt, blurred, -amt, 0)
    bgr = np.clip(bgr, 0, 255).astype(np.uint8)

    # ----- STEP 5: HSV 채도 부스트 -----
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * params["saturation_boost"], 0, 255)
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ----- STEP 6: 워밍 LUT (피부톤 워밍) -----
    w = params["warmth"]
    r_warm = np.clip(lut + w * 255, 0, 255).astype(np.uint8)
    b_cool = np.clip(lut - w * 128, 0, 255).astype(np.uint8)
    b2, g2, r2 = cv2.split(bgr)
    bgr = cv2.merge([
        cv2.LUT(b2, b_cool),
        g2,
        cv2.LUT(r2, r_warm),
    ])

    # BGR → RGB → PIL
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ────────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────────

st.set_page_config(
    page_title="수중 사진 보정",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─ 헤더 스타일 ─
st.markdown("""
<style>
    /* 전체 배경 */
    .stApp { background: #0a0e1a; color: #e8eaf6; }
    /* 헤더 영역 */
    .hero-header {
        background: linear-gradient(135deg, #0d1b3e 0%, #0a2744 50%, #0d1f3c 100%);
        border-bottom: 1px solid #1e3a5f;
        padding: 2rem 2.5rem 1.5rem;
        margin: -1rem -1rem 2rem;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4fc3f7, #81d4fa, #b3e5fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 .4rem;
    }
    .hero-sub {
        color: #7986cb;
        font-size: .95rem;
        margin: 0;
    }
    /* 보정 버튜 */
    .stButton > button {
        background: linear-gradient(135deg, #1565c0, #0d47a1);
        color: #fff;
        border: none;
        border-radius: 12px;
        font-size: 1.15rem;
        font-weight: 700;
        padding: .8rem 2.5rem;
        width: 100%;
        cursor: pointer;
        transition: opacity .2s;
    }
    .stButton > button:hover { opacity: .85; }
    /* 아나리시스 배지 */
    .badge-box {
        background: #0d1b3e;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 1rem 1.4rem;
        margin: 1rem 0;
        font-size: .85rem;
        color: #90caf9;
        line-height: 1.7;
    }
    .badge-box b { color: #4fc3f7; }
    /* 이미지 라벨 */
    .img-label {
        text-align: center;
        font-size: .9rem;
        font-weight: 600;
        margin-bottom: .4rem;
        letter-spacing: .06em;
    }
    .label-before { color: #ef9a9a; }
    .label-after  { color: #a5d6a7; }
</style>
""", unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero-header">
    <p class="hero-title">🐙 수중 사진 자동 보정</p>
    <p class="hero-sub">업로드 한 장·버튼 클릭 한 번&nbsp;&mdash;&nbsp;전문가가 직접 보정한 것 같은 결과물</p>
</div>
""", unsafe_allow_html=True)

# 파일 업로더
uploaded = st.file_uploader(
    "수중 사진을 업로드하세요 (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded:
    # 사진 로드 & 안전 리사이즈
    pil_orig = Image.open(uploaded).convert("RGB")
    pil_orig = safe_resize(pil_orig, max_px=2400)

    # 보정 실행 버튜
    run_btn = st.button("🎨 보정 실행")

    if run_btn:
        with st.spinner("이미지 분석 중… 최적 보정 파라미터 쪼고 있어요 🤓"):
            bgr_orig = cv2.cvtColor(np.array(pil_orig), cv2.COLOR_RGB2BGR)
            params  = analyze_image(bgr_orig)
            pil_out = enhance_underwater(pil_orig, params)

        # ─ 분석 배지 ─
        depth_label = (
            "💙 서피스 (0~5m)" if params["_red_deficit"] < 0.08 else
            "💙 중심도 (5~15m)" if params["_red_deficit"] < 0.18 else
            "💙 심수 (15m+)"
        )
        clarity_label = (
            "✨ 맑음" if params["_laplacian_var"] > 300 else
            "🌫️ 약간 후리진" if params["_laplacian_var"] > 80 else
            "🌫️ 후리진"
        )
        st.markdown(f"""
        <div class="badge-box">
            <b>🔍 자동 분석 결과</b><br>
            수심: {depth_label} &nbsp;|
            선명도: {clarity_label} &nbsp;|
            적색 보상: <b>+{int((params['red_boost']-1)*100)}%</b> &nbsp;|
            포그 제거: <b>{int(params['defog_strength']*100)}%</b> &nbsp;|
            채도: <b>+{int((params['saturation_boost']-1)*100)}%</b>
        </div>
        """, unsafe_allow_html=True)

        # ─ BEFORE / AFTER ─
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="img-label label-before">■ BEFORE (원본)</p>',
                        unsafe_allow_html=True)
            st.image(pil_orig, use_container_width=True)
        with col2:
            st.markdown('<p class="img-label label-after">■ AFTER (보정)</p>',
                        unsafe_allow_html=True)
            st.image(pil_out, use_container_width=True)

        # ─ 다운로드 ─
        st.markdown("---")
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                label="⬇️ JPG 다운로드",
                data=pil_to_bytes(pil_out, "JPEG"),
                file_name="underwater_corrected.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                label="⬇️ PNG 다운로드",
                data=pil_to_bytes(pil_out, "PNG"),
                file_name="underwater_corrected.png",
                mime="image/png",
                use_container_width=True,
            )
    else:
        # 버튜 누르기 전 미리보기
        st.image(pil_orig, caption="업로드된 원본 사진", use_container_width=True)
else:
    st.info("📷 JPG 또는 PNG 파일을 위에 드래그하시거나 클릭해서 업로드하세요.")
