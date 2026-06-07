import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import base64
import os
import sys
import uuid
import sys
import os
from streamlit_mic_recorder import mic_recorder

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "bot_whisper"
    )
)

from ragbot import build_rag_pipeline

@st.cache_resource
def load_rag():
    return build_rag_pipeline()

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "the_rag_one"
    )
)

from chat_engine import get_chat_response

# ─────────────────────────────────────────────
# 1. PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PlantGuard · AI Crop Diagnostics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# 2. KNOWLEDGE BASE
# ─────────────────────────────────────────────
PLANT_NAME_MAPPING = {
    'Apple___healthy': 'Apple',
    'Corn___Healthy': 'Corn (Maize)',
    'Pepper__bell___healthy': 'Bell Pepper',
    'Potato___Healthy': 'Potato',
    'Rice___Healthy': 'Rice',
    'Sugarcane_Healthy': 'Sugarcane',
    'Tomato_healthy': 'Tomato',
    'Wheat___Healthy': 'Wheat'
}

ROUTER_CLASS_NAMES = [
    'Apple___healthy', 'Corn___Healthy', 'Pepper__bell___healthy',
    'Potato___Healthy', 'Rice___Healthy', 'Sugarcane_Healthy',
    'Tomato_healthy', 'Wheat___Healthy'
]

MODEL_ROUTER = {
    'Apple___healthy': "apple.keras",
    'Corn___Healthy': "corn.keras",
    'Pepper__bell___healthy': "bellpepper.keras",
    'Potato___Healthy': "potato.keras",
    'Rice___Healthy': "rice.keras",
    'Sugarcane_Healthy': "sugarcane.keras",
    'Tomato_healthy': "tomato.keras",
    'Wheat___Healthy': "wheat.keras"
}

DISEASE_CLASSES = {
    "apple.keras": ["Apple Scab", "Black Rot", "Cedar Apple Rust", "Healthy"],
    "bellpepper.keras": ["Bacterial Spot", "Healthy"],
    "corn.keras": ["Common Rust", "Gray Leaf Spot", "Northern Leaf Blight", "Healthy"],
    "potato.keras": ["Early Blight", "Late Blight", "Healthy"],
    "rice.keras": ["Brown Spot", "Leaf Blast", "Neck Blast", "Healthy"],
    "sugarcane.keras": ["Bacterial Blight", "Red Rot", "Healthy"],
    "tomato.keras": [
        "Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold",
        "Septoria Leaf Spot", "Spider Mites", "Target Spot",
        "Yellow Leaf Curl Virus", "Healthy"
    ],
    "wheat.keras": ["Brown Rust", "Yellow Rust", "Healthy"]
}

DISEASE_INFO = {
    "Early Blight": {
        "icon": "🍂",
        "symptoms": ["Dark concentric rings on older leaves", "Yellowing tissue surrounding spots", "Premature leaf drop"],
        "causes": ["Fungus Alternaria solani", "Warm temperatures + high humidity", "Overcrowded planting"],
        "prevention": ["Crop rotation — avoid consecutive nightshades", "Mulching to prevent soil splash", "Use drip irrigation"],
        "treatment": "Apply chlorothalonil or mancozeb fungicide every 7–10 days. Remove and destroy infected leaves promptly.",
        "buy_link": "https://www.amazon.in/s?k=chlorothalonil+fungicide"
    },
    "Late Blight": {
        "icon": "🦠",
        "symptoms": ["Large dark brown blotches with green-gray edges", "White fungal growth on leaf undersides", "Rapid rotting of fruit/tubers"],
        "causes": ["Oomycete Phytophthora infestans", "Cool wet weather", "Infected seed tubers"],
        "prevention": ["Use certified resistant varieties", "Apply preventative fungicides", "Destroy infected plant debris"],
        "treatment": "Spray metalaxyl or cymoxanil-based fungicide immediately. Destroy infected crop residue.",
        "buy_link": "https://www.amazon.in/s?k=metalaxyl+fungicide"
    },
    "Bacterial Spot": {
        "icon": "🔬",
        "symptoms": ["Small water-soaked leaf spots", "Spots turn brown with shot-hole effect", "Raised scab-like spots on fruit"],
        "causes": ["Bacteria Xanthomonas spp.", "Splashing rain or overhead irrigation", "Warm moist conditions"],
        "prevention": ["Use disease-free certified seeds", "Copper-based bactericides as preventative", "Avoid working in wet fields"],
        "treatment": "Apply copper hydroxide bactericide. Remove heavily infected plants to prevent spread.",
        "buy_link": "https://www.amazon.in/s?k=copper+hydroxide+bactericide"
    },
    "Common Rust": {
        "icon": "🌾",
        "symptoms": ["Reddish-brown pustules on both leaf surfaces", "Leaves yellowing and drying out"],
        "causes": ["Fungal pathogen Puccinia sorghi", "Cool moist weather"],
        "prevention": ["Plant resistant hybrids", "Apply fungicides early in season"],
        "treatment": "Apply propiconazole or azoxystrobin fungicide. Ensure good field air circulation.",
        "buy_link": "https://www.amazon.in/s?k=propiconazole+fungicide"
    },
    "Apple Scab": {
        "icon": "🍎",
        "symptoms": ["Olive-green or brown velvety spots on leaves", "Scabby corky lesions on fruit", "Premature defoliation"],
        "causes": ["Fungus Venturia inaequalis", "Cool wet spring weather", "High humidity"],
        "prevention": ["Use scab-resistant apple varieties", "Prune for air circulation", "Rake fallen leaves"],
        "treatment": "Apply captan or myclobutanil fungicide at bud break and repeat every 10–14 days.",
        "buy_link": "https://www.amazon.in/s?k=captan+fungicide+apple"
    },
    "Default": {
        "icon": "🔍",
        "symptoms": ["Visible discoloration or lesions on leaf surface", "Stunted growth or wilting"],
        "causes": ["Pathogens (Fungal, Bacterial, Viral) or Environmental stress"],
        "prevention": ["Isolate affected plants", "Ensure proper air circulation", "Consult a local agricultural extension"],
        "treatment": "Identify the exact pathogen before treatment. Use broad-spectrum neem oil as first response.",
        "buy_link": "https://www.amazon.in/s?k=neem+oil+plant+disease"
    }
}

# ─────────────────────────────────────────────
# 3. GLOBAL CSS — FULL REDESIGN
# ─────────────────────────────────────────────
def inject_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&family=DM+Mono:wght@400;500&display=swap');

    /* ── RESET & ROOT ── */
    :root {
        --ink:        #0d1f17;
        --deep:       #1a3a2a;
        --mid:        #2d6a4f;
        --bright:     #52b788;
        --light:      #95d5b2;
        --cream:      #f8f4ec;
        --amber:      #e9c46a;
        --rust:       #e76f51;
        --glass:      rgba(255,255,255,0.05);
        --border:     rgba(149,213,178,0.15);
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--deep) !important;
        font-family: 'DM Sans', sans-serif !important;
        color: var(--cream) !important;
    }

    /* Animated radial background */
    [data-testid="stAppViewContainer"]::before {
        content: '';
        position: fixed;
        inset: 0;
        background:
            radial-gradient(ellipse 90% 60% at 10% 0%, rgba(82,183,136,0.14) 0%, transparent 55%),
            radial-gradient(ellipse 70% 50% at 90% 100%, rgba(26,58,42,0.5) 0%, transparent 60%);
        pointer-events: none;
        z-index: 0;
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header, [data-testid="stToolbar"],
    [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
    [data-testid="block-container"] { padding: 2rem 3rem 4rem !important; max-width: 1100px !important; }

    /* Ensure custom HTML renders correctly */
    .element-container { overflow: visible !important; }
    .stMarkdown { overflow: visible !important; }
    .stMarkdown > div { overflow: visible !important; }

    /* ── TYPOGRAPHY ── */
    h1 { font-family: 'Playfair Display', serif !important; font-size: clamp(40px,5vw,64px) !important;
         font-weight: 900 !important; letter-spacing: -2px !important; line-height: 1.05 !important;
         color: var(--cream) !important; text-shadow: none !important; }
    h2, h3 { font-family: 'Playfair Display', serif !important; color: var(--cream) !important; text-shadow: none !important; }
    p, label { color: var(--cream) !important; }
    .stMarkdown p { font-size: 15px !important; color: rgba(248,244,236,0.75) !important; }
    .stMarkdown div { color: var(--cream) !important; }

    /* ── FILE UPLOADER ── */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.04) !important;
        border: 1.5px dashed rgba(149,213,178,0.3) !important;
        border-radius: 18px !important;
        padding: 12px !important;
        transition: border-color 0.3s !important;
    }
    [data-testid="stFileUploader"]:hover { border-color: rgba(149,213,178,0.6) !important; }
    [data-testid="stFileUploader"] label { color: var(--light) !important; font-family: 'DM Mono', monospace !important;
        font-size: 11px !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; }
    [data-testid="stFileUploader"] section { background: transparent !important; border: none !important; }
    [data-testid="stFileDropzoneInstructions"] { color: rgba(248,244,236,0.5) !important; }

    /* ── BUTTONS ── */
    .stButton > button {
        background: linear-gradient(135deg, var(--mid), var(--bright)) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 16px 40px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        letter-spacing: 0.02em !important;
        cursor: pointer !important;
        box-shadow: 0 8px 28px rgba(45,106,79,0.45) !important;
        transition: all 0.25s !important;
        width: 100% !important;
    }
    .stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 14px 36px rgba(45,106,79,0.55) !important; }

    /* ── INFO / ALERT ── */
    [data-testid="stAlert"] {
        background: rgba(82,183,136,0.1) !important;
        border: 1px solid rgba(82,183,136,0.25) !important;
        border-radius: 14px !important;
        color: var(--light) !important;
    }
    [data-testid="stAlert"] p { color: var(--light) !important; }
    .stSuccess { background: rgba(82,183,136,0.12) !important; border-color: rgba(82,183,136,0.3) !important; }

    /* ── PROGRESS BARS ── */
    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, var(--mid), var(--bright)) !important;
        border-radius: 4px !important;
    }
    [data-testid="stProgressBar"] { background: rgba(255,255,255,0.08) !important; border-radius: 4px !important; }

    /* ── IMAGES ── */
    [data-testid="stImage"] img { border-radius: 16px !important; border: 1px solid var(--border) !important; }

    /* ── SPINNER ── */
    [data-testid="stSpinner"] { color: var(--bright) !important; }

    /* ── CAPTIONS ── */
    .stCaption { color: rgba(248,244,236,0.45) !important; font-family: 'DM Mono', monospace !important;
        font-size: 11px !important; letter-spacing: 0.06em !important; text-transform: uppercase !important; }

    /* ── DIVIDER ── */
    hr { border-color: rgba(149,213,178,0.12) !important; }

    /* ── SCROLLBAR ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--ink); }
    ::-webkit-scrollbar-thumb { background: var(--mid); border-radius: 3px; }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 4. CUSTOM HTML COMPONENTS
# ─────────────────────────────────────────────
def render_hero():
    st.markdown("""
    <div style="padding: 20px 0 48px; animation: fadeUp 0.7s ease both;">
        <div style="display:flex; align-items:center; gap:16px; margin-bottom:20px;">
            <div style="
                width:52px; height:52px; border-radius:14px;
                background: linear-gradient(135deg, #2d6a4f, #52b788);
                display:flex; align-items:center; justify-content:center;
                font-size:26px; box-shadow: 0 8px 24px rgba(45,106,79,0.4);">
                🌿
            </div>
            <div>
                <div style="font-family:'DM Mono',monospace; font-size:10px; text-transform:uppercase;
                    letter-spacing:0.15em; color:#52b788; margin-bottom:2px;">
                    AI · CROP DIAGNOSTICS
                </div>
                <h1 style="margin:0; font-family:'Playfair Display',serif; font-size:42px;
                    font-weight:900; letter-spacing:-1.5px; color:#f8f4ec;">
                    Plant<span style="color:#95d5b2;">Guard</span>
                </h1>
            </div>
        </div>
        <p style="font-size:17px; color:rgba(248,244,236,0.6); max-width:540px;
            line-height:1.7; font-weight:300; font-family:'DM Sans',sans-serif;">
            Upload two leaf images — one to identify the crop, one to diagnose disease.
            Our CNN pipeline delivers clinical-grade results in seconds.
        </p>
    </div>
    <style>
    @keyframes fadeUp {
        from { opacity:0; transform:translateY(18px); }
        to   { opacity:1; transform:translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)


def render_section_label(num, label, desc):
    st.markdown(f"""
    <div style="margin-bottom:18px;">
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:6px;">
            <div style="
                width:28px; height:28px; border-radius:8px;
                background: linear-gradient(135deg, #2d6a4f, #52b788);
                display:flex; align-items:center; justify-content:center;
                font-family:'DM Mono',monospace; font-size:12px; font-weight:500; color:white;
                flex-shrink:0;">
                {num}
            </div>
            <span style="font-family:'Playfair Display',serif; font-size:20px; font-weight:700; color:#f8f4ec;">
                {label}
            </span>
        </div>
        <p style="font-size:13px; color:rgba(248,244,236,0.45); margin:0; padding-left:40px;
            font-family:'DM Mono',monospace; text-transform:uppercase; letter-spacing:0.06em;">
            {desc}
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_result_card(plant_name, disease_name, confidence, is_healthy, status_msg):
    if is_healthy:
        accent = "#52b788"
        bg = "linear-gradient(135deg, rgba(45,106,79,0.35), rgba(82,183,136,0.15))"
        border = "rgba(82,183,136,0.4)"
        badge_bg = "rgba(82,183,136,0.2)"
        badge_color = "#95d5b2"
        badge_text = "✅ HEALTHY PLANT"
        icon = "🌱"
    elif confidence >= 70:
        accent = "#e9c46a"
        bg = "linear-gradient(135deg, rgba(233,196,106,0.2), rgba(244,162,97,0.1))"
        border = "rgba(233,196,106,0.4)"
        badge_bg = "rgba(233,196,106,0.15)"
        badge_color = "#e9c46a"
        badge_text = "🔴 DISEASE DETECTED"
        icon = "⚠️"
    else:
        accent = "#e76f51"
        bg = "linear-gradient(135deg, rgba(231,111,81,0.2), rgba(244,162,97,0.1))"
        border = "rgba(231,111,81,0.35)"
        badge_bg = "rgba(231,111,81,0.15)"
        badge_color = "#ffb4a2"
        badge_text = "⚠️ LOW CONFIDENCE"
        icon = "🔍"

    st.markdown(f"""
    <div style="
        background: {bg};
        border: 1px solid {border};
        border-radius: 24px;
        padding: 36px 40px;
        margin: 28px 0;
        position: relative;
        overflow: hidden;
    ">
        <!-- Decorative circle -->
        <div style="
            position:absolute; top:-40px; right:-40px;
            width:180px; height:180px; border-radius:50%;
            background: radial-gradient(circle, {accent + "18"}, transparent 70%);
            pointer-events:none;
        "></div>

        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px;">
            <div>
                <div style="
                    font-family:'DM Mono',monospace; font-size:11px;
                    text-transform:uppercase; letter-spacing:0.1em;
                    color:rgba(248,244,236,0.4); margin-bottom:6px;
                ">Identified Crop</div>
                <div style="font-family:'DM Sans',sans-serif; font-size:15px; font-weight:500;
                    color:rgba(248,244,236,0.8); margin-bottom:14px;">
                    {plant_name}
                </div>
                <div style="font-family:'Playfair Display',serif; font-size:clamp(28px,4vw,44px);
                    font-weight:900; letter-spacing:-1.5px; color:#f8f4ec; line-height:1.05;">
                    {disease_name}
                </div>
                <div style="font-family:'DM Sans',sans-serif; font-size:14px;
                    color:rgba(248,244,236,0.55); margin-top:8px;">
                    {status_msg}
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-family:'DM Mono',monospace; font-size:10px;
                    text-transform:uppercase; letter-spacing:0.12em;
                    color:rgba(248,244,236,0.4); margin-bottom:6px;">Confidence</div>
                <div style="font-family:'Playfair Display',serif; font-size:52px;
                    font-weight:900; color:{accent}; line-height:1;">
                    {confidence:.0f}%
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_distribution_bar(name, prob, color):
    bar_width = min(int(prob), 100)
    st.markdown(f"""
    <div style="margin-bottom:18px;">
        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
            <span style="font-size:14px; font-weight:500; color:#f8f4ec;">{name}</span>
            <span style="font-family:'DM Mono',monospace; font-size:12px; color:{color};">
                {prob:.1f}%
            </span>
        </div>
        <div style="height:6px; background:rgba(255,255,255,0.08); border-radius:3px; overflow:hidden;">
            <div style="
                width:{bar_width}%;
                height:100%;
                background: linear-gradient(90deg, {color}, {color + "88"});
                border-radius:3px;
                transition: width 1s ease;
            "></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_disease_profile(disease_name, info):
    icon = info.get("icon", "🔬")
    treatment = info.get("treatment", "")
    buy_link = info.get("buy_link", "#")

    st.markdown(f"""
    <div style="margin: 36px 0 20px;">
        <div style="font-family:'DM Mono',monospace; font-size:10px; text-transform:uppercase;
            letter-spacing:0.15em; color:#52b788; margin-bottom:8px;">
            Disease Profile
        </div>
        <h2 style="font-family:'Playfair Display',serif; font-size:32px; font-weight:900;
            letter-spacing:-1px; color:#f8f4ec; margin:0;">
            {icon} {disease_name}
        </h2>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    card_style = """
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(149,213,178,0.12);
        border-radius: 20px;
        padding: 24px;
        height: 100%;
    """

    with c1:
        items = "".join([f'<li style="margin-bottom:8px; color:rgba(248,244,236,0.8); font-size:14px; line-height:1.5;">{s}</li>' for s in info["symptoms"]])
        st.markdown(f"""
        <div style="{card_style}">
            <div style="font-family:'DM Mono',monospace; font-size:10px; text-transform:uppercase;
                letter-spacing:0.12em; color:#52b788; margin-bottom:14px;">🩺 Symptoms</div>
            <ul style="padding-left:18px; margin:0;">{items}</ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        items = "".join([f'<li style="margin-bottom:8px; color:rgba(248,244,236,0.8); font-size:14px; line-height:1.5;">{s}</li>' for s in info["causes"]])
        st.markdown(f"""
        <div style="{card_style}">
            <div style="font-family:'DM Mono',monospace; font-size:10px; text-transform:uppercase;
                letter-spacing:0.12em; color:#e9c46a; margin-bottom:14px;">🦠 Causes</div>
            <ul style="padding-left:18px; margin:0;">{items}</ul>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        items = "".join([f'<li style="margin-bottom:8px; color:rgba(248,244,236,0.8); font-size:14px; line-height:1.5;">{s}</li>' for s in info["prevention"]])
        st.markdown(f"""
        <div style="{card_style}">
            <div style="font-family:'DM Mono',monospace; font-size:10px; text-transform:uppercase;
                letter-spacing:0.12em; color:#95d5b2; margin-bottom:14px;">🛡️ Prevention</div>
            <ul style="padding-left:18px; margin:0;">{items}</ul>
        </div>
        """, unsafe_allow_html=True)

    # Treatment + Buy card
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(45,106,79,0.2), rgba(82,183,136,0.08));
        border: 1px solid rgba(82,183,136,0.2);
        border-radius: 20px;
        padding: 28px 32px;
        margin-top: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 24px;
        flex-wrap: wrap;
    ">
        <div style="flex:1; min-width:260px;">
            <div style="font-family:'DM Mono',monospace; font-size:10px; text-transform:uppercase;
                letter-spacing:0.12em; color:#52b788; margin-bottom:10px;">💊 Recommended Treatment</div>
            <p style="font-size:15px; color:rgba(248,244,236,0.85); line-height:1.7; margin:0;">{treatment}</p>
        </div>
        <a href="{buy_link}" target="_blank" style="
            display:inline-flex; align-items:center; gap:10px;
            background: linear-gradient(135deg, #2d6a4f, #52b788);
            color: white; text-decoration: none;
            padding: 14px 28px; border-radius: 50px;
            font-family: 'DM Sans', sans-serif;
            font-size: 14px; font-weight: 500;
            box-shadow: 0 8px 24px rgba(45,106,79,0.4);
            white-space: nowrap;
            flex-shrink: 0;
        ">
            🛒 Buy Medicine ↗
        </a>
    </div>
    """, unsafe_allow_html=True)


def render_section_card(title, content_html):
    st.markdown(f"""
    <div style="
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(149,213,178,0.12);
        border-radius: 20px;
        padding: 28px;
        margin-bottom: 20px;
    ">
        <div style="font-family:'DM Mono',monospace; font-size:10px; text-transform:uppercase;
            letter-spacing:0.12em; color:#52b788; margin-bottom:16px;">
            {title}
        </div>
        {content_html}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 5. CORE ML PIPELINE (unchanged logic)
# ─────────────────────────────────────────────
@st.cache_resource
def load_cached_model(path):
    if not os.path.exists(path):
        return None
    return load_model(path)


def preprocess_image(image_file):
    img = Image.open(image_file)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)
    return img_array, img


def predict_pipeline(model, img_array, class_names):
    preds = model.predict(img_array)
    score = tf.nn.softmax(preds[0])
    top_idx = np.argmax(score)
    confidence = 100 * np.max(score)
    top_class = class_names[top_idx] if top_idx < len(class_names) else f"Unknown ({top_idx})"
    top_3_indices = np.argsort(score)[::-1][:3]
    top_3 = [(class_names[i], float(100 * score[i])) for i in top_3_indices if i < len(class_names)]
    return top_class, confidence, top_3, top_idx


# ─────────────────────────────────────────────
# 6. MAIN APP
# ─────────────────────────────────────────────
def main():
    

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    inject_styles()
    
    tab1, tab2 = st.tabs([
        "🤖 AgriMind AI",
        "🌿 Disease Detection"
    ])
    with tab1:
            if "rag_messages" not in st.session_state:
                st.session_state.rag_messages = []
            for msg in st.session_state.rag_messages:

                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
            
            st.title("🤖 AgriMind AI")

            st.caption(
                "Ask any agriculture or crop disease question"
            )
            pipeline = load_rag()
            col1, col2 = st.columns([20,1])
            with col2:
                audio = mic_recorder(
                    start_prompt="🎤",
                    stop_prompt="⏹️",
                    just_once=True,
                    key="rag_mic"
                )
                if audio and "bytes" in audio:

                    with st.spinner("Transcribing..."):

                        query, answer = pipeline.ask_voice(
                            audio["bytes"]
                        )

                    st.session_state.rag_messages.append({
                        "role": "user",
                        "content": query
                    })

                    st.session_state.rag_messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                    st.rerun()
            for msg in st.session_state.rag_messages:

                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

        

            pipeline = load_rag()
            with col1:
                if "rag_messages" not in st.session_state:
                    st.session_state.rag_messages = []

                for msg in st.session_state.rag_messages:

                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])

                query = st.chat_input(
                    "Ask about crops, diseases, fertilizers..."
                )

            if query:

                st.session_state.rag_messages.append({
                    "role": "user",
                    "content": query
                })

                with st.chat_message("user"):
                    st.write(query)

                with st.spinner("Thinking..."):

                    answer = pipeline.ask(query)

                st.session_state.rag_messages.append({
                    "role": "assistant",
                    "content": answer
                })

                with st.chat_message("assistant"):
                    st.write(answer)
    with tab2:
        render_hero()

        st.markdown('<hr style="border-color:rgba(149,213,178,0.1); margin-bottom:40px;">', unsafe_allow_html=True)

        # ── INPUT SECTION ──
        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            render_section_label("01", "Crop Identification", "Upload a clear healthy leaf to identify crop type")
            file_plant_id = st.file_uploader(
                "Upload Leaf Image",
                type=["jpg", "png", "jpeg"],
                key="plant",
                label_visibility="collapsed"
            )
            if file_plant_id:
                st.image(file_plant_id, caption="Identification sample", use_container_width=True)

        with col_b:
            render_section_label("02", "Disease Analysis", "Upload the affected leaf area for diagnosis")
            file_disease = st.file_uploader(
                "Upload Affected Leaf",
                type=["jpg", "png", "jpeg"],
                key="disease",
                label_visibility="collapsed"
            )
            if file_disease:
                st.image(file_disease, caption="Diagnostic sample", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── RUN BUTTON ──
        left, center, right = st.columns([1, 2, 1])

        with center:
            run = st.button("🔬  Run Comprehensive Analysis", type="primary", use_container_width=True)

        # ── EXECUTION ──
        if run:
            if not file_plant_id or not file_disease:
                st.error("⚠️ Please upload **both** images to proceed.")
                st.stop()

            with st.spinner("Analyzing bio-markers · running CNN diagnostics…"):
                try:
                    # STAGE 1 — ROUTER
                    router_model = load_cached_model("healthy.keras")
                    if not router_model:
                        st.error("❌ Router model 'healthy.keras' not found.")
                        st.stop()

                    img_array_id, _ = preprocess_image(file_plant_id)
                    detected_raw, _, _, _ = predict_pipeline(router_model, img_array_id, ROUTER_CLASS_NAMES)
                    plant_display_name = PLANT_NAME_MAPPING.get(detected_raw, detected_raw.split('_')[0])

                    # STAGE 2 — DISEASE MODEL
                    target_model_file = MODEL_ROUTER.get(detected_raw)
                    if not target_model_file:
                        st.error(f"❌ No model available for: {detected_raw}")
                        st.stop()

                    disease_model = load_cached_model(target_model_file)
                    if not disease_model:
                        st.error(f"❌ Model file '{target_model_file}' is missing.")
                        st.stop()

                    specific_classes = DISEASE_CLASSES.get(target_model_file, [f"Class {i}" for i in range(10)])
                    img_array_disease, _ = preprocess_image(file_disease)
                    disease_name, confidence, top_3, top_idx = predict_pipeline(
                        disease_model, img_array_disease, specific_classes
                    )

                    # STAGE 3 — DETERMINE STATUS
                    is_healthy = "Healthy" in disease_name
                    if confidence < 55:
                        disease_name = "Unidentified Issue"
                        status_msg = "Low confidence — please try a clearer photo"
                    elif confidence < 70:
                        status_msg = "Moderate confidence — consider a second sample"
                    else:
                        status_msg = "High confidence match · clinically actionable"

                except Exception as e:
                    st.error(f"An internal error occurred: {e}")
                    st.stop()

            # ── RESULTS ──
            st.markdown('<hr style="border-color:rgba(149,213,178,0.1); margin:32px 0;">', unsafe_allow_html=True)

            # Main result card
            render_result_card(plant_display_name, disease_name, confidence, is_healthy, status_msg)

            # Distribution
            st.markdown("""
            <div style="font-family:'DM Mono',monospace; font-size:10px; text-transform:uppercase;
                letter-spacing:0.15em; color:#52b788; margin:32px 0 16px;">
                📊 Prediction Distribution
            </div>
            """, unsafe_allow_html=True)

            bar_colors = ["#52b788", "#e9c46a", "#e76f51"]
            for i, (name, prob) in enumerate(top_3):
                render_distribution_bar(name, prob, bar_colors[i % len(bar_colors)])

            # Disease profile (only if not healthy and confident enough)
            if not is_healthy and confidence >= 55 and disease_name != "Unidentified Issue":
                st.markdown('<hr style="border-color:rgba(149,213,178,0.1); margin:32px 0;">', unsafe_allow_html=True)
                info = DISEASE_INFO.get(disease_name, DISEASE_INFO["Default"])
                render_disease_profile(disease_name, info)

            # Healthy message
            if is_healthy:
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, rgba(82,183,136,0.15), rgba(149,213,178,0.08));
                    border: 1px solid rgba(82,183,136,0.3);
                    border-radius: 20px;
                    padding: 32px;
                    text-align: center;
                    margin-top: 28px;
                ">
                    <div style="font-size:48px; margin-bottom:14px;">🌱</div>
                    <div style="font-family:'Playfair Display',serif; font-size:26px; font-weight:700;
                        color:#f8f4ec; margin-bottom:10px;">Your plant looks healthy!</div>
                    <p style="font-size:15px; color:rgba(248,244,236,0.65); max-width:500px;
                        margin:0 auto; line-height:1.7;">
                        No disease signs detected. Keep up with regular watering, adequate sunlight,
                        and periodic crop rotation to maintain plant health.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Footer note
            # Save prediction info
            st.session_state["dedicated_plant"] = plant_display_name
            st.session_state["dedicated_disease"] = disease_name
            st.session_state["dedicated_confidence"] = confidence

            
            st.markdown("""
            <div style="margin-top:48px; padding-top:24px;
                border-top: 1px solid rgba(149,213,178,0.1);
                text-align:center; font-family:'DM Mono',monospace;
                font-size:11px; color:rgba(248,244,236,0.25); letter-spacing:0.06em;">
                PlantGuard · CNN-based Crop Disease Diagnostics · For advisory use only
            </div>
            """, unsafe_allow_html=True)
        if "dedicated_plant" in st.session_state:

            st.markdown("---")
            st.subheader("🌾 Ask AgriMind")

            prompt = st.chat_input(
                "Ask anything about this disease..."
            )

            if prompt:

                with st.chat_message("user"):
                    st.write(prompt)

                response = get_chat_response(
                    plant=st.session_state["dedicated_plant"],
                    disease=st.session_state["dedicated_disease"],
                    confidence=st.session_state["dedicated_confidence"],
                    user_question=prompt,
                    thread_id=st.session_state.thread_id
                )

                with st.chat_message("assistant"):
                    st.write(response)


if __name__ == "__main__":
    main()
