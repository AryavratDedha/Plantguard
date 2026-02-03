import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import base64
import os

# --- 1. CONFIGURATION & PAGE SETUP ---
st.set_page_config(
    page_title="PlantGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. KNOWLEDGE BASE (CONTENT MANAGEMENT) ---

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
        "Septoria Leaf Spot", "Spider Mites", "Target Spot", "Yellow Leaf Curl Virus", "Healthy"
    ],
    "wheat.keras": ["Brown Rust", "Yellow Rust", "Healthy"]
}

DISEASE_INFO = {
    "Early Blight": {
        "symptoms": ["Dark, concentric rings on older leaves.", "Yellowing tissue surrounding spots.", "Premature leaf drop."],
        "causes": ["Fungus Alternaria solani.", "Warm temperatures and high humidity.", "Overcrowded planting."],
        "prevention": ["Crop rotation (don't plant nightshades consecutively).", "Mulching to prevent soil splash.", "Drip irrigation."]
    },
    "Late Blight": {
        "symptoms": ["Large, dark brown blotches with green-gray edges.", "White fungal growth on undersides.", "Rapid rotting of fruit/tubers."],
        "causes": ["Oomycete pathogen Phytophthora infestans.", "Cool, wet weather.", "Infected seed tubers."],
        "prevention": ["Use resistant varieties.", "Apply fungicides preventatively.", "Destroy infected plant debris immediately."]
    },
    "Bacterial Spot": {
        "symptoms": ["Small, water-soaked spots on leaves.", "Spots turn brown and may fall out (shot-hole effect).", "Raised, scab-like spots on fruit."],
        "causes": ["Bacteria Xanthomonas.", "Splashing rain or overhead irrigation.", "Warm, moist conditions."],
        "prevention": ["Use disease-free seeds.", "Copper-based bactericides.", "Avoid working in wet fields."]
    },
    "Common Rust": {
        "symptoms": ["Reddish-brown pustules on both leaf surfaces.", "Leaves turning yellow and drying out."],
        "causes": ["Fungal pathogen Puccinia sorghi.", "Cool, moist weather."],
        "prevention": ["Plant resistant hybrids.", "Apply fungicides early."]
    },
    "Default": {
        "symptoms": ["Visible discoloration or lesions on leaf surface.", "Stunted growth or wilting."],
        "causes": ["Pathogens (Fungal, Bacterial, Viral) or Environmental Stress."],
        "prevention": ["Isolate affected plants.", "Ensure proper air circulation.", "Consult a local agricultural extension."]
    }
}

# --- 3. UI STYLING & ASSETS ---

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as file:
            return base64.b64encode(file.read()).decode()
    except FileNotFoundError:
        return ""

def set_custom_style(bg_image_path):
    bin_str = get_base64_image(bg_image_path)
    
    st.markdown(f"""
    <style>
    /* 1. Global Background with Dark Overlay for Readability */
    .stApp {{
        background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    
    /* 2. Typography - White for Dark Background */
    h1, h2, h3, h4, h5, h6 {{ 
        color: #ffffff !important; 
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }}
    p, label, .stMarkdown {{ 
        color: #e0e0e0 !important; 
        font-size: 16px;
    }}
    
    /* 3. Glassmorphism Cards (Semi-Transparent) */
    .glass-card {{
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 20px;
    }}
    
    /* 4. Result Status Cards (Solid Colors for Visibility) */
    .status-card {{
        padding: 25px;
        border-radius: 15px;
        margin-top: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }}
    .status-success {{ background-color: #2e7d32; color: white; border: 2px solid #a5d6a7; }}
    .status-warning {{ background-color: #ff9800; color: black; border: 2px solid #ffe0b2; }}
    .status-danger  {{ background-color: #c62828; color: white; border: 2px solid #ef9a9a; }}

    .diagnosis-title {{ font-size: 30px; font-weight: 900; text-transform: uppercase; margin: 15px 0; }}
    .confidence-text {{ font-size: 20px; font-weight: 700; opacity: 0.9; }}
    
    /* 5. Info Box (Light for readability) */
    .info-box {{
        background-color: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #1b5e20;
        margin-top: 15px;
        color: black !important;
    }}
    .info-box p, .info-box li {{ color: #1a1a1a !important; }}
    .info-box h5 {{ color: #1b5e20 !important; }}

    </style>
    """, unsafe_allow_html=True)

# --- 4. CORE ML PIPELINE ---

@st.cache_resource
def load_cached_model(path):
    if not os.path.exists(path): return None
    return load_model(path)

def preprocess_image(image_file):
    img = Image.open(image_file)
    if img.mode != "RGB": img = img.convert("RGB")
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
    top_3 = [(class_names[i], 100 * score[i]) for i in top_3_indices if i < len(class_names)]
    return top_class, confidence, top_3, top_idx

# --- 5. MAIN APPLICATION ---

def main():
    # Use the original image background
    if os.path.exists('crops.jpg'):
        set_custom_style('crops.jpg')
    else:
        st.warning("Background image 'crops.jpg' not found.")

    # Layout Structure
    col1, col2, col3 = st.columns([1, 8, 1])
    
    with col2:
        # Header - Transparent Glass
        st.title("🌿 PlantGuard")
        st.markdown("### Crop Disease Diagnosis System")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # --- INPUT SECTION ---
        row1_col1, row1_col2 = st.columns(2)
        
        with row1_col1:
            st.markdown("#### 1. Crop Identification")
            st.info("Upload a clear leaf image to identify the crop type.")
            file_plant_id = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"], key="plant")
            if file_plant_id:
                st.image(file_plant_id, caption="Crop ID Sample", use_container_width=True)

        with row1_col2:
            st.markdown("#### 2. Disease Analysis")
            st.info("Upload the affected area for diagnosis.")
            file_disease = st.file_uploader("Upload Affected Leaf", type=["jpg", "png", "jpeg"], key="disease")
            if file_disease:
                st.image(file_disease, caption="Diagnostic Sample", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # --- EXECUTION ---
        if st.button(" Run Comprehensive Analysis", type="primary", use_container_width=True):
            if not file_plant_id or not file_disease:
                st.error("⚠️ Incomplete Data: Please upload BOTH images to proceed.")
                st.stop()
            
            with st.spinner("Analyzing bio-markers & running diagnostics..."):
                try:
                    # --- STAGE 1: ROUTER ---
                    router_model = load_cached_model("healthy.keras")
                    if not router_model: 
                        st.error("❌ Critical: Router model 'healthy.keras' not found.")
                        st.stop()
                    
                    img_array_id, _ = preprocess_image(file_plant_id)
                    detected_raw, _, _, _ = predict_pipeline(router_model, img_array_id, ROUTER_CLASS_NAMES)
                    plant_display_name = PLANT_NAME_MAPPING.get(detected_raw, detected_raw.split('_')[0])
                    
                    # --- STAGE 2: DISEASE MODEL ---
                    target_model_file = MODEL_ROUTER.get(detected_raw)
                    if not target_model_file:
                        st.error(f"❌ No specialized model available for: {detected_raw}")
                        st.stop()
                        
                    disease_model = load_cached_model(target_model_file)
                    if not disease_model:
                        st.error(f"❌ Model file '{target_model_file}' is missing from server.")
                        st.stop()
                        
                    specific_classes = DISEASE_CLASSES.get(target_model_file, [])
                    if not specific_classes: specific_classes = [f"Class {i}" for i in range(10)]
                        
                    img_array_disease, original_disease_img = preprocess_image(file_disease)
                    disease_name, confidence, top_3, top_idx = predict_pipeline(disease_model, img_array_disease, specific_classes)
                    
                    # --- RESULTS DISPLAY ---
                    st.success("Analysis Complete.")
                    
                    # 1. Determine Status & Color
                    if confidence >= 70:
                        status_class = "status-success"
                        status_msg = " Strong Confidence Match"
                        text_color = "white"
                    elif 55 <= confidence < 70:
                        status_class = "status-warning"
                        status_msg = " Prediction Uncertain"
                        text_color = "black"
                    else:
                        status_class = "status-danger"
                        status_msg = " Low Confidence - Inconclusive"
                        text_color = "white"
                        disease_name = "Unidentified Issue"
                    
                    # 2. Main Result Card
                    st.markdown(f"""
                    <div class="status-card {status_class}">
                        <p style="font-weight: 800; margin-bottom: 5px; color: {text_color}; font-size: 1.2rem;">IDENTIFIED CROP: {plant_display_name}</p>
                        <div class="diagnosis-title" style="color: {text_color};">{disease_name}</div>
                        <div class="confidence-text" style="color: {text_color};">Confidence Score: {confidence:.2f}%</div>
                        <p style="font-size: 16px; margin-top: 15px; font-weight: 600; color: {text_color};">{status_msg}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 3. Visualization Section (Heatmap Removed)
                    st.markdown("---")
                    st.markdown("#### Prediction Distribution")
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    for name, prob in top_3:
                        st.write(f"**{name}**")
                        st.progress(int(prob))
                        st.caption(f"Probability: {prob:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                        
                    # 4. Disease Info Section
                    if confidence >= 55 and "Healthy" not in disease_name and disease_name != "Unidentified Issue":
                        info = DISEASE_INFO.get(disease_name, DISEASE_INFO["Default"])
                        st.markdown("---")
                        st.markdown(f"###  Disease Profile: {disease_name}")
                        
                        info_c1, info_c2, info_c3 = st.columns(3)
                        with info_c1:
                            st.markdown('<div class="info-box"><h5> Symptoms</h5>', unsafe_allow_html=True)
                            for s in info["symptoms"]: st.markdown(f"- {s}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with info_c2:
                            st.markdown('<div class="info-box"><h5> Causes</h5>', unsafe_allow_html=True)
                            for c in info["causes"]: st.markdown(f"- {c}")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                        with info_c3:
                            st.markdown('<div class="info-box"><h5> Prevention</h5>', unsafe_allow_html=True)
                            for p in info["prevention"]: st.markdown(f"- {p}")
                            st.markdown('</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"An internal error occurred: {e}")
        

if __name__ == "__main__":
    main()