# app2.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch
import time
import numpy as np
import plotly.express as px

# ---------------------------
# Configuración de la página
# ---------------------------
st.set_page_config(
    page_title="Detección de Enfermedades Oculares",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------
# Ruta al modelo entrenado
# ---------------------------
MODEL_PATH = r"C:\proyecto grafica off 3\runs\classify\modelo_enfermedades_ojo\weights\best.pt"

# ---------------------------
# Descripciones
# ---------------------------
descripciones = {
    "ojo sano": "El ojo no presenta signos visibles de enfermedad. Mantener buenos hábitos y controles periódicos ayuda a conservar la salud visual.",
    "ojo conjuntivitis": "La conjuntivitis es una inflamación de la conjuntiva, causando enrojecimiento, picor, molestias y secreción. Puede ser infecciosa, alérgica o irritativa.",
    "ojo ictericia": "La ictericia se manifiesta como una coloración amarilla en la parte blanca del ojo, usualmente asociada a alteraciones hepáticas o metabólicas. Es recomendable consultar al especialista.",
    "ojo catarata": "La catarata ocurre por opacificación del cristalino, provocando visión borrosa progresiva. Es más frecuente en adultos mayores y se trata generalmente con cirugía.",
    "ojo pterigion": "El pterigion es un crecimiento de tejido sobre la conjuntiva que puede avanzar hacia la córnea. Afecta la visión y puede causar irritación o molestias."
}

# ---------------------------
# Estilo CSS
# ---------------------------
st.markdown("""
<style>
.stApp { background-color: #0e1117; color: #e6eef8; }
.header {
    background: linear-gradient(90deg,#00b4db,#0083b0);
    border-radius: 12px;
    padding: 28px;
    text-align: center;
    margin-bottom: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.6);
}
.header h1 { color: white; font-weight: 800; margin:0; font-size: 2.3rem; }
.subtitle { color: #dff6ff; margin-top:6px; opacity:0.95; }
.result-card {
    background: #0f1720;
    border-radius: 10px;
    padding: 18px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.5);
}
.small-muted { color:#9aa6b2; font-size:0.95rem; }
.uploaded-img {
    border-radius: 12px;
    box-shadow: 0 0 18px rgba(0,0,0,0.5);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Encabezado
# ---------------------------
st.markdown(
    "<div class='header'><h1>👁️ Detección de Enfermedades Oculares</h1>"
    "<div class='subtitle'>Subí una imagen o tomá una foto con la cámara — el modelo clasificará entre ojo sano, conjuntivitis, ictericia, catarata o pterigion.</div></div>",
    unsafe_allow_html=True,
)

# ---------------------------
# Cargar modelo
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        return {"error": str(e)}

model_obj = load_model(MODEL_PATH)
if isinstance(model_obj, dict) and "error" in model_obj:
    st.error(f"❌ Error al cargar el modelo: {model_obj['error']}")
    st.stop()
model = model_obj

# ---------------------------
# Seleccionar método de entrada
# ---------------------------
st.subheader("📸 Elegí cómo subir la imagen")
modo = st.radio(
    "Seleccioná una opción:",
    ("Subir desde archivo", "Tomar foto con cámara"),
    horizontal=True
)

uploaded_file = None

if modo == "Subir desde archivo":
    uploaded_file = st.file_uploader("📤 Arrastrá o seleccioná una imagen del ojo", type=["jpg", "jpeg", "png"])
elif modo == "Tomar foto con cámara":
    uploaded_file = st.camera_input("📷 Capturá una foto del ojo")

# ---------------------------
# Mostrar imagen
# ---------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="🖼️ Imagen capturada", width=320, output_format="PNG")
    st.success("✅ Imagen lista para analizar.")

analyze = st.button("🔍 Analizar imagen")

# ---------------------------
# Funciones auxiliares
# ---------------------------
def parse_results(results):
    names_raw = model.names
    if isinstance(names_raw, dict):
        ordered = [names_raw[i] for i in sorted(names_raw.keys())]
    else:
        ordered = list(names_raw)
    try:
        probs = results[0].probs.data.cpu().numpy().astype(float)
    except Exception:
        probs = np.zeros(len(ordered), dtype=float)
    return ordered, probs


def simple_loading_animation(duration=1.2):
    placeholder = st.empty()
    end = time.time() + duration
    dots = 0
    while time.time() < end:
        dots = (dots + 1) % 4
        placeholder.markdown(f"**Analizando imagen**{' .' * dots}")
        time.sleep(0.3)
    placeholder.empty()

# ---------------------------
# Análisis
# ---------------------------
if uploaded_file is not None and analyze:
    try:
        simple_loading_animation(1.2)
        with st.spinner("Procesando la imagen con el modelo..."):
            results = model.predict(image, verbose=False)

        labels, probs = parse_results(results)
        if len(labels) == 0 or probs.sum() == 0:
            st.warning("No se pudieron obtener predicciones válidas.")
        else:
            pred_idx = int(np.argmax(probs))
            pred_label = labels[pred_idx]
            confidence = float(probs[pred_idx] * 100.0)

            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.markdown(f"### 🩺 Resultado: **{pred_label}**")
                st.markdown(f"**Confianza:** {confidence:.2f}%")

                desc = descripciones.get(pred_label.lower().strip(), "Descripción no disponible.")
                st.markdown(f"**Descripción:** {desc}")
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                fig = px.bar(
                    x=probs,
                    y=labels,
                    orientation="h",
                    labels={"x": "Probabilidad", "y": ""},
                    title="Distribución de probabilidades",
                    width=800,
                    height=360,
                )
                fig.update_layout(
                    margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                fig.update_xaxes(range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error procesando la imagen: {e}")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#9aa6b2'>© 2025 — Detección de Enfermedades Oculares · DarkPapaletax</div>",
    unsafe_allow_html=True,
)
