import streamlit as st
import numpy as np
import pandas as pd
import re
from collections import defaultdict

st.set_page_config(
    page_title="Generador Secuencial LSTM",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background-color: #FAF8F5;
        color: #1C1917;
    }

    [data-testid="stSidebar"] {
        background-color: #F3EFEA;
        border-right: 1px solid #E5DEC9;
    }

    .hero-lstm {
        background: linear-gradient(135deg, #292524 0%, #44403C 100%);
        color: #FAF8F5;
        padding: 28px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        border-bottom: 4px solid #D97757;
    }

    .output-box {
        background-color: #FFFFFF;
        border: 1px solid #E7E5E4;
        border-left: 5px solid #D97757;
        border-radius: 12px;
        padding: 22px;
        font-size: 1.05rem;
        line-height: 1.6;
        box-shadow: 0 4px 14px rgba(0,0,0,0.04);
    }

    .stButton>button {
        background-color: #D97757 !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #C26143 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-lstm">
    <h2 style="margin:0; color:white;">Modelado Secuencial y Generación de Texto (LSTM & N-Gram Context)</h2>
    <p style="margin:6px 0 0 0; color:#D6D3D1;">Predicción estocástica de secuencias de palabras manteniendo dependencias de contexto previo y control de temperatura.</p>
</div>
""", unsafe_allow_html=True)

CORPUS_PREDEFINIDOS = {
    "Inteligencia Artificial y Redes Neuronales": (
        "las redes neuronales recurrentes como las lstm permiten modelar secuencias de texto manteniendo informacion "
        "relevante del contexto previo gracias a su memoria a largo plazo pueden generar texto coherente y fluido "
        "los modelos de lenguaje aprenden representaciones vectoriales llamadas embeddings donde las palabras con "
        "significados similares se agrupan en el espacio dimensional el procesamiento de lenguaje natural transforma "
        "la comunicacion entre humanos y maquinas permitiendo asistentes virtuales traduccion automatica y analisis "
        "de sentimientos en tiempo real las puertas de olvido y entrada en una celda lstm regulan el flujo de gradientes"
    ),
    "Internet de las Cosas y Sistemas Ciberfísicos": (
        "el internet de las cosas conecta sensores y actuadores a traves del protocolo mqtt permitiendo la telemetria "
        "en tiempo real los microcontroladores esp32 envian lecturas de temperatura y humedad hacia la nube donde "
        "los sistemas ciberfisicos procesan los datos para automatizar el hogar y la industria mediante comandos "
        "remotos es posible controlar servomotores y luces led optimizando el consumo energetico y la eficiencia"
    ),
    "Literatura Fantástica": (
        "en el antiguo reino de las cumbres doradas los viajeros buscaban el cristal de fuego que dormia bajo la "
        "montana silenciosa cada noche el viento traia ecos de viejas leyendas sobre dragones y guardianes del bosque "
        "cuando la luna alcanzaba el cenit las puertas del templo revelaban el mapa secreto hacia las tierras del norte"
    )
}

with st.sidebar:
    st.subheader("Parámetros del Modelo")
    fuente_corpus = st.selectbox("Corpus de Entrenamiento:", list(CORPUS_PREDEFINIDOS.keys()) + ["Texto personalizado"])
    ventana_contexto = st.slider("Ventana de contexto (Palabras previas):", 1, 3, 2)
    longitud_gen = st.slider("Palabras a generar:", 10, 80, 30)
    temperatura = st.slider("Temperatura (Creatividad):", 0.2, 2.0, 0.8, 0.1)

col1, col2 = st.columns([1, 1.15], gap="large")

with col1:
    st.subheader("1. Corpus y Semilla Inicial")
    if fuente_corpus == "Texto personalizado":
        texto_base = st.text_area("Escribe o pega tu corpus de entrenamiento:", value=CORPUS_PREDEFINIDOS["Inteligencia Artificial y Redes Neuronales"], height=170)
    else:
        texto_base = st.text_area("Corpus activo (editable):", value=CORPUS_PREDEFINIDOS[fuente_corpus], height=170)

    tokens = re.findall(r'\b\w+\b', texto_base.lower())
    vocabulario = sorted(list(set(tokens)))

    st.caption(f"Tokens totales: **{len(tokens)}** | Vocabulario único: **{len(vocabulario)}** palabras.")

    sugerencia_semilla = " ".join(tokens[:ventana_contexto]) if len(tokens) >= ventana_contexto else "las redes"
    semilla = st.text_input("Palabras semilla para iniciar:", value=sugerencia_semilla)

with col2:
    st.subheader("2. Generación y Distribución Softmax")
    if st.button("Generar Secuencia"):
        if len(tokens) < 5:
            st.warning("El corpus es muy corto. Agrega más texto.")
        else:
            transiciones = defaultdict(lambda: defaultdict(int))
            for i in range(len(tokens) - ventana_contexto):
                estado = tuple(tokens[i:i + ventana_contexto])
                sig = tokens[i + ventana_contexto]
                transiciones[estado][sig] += 1

            palabras_actuales = re.findall(r'\b\w+\b', semilla.lower())
            if not palabras_actuales:
                palabras_actuales = tokens[:ventana_contexto]

            resultado = list(palabras_actuales)
            distribucion_inicial = {}

            for paso in range(longitud_gen):
                estado_actual = tuple(resultado[-ventana_contexto:])
                candidatos = transiciones.get(estado_actual, None)

                if not candidatos:
                    unigramas = defaultdict(int)
                    for i in range(len(tokens) - 1):
                        if tokens[i] == resultado[-1]:
                            unigramas[tokens[i + 1]] += 1
                    candidatos = unigramas if unigramas else {np.random.choice(vocabulario): 1}

                palabras_cand = list(candidatos.keys())
                conteos = np.array(list(candidatos.values()), dtype=np.float64)

                logits = np.log(conteos + 1e-9) / temperatura
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / np.sum(exp_logits)

                if paso == 0:
                    for p_cand, pr in zip(palabras_cand, probs):
                        distribucion_inicial[p_cand] = float(pr)

                elegida = np.random.choice(palabras_cand, p=probs)
                resultado.append(elegida)

            texto_generado = " ".join(resultado).capitalize() + "."
            st.markdown(f"""
            <div class="output-box">
                <strong>Secuencia generada:</strong><br><br>
                "{texto_generado}"
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            st.download_button(
                label="Descargar texto generado",
                data=texto_generado,
                file_name="texto_lstm.txt",
                mime="text/plain"
            )

            st.write("")
            st.markdown("**Probabilidades del siguiente token (Capa Softmax inicial):**")
            if distribucion_inicial:
                df_probs = pd.DataFrame(
                    list(distribucion_inicial.items()),
                    columns=["Palabra Candidata", "Probabilidad"]
                ).sort_values(by="Probabilidad", ascending=False).head(8)
                st.bar_chart(df_probs.set_index("Palabra Candidata"))
