import os
import json
import time
import streamlit as st
import streamlit.components.v1 as components
import PyPDF2
from dotenv import load_dotenv
from google import genai

# Cargar variables de entorno
load_dotenv()

# Configuración de API Key
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

st.set_page_config(page_title="AI Study Buddy", page_icon="🤖", layout="centered")

# --- ESTADO DE SESIÓN ---
if "notes_content" not in st.session_state:
    st.session_state.notes_content = ""
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = []
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "time_limit" not in st.session_state:
    st.session_state.time_limit = 60

# --- FUNCIONES AUXILIARES ---

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def safe_gemini_call(prompt):
    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"⚠️ **Error al generar respuesta:** {str(e)}"

def generate_summary(notes_text):
    prompt = f"Eres un profesor experto. Genera un resumen conciso y bien estructurado de estos apuntes:\n\n{notes_text}"
    return safe_gemini_call(prompt)

def generate_interactive_quiz(notes_text, num_questions=5):
    prompt = f"""
    Eres un profesor experto. Basándote en el siguiente texto, genera exactamente {num_questions} preguntas de opción múltiple.
    DEBES responder ÚNICAMENTE en formato JSON válido, sin texto adicional.

    Estructura esperada:
    [
      {{
        "id": 1,
        "question": "Texto de la pregunta...",
        "options": ["Opción A", "Opción B", "Opción C", "Opción D"],
        "correct_index": 0,
        "explanation": "Explicación breve."
      }}
    ]

    Apuntes:
    {notes_text}
    """
    raw_response = safe_gemini_call(prompt)
    try:
        cleaned = raw_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return json.loads(cleaned.strip())
    except Exception:
        return None

def generate_flashcards_json(notes_text, num_cards=5):
    prompt = f"""
    Eres un profesor experto. Crea exactamente {num_cards} fichas de estudio (flashcards) sobre estos apuntes.
    DEBES responder ÚNICAMENTE en formato JSON válido.

    Estructura esperada:
    [
      {{
        "front": "Pregunta o concepto clave...",
        "back": "Respuesta o explicación detallada..."
      }}
    ]

    Apuntes:
    {notes_text}
    """
    raw_response = safe_gemini_call(prompt)
    try:
        cleaned = raw_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return json.loads(cleaned.strip())
    except Exception:
        return None

def generate_concept_map_dot(notes_text):
    """Genera código DOT (Graphviz) estructurado para dibujar el mapa conceptual."""
    prompt = f"""
    Eres un diseñador de mapas conceptuales e infografías educativas.
    Analiza el texto facilitado y genera un código Graphviz en formato DOT válido.

    Instrucciones de formato:
    - Usa 'digraph G {{ ... }}'.
    - Agrega estilo a los nodos: `node [shape=box, style="filled,rounded", color="#2b2d42", fontcolor=white, fontname="Helvetica"];`
    - Diseña relaciones conceptuales claras (Ejemplo: "Concepto A" -> "Concepto B" [label="relación"]).
    - Mantén las etiquetas breves y precisas.
    - DEVUELVE ÚNICAMENTE EL CÓDIGO DOT SIN BLOQUES MARKDOWN NI TEXTO ADICIONAL.

    Apuntes:
    {notes_text}
    """
    raw_response = safe_gemini_call(prompt)
    cleaned = raw_response.strip()
    if cleaned.startswith("```dot"):
        cleaned = cleaned[6:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()

def render_flip_card(front_text, back_text, card_id):
    card_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    .flip-card {{
      background-color: transparent;
      width: 100%;
      height: 200px;
      perspective: 1000px;
      margin-bottom: 20px;
      font-family: system-ui, -apple-system, sans-serif;
    }}

    .flip-card-inner {{
      position: relative;
      width: 100%;
      height: 100%;
      text-align: center;
      transition: transform 0.6s;
      transform-style: preserve-3d;
      cursor: pointer;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      border-radius: 12px;
    }}

    .flip-card.flipped .flip-card-inner {{
      transform: rotateY(180deg);
    }}

    .flip-card-front, .flip-card-back {{
      position: absolute;
      width: 100%;
      height: 100%;
      -webkit-backface-visibility: hidden;
      backface-visibility: hidden;
      border-radius: 12px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      padding: 20px;
      box-sizing: border-box;
    }}

    .flip-card-front {{
      background-color: #2b2d42;
      color: #edf2f4;
      border: 2px solid #8d99ae;
    }}

    .flip-card-back {{
      background-color: #d90429;
      color: #ffffff;
      transform: rotateY(180deg);
    }}

    .hint {{
      font-size: 12px;
      opacity: 0.7;
      margin-top: 10px;
    }}
    </style>
    </head>
    <body>

    <div class="flip-card" id="card-{card_id}" onclick="this.classList.toggle('flipped')">
      <div class="flip-card-inner">
        <div class="flip-card-front">
          <strong style="font-size: 16px;">{front_text}</strong>
          <span class="hint">👆 Haz clic para ver el reverso</span>
        </div>
        <div class="flip-card-back">
          <p style="font-size: 15px; margin: 0;">{back_text}</p>
          <span class="hint">🔄 Haz clic para voltear</span>
        </div>
      </div>
    </div>

    </body>
    </html>
    """
    components.html(card_html, height=220)

# --- INTERFAZ PRINCIPAL ---

st.title("🤖 AI Study Buddy (Powered by Gemini)")
st.caption("Sube tus apuntes y conviértelos en resúmenes, quizzes, tarjetas y mapas conceptuales.")

if not api_key:
    st.error("⚠️ No se encontró la GEMINI_API_KEY. Configúrala en Secrets (Streamlit Cloud) o en tu archivo .env.")
    st.stop()

# --- CARGA DE TEXTO ---
st.subheader("📄 Carga tus apuntes")
uploaded_file = st.file_uploader("Sube tus apuntes (PDF o TXT)", type=["pdf", "txt"])

with st.form("notes_form", clear_on_submit=False):
    text_input = st.text_area("O pega directamente tus notas aquí:", height=150)
    submit_text = st.form_submit_button("Procesar Texto ↵")

if submit_text or uploaded_file is not None:
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            st.session_state.notes_content = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "text/plain":
            st.session_state.notes_content = uploaded_file.read().decode("utf-8")
    elif text_input.strip():
        st.session_state.notes_content = text_input.strip()

# --- PESTAÑAS ---
if st.session_state.notes_content:
    st.success("✅ Apuntes cargados correctamente.")
    tab1, tab2, tab3, tab4 = st.tabs(["📌 Resumen", "❓ Quiz Interactivo", "🎴 Flashcards", "🗺️ Mapa Conceptual"])

    # 1. RESUMEN
    with tab1:
        st.subheader("Resumen de apuntes")
        if st.button("Generar Resumen", type="primary"):
            with st.spinner("Procesando resumen..."):
                st.markdown(generate_summary(st.session_state.notes_content))

    # 2. QUIZ INTERACTIVO
    with tab2:
        st.subheader(" Cuestionario Interactivo")
        col_q, col_t = st.columns(2)
        with col_q:
            num_q = st.slider("Número de preguntas:", min_value=1, max_value=10, value=5)
        with col_t:
            seconds_per_q = st.number_input("Segundos por pregunta:", min_value=10, max_value=120, value=30)

        if st.button("🚀 Comenzar Quiz", type="primary"):
            with st.spinner("Generando preguntas..."):
                data = generate_interactive_quiz(st.session_state.notes_content, num_questions=num_q)
                if data:
                    st.session_state.quiz_data = data
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_answers = {}
                    st.session_state.time_limit = len(data) * seconds_per_q
                    st.session_state.start_time = time.time()
                    st.rerun()

        if st.session_state.quiz_data:
            total_time = st.session_state.time_limit
            elapsed = time.time() - st.session_state.start_time if st.session_state.start_time else 0
            remaining = max(0, int(total_time - elapsed))

            if not st.session_state.quiz_submitted:
                percent_left = remaining / total_time
                st.write(f"⏱️ **Tiempo restante:** `{remaining} segundos`")

                if percent_left > 0.5:
                    st.markdown("""<style>.stProgress > div > div > div > div { background-color: #28a745 !important; }</style>""", unsafe_allow_html=True)
                elif percent_left > 0.2:
                    st.markdown("""<style>.stProgress > div > div > div > div { background-color: #ffc107 !important; }</style>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<style>.stProgress > div > div > div > div { background-color: #dc3545 !important; }</style>""", unsafe_allow_html=True)

                st.progress(percent_left)

                if remaining == 0:
                    st.warning("⏰ ¡Tiempo agotado!")
                    st.session_state.quiz_submitted = True
                    st.rerun()

            st.divider()

            with st.form("quiz_form"):
                for q_idx, q in enumerate(st.session_state.quiz_data):
                    st.write(f"**Pregunta {q_idx + 1}:** {q['question']}")
                    selected_option = st.radio(
                        label=f"Selecciona una opción ({q_idx + 1}):",
                        options=q["options"],
                        key=f"q_{q_idx}",
                        disabled=st.session_state.quiz_submitted
                    )
                    st.session_state.quiz_answers[q_idx] = selected_option
                    st.write("---")

                submit_quiz = st.form_submit_button("Enviar y Calificar 📝")
                if submit_quiz:
                    st.session_state.quiz_submitted = True
                    st.rerun()

            if st.session_state.quiz_submitted:
                correct_count = 0
                total_q = len(st.session_state.quiz_data)
                st.subheader("📊 Resultados Finales")

                for q_idx, q in enumerate(st.session_state.quiz_data):
                    user_ans = st.session_state.quiz_answers.get(q_idx)
                    correct_ans = q["options"][q["correct_index"]]

                    if user_ans == correct_ans:
                        correct_count += 1
                        st.success(f"**Pregunta {q_idx + 1}: CORRECTA ✅**")
                    else:
                        st.error(f"**Pregunta {q_idx + 1}: INCORRECTA ❌**")
                        st.write(f"👉 Tu respuesta: `{user_ans}`")
                        st.write(f"✅ Respuesta correcta: `{correct_ans}`")

                    st.info(f"💡 **Explicación:** {q['explanation']}")
                    st.write("---")

                score_percentage = (correct_count / total_q) * 100
                st.metric(label="Puntuación Final", value=f"{correct_count} / {total_q}", delta=f"{score_percentage:.1f}%")

                if score_percentage >= 70:
                    st.balloons()

    # 3. FLASHCARDS
    with tab3:
        st.subheader("🎴 Fichas de Estudio Interactivas")
        num_f = st.slider("Número de fichas:", min_value=1, max_value=10, value=5)

        if st.button("Generar Flashcards 🎴", type="primary"):
            with st.spinner("Generando tarjetas interactivas..."):
                cards_data = generate_flashcards_json(st.session_state.notes_content, num_cards=num_f)
                if cards_data:
                    st.session_state["flashcards_data"] = cards_data

        if "flashcards_data" in st.session_state:
            st.info("💡 Haz clic sobre cualquier tarjeta para voltearla y descubrir el reverso.")
            for idx, card in enumerate(st.session_state["flashcards_data"]):
                render_flip_card(card["front"], card["back"], card_id=idx)

    # 4. MAPA CONCEPTUAL (GRAPHVIZ)
    with tab4:
        st.subheader("🗺️ Mapa Conceptual de los Apuntes")
        st.caption("Genera un diagrama jerárquico automatizado para conectar las ideas principales.")

        if st.button("Generar Mapa Conceptual 🗺️", type="primary"):
            with st.spinner("Diseñando diagrama de flujo y relaciones de temas..."):
                dot_code = generate_concept_map_dot(st.session_state.notes_content)
                if dot_code:
                    st.session_state["concept_map_dot"] = dot_code

        if "concept_map_dot" in st.session_state:
            try:
                st.graphviz_chart(st.session_state["concept_map_dot"])
            except Exception as e:
                st.error("No se pudo estructurar el diagrama automáticamente. Inténtalo de nuevo.")

else:
    st.info("💡 Por favor, sube un archivo o escribe tus notas arriba para empezar.")
