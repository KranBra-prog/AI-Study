import os
import json
import time
import streamlit as st
import PyPDF2
from dotenv import load_dotenv
from google import genai

# Cargar variables de entorno
load_dotenv()

# Configuración de API Key
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

st.set_page_config(page_title="AI Study BGA", page_icon="🤖", layout="centered")

# --- INICIALIZACIÓN DEL ESTADO DE SESIÓN (st.session_state) ---
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
    st.session_state.time_limit = 60  # Tiempo por defecto en segundos

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

def generate_flashcards(notes_text, num_cards=5):
    prompt = f"Crea {num_cards} fichas de estudio (flashcards) basadas en estos apuntes. Formatea cada una como:\n**Frente (Concepto/Pregunta):** ...\n**Reverso (Respuesta/Explicación):** ...\n\n{notes_text}"
    return safe_gemini_call(prompt)

def generate_interactive_quiz(notes_text, num_questions=5):
    """Solicita a Gemini un JSON estricto con las preguntas, opciones, respuesta correcta y explicación."""
    prompt = f"""
    Eres un profesor experto. Basándote en el siguiente texto, genera exactamente {num_questions} preguntas de opción múltiple.
    DEBES responder ÚNICAMENTE en formato JSON válido, sin texto adicional ni bloques de formato markdown innecesarios.

    Estructura esperada del JSON:
    [
      {{
        "id": 1,
        "question": "Texto de la pregunta...",
        "options": ["Opción A", "Opción B", "Opción C", "Opción D"],
        "correct_index": 0,
        "explanation": "Explicación breve de por qué esta opción es la correcta."
      }}
    ]

    Apuntes:
    {notes_text}
    """
    raw_response = safe_gemini_call(prompt)
    try:
        # Limpiar posibles marcadores de bloque de código json
        cleaned = raw_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return json.loads(cleaned.strip())
    except Exception as e:
        st.error("Error al procesar el formato del cuestionario. Reintentando...")
        return None

# --- INTERFAZ PRINCIPAL ---

st.title("🤖 AI Study BGA (Powered by Gemini)")
st.caption("Sube tus apuntes y pon a prueba tus conocimientos de forma interactiva.")

if not api_key:
    st.error("⚠️ No se encontró la GEMINI_API_KEY. Configúrala en Secrets (Streamlit Cloud) o en tu archivo .env.")
    st.stop()

# --- CARGA DE TEXTO CON BOTÓN ENTER ---
st.subheader("📄 Carga tus apuntes")

uploaded_file = st.file_uploader("Sube tus apuntes (PDF o TXT)", type=["pdf", "txt"])

with st.form("notes_form", clear_on_submit=False):
    text_input = st.text_area("O pega directamente tus notas aquí:", height=150)
    # Al presionar Enter estando dentro del formulario o dar clic al botón, se procesa el texto
    submit_text = st.form_submit_button("Procesar Texto ↵")

if submit_text or uploaded_file is not None:
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            st.session_state.notes_content = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "text/plain":
            st.session_state.notes_content = uploaded_file.read().decode("utf-8")
    elif text_input.strip():
        st.session_state.notes_content = text_input.strip()

# --- PESTAÑAS DE TRABAJO ---
if st.session_state.notes_content:
    st.success("✅ Apuntes cargados correctamente.")
    tab1, tab2, tab3 = st.tabs(["📌 Resumen", "❓ Quiz Interactivo", "🎴 Flashcards"])

    # 1. RESUMEN
    with tab1:
        st.subheader("Resumen de apuntes")
        if st.button("Generar Resumen", type="primary"):
            with st.spinner("Procesando resumen..."):
                st.markdown(generate_summary(st.session_state.notes_content))

    # 2. QUIZ INTERACTIVO PARA ALUMNOS
    with tab2:
        st.subheader(" Cuestionario Interactivo")

        # Controles de configuración antes de iniciar
        col_q, col_t = st.columns(2)
        with col_q:
            num_q = st.slider("Número de preguntas:", min_value=1, max_value=10, value=5)
        with col_t:
            seconds_per_q = st.number_input("Segundos por pregunta:", min_value=10, max_value=120, value=30)

        if st.button("🚀 Comenzar Quiz", type="primary"):
            with st.spinner("Generando preguntas y opciones..."):
                data = generate_interactive_quiz(st.session_state.notes_content, num_questions=num_q)
                if data:
                    st.session_state.quiz_data = data
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_answers = {}
                    st.session_state.time_limit = len(data) * seconds_per_q
                    st.session_state.start_time = time.time()
                    st.rerun()

        # RENDERIZADO DEL QUIZ EN CURSO
        if st.session_state.quiz_data:
            total_time = st.session_state.time_limit
            elapsed = time.time() - st.session_state.start_time if st.session_state.start_time else 0
            remaining = max(0, int(total_time - elapsed))

            # TEMPORIZADOR Y BARRA DE PROGRESO TRICOLOR (Verde / Amarillo / Rojo)
            if not st.session_state.quiz_submitted:
                percent_left = remaining / total_time
                st.write(f"⏱️ **Tiempo restante:** `{remaining} segundos`")

                # Lógica del color de la barra de progreso
                if percent_left > 0.5:
                    st.markdown("""<style>.stProgress > div > div > div > div { background-color: #28a745 !important; }</style>""", unsafe_allow_html=True)
                elif percent_left > 0.2:
                    st.markdown("""<style>.stProgress > div > div > div > div { background-color: #ffc107 !important; }</style>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<style>.stProgress > div > div > div > div { background-color: #dc3545 !important; }</style>""", unsafe_allow_html=True)

                st.progress(percent_left)

                if remaining == 0:
                    st.warning("⏰ ¡Se agotó el tiempo! Evaluando respuestas hasta el momento...")
                    st.session_state.quiz_submitted = True
                    st.rerun()

            st.divider()

            # FORMULARIO DE RESPUESTAS
            with st.form("quiz_form"):
                for q_idx, q in enumerate(st.session_state.quiz_data):
                    st.write(f"**Pregunta {q_idx + 1}:** {q['question']}")
                    selected_option = st.radio(
                        label=f"Selecciona una opción para la pregunta {q_idx + 1}:",
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

            # MOSTRAR RESULTADOS Y RETROALIMENTACIÓN
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

                # VISUALIZACIÓN DE PUNTAJE FINAL
                score_percentage = (correct_count / total_q) * 100
                st.metric(label="Puntuación Final", value=f"{correct_count} / {total_q}", delta=f"{score_percentage:.1f}%")

                if score_percentage >= 70:
                    st.balloons()
                    st.success("🎉 ¡Excelente trabajo! Has demostrado una gran comprensión de los temas.")
                else:
                    st.warning("📚 Te sugerimos repasar los puntos clave y volver a intentarlo.")

    # 3. FLASHCARDS
    with tab3:
        st.subheader("Fichas de Estudio (Flashcards)")
        num_f = st.slider("Número de fichas:", min_value=1, max_value=10, value=5)
        if st.button("Generar Flashcards", type="primary"):
            with st.spinner("Generando flashcards..."):
                st.markdown(generate_flashcards(st.session_state.notes_content, num_cards=num_f))

else:
    st.info("💡 Por favor, sube un archivo o escribe tus notas arriba y presiona 'Procesar Texto' para empezar.")
