import os
import streamlit as st
import PyPDF2
from dotenv import load_dotenv
from google import genai

# Cargar variables de entorno local
load_dotenv()

# Obtener la API Key de Gemini desde Secrets (Streamlit Cloud) o .env (Local)
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

# Inicializar cliente de Gemini si la clave está disponible
client = genai.Client(api_key=api_key) if api_key else None

st.set_page_config(page_title="AI Study BGA", page_icon="🤖", layout="centered")

# --- FUNCIONES AUXILIARES ---

def extract_text_from_pdf(pdf_file):
    """Extrae texto de un archivo PDF subido."""
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def safe_gemini_call(prompt):
    """Realiza la consulta al modelo de Gemini de forma segura."""
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"⚠️ **Error al generar respuesta:** {str(e)}"

def generate_summary(notes_text):
    prompt = f"Eres un profesor experto. A partir de los siguientes apuntes, genera un resumen conciso y bien estructurado destacando los conceptos clave:\n\nApuntes:\n{notes_text}"
    return safe_gemini_call(prompt)

def generate_quiz(notes_text, num_questions=5):
    prompt = f"Eres un profesor experto. Crea un cuestionario de {num_questions} preguntas de opción múltiple con 4 opciones cada una (A, B, C, D) basadas en estos apuntes. Indica al final la respuesta correcta para cada una:\n\nApuntes:\n{notes_text}"
    return safe_gemini_call(prompt)

def generate_flashcards(notes_text, num_cards=5):
    prompt = f"Crea {num_cards} fichas de estudio (flashcards) basadas en estos apuntes. Formatea cada una como:\n**Frente (Concepto/Pregunta):** ...\n**Reverso (Respuesta/Explicación):** ...\n\nApuntes:\n{notes_text}"
    return safe_gemini_call(prompt)

# --- INTERFAZ DE USUARIO ---

st.title("🤖 AI Study BGA")
st.caption("Sube tus apuntes y conviértelos en resúmenes, quizzes y tarjetas de estudio.")

if not api_key:
    st.error("⚠️ No se encontró la GEMINI_API_KEY. Configúrala en Secrets (Streamlit Cloud) o en tu archivo .env.")
    st.stop()

uploaded_file = st.file_uploader("Sube tus apuntes (PDF o TXT)", type=["pdf", "txt"])
text_input = st.text_area("O pega directamente tus notas aquí:", height=150)

notes_content = ""
if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        notes_content = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "text/plain":
        notes_content = uploaded_file.read().decode("utf-8")
elif text_input.strip():
    notes_content = text_input.strip()

if notes_content:
    st.success("✅ Apuntes cargados correctamente.")
    tab1, tab2, tab3 = st.tabs(["📌 Resumen", "❓ Quiz", "🎴 Flashcards"])

    with tab1:
        st.subheader("Resumen de apuntes")
        if st.button("Generar Resumen", type="primary"):
            with st.spinner("Procesando resumen..."):
                st.markdown(generate_summary(notes_content))

    with tab2:
        st.subheader("Cuestionario (Quiz)")
        num_q = st.slider("Número de preguntas:", min_value=1, max_value=10, value=5)
        if st.button("Generar Cuestionario", type="primary"):
            with st.spinner("Generando preguntas..."):
                st.markdown(generate_quiz(notes_content, num_questions=num_q))

    with tab3:
        st.subheader("Fichas de Estudio (Flashcards)")
        num_f = st.slider("Número de fichas:", min_value=1, max_value=10, value=5)
        if st.button("Generar Flashcards", type="primary"):
            with st.spinner("Generando flashcards..."):
                st.markdown(generate_flashcards(notes_content, num_cards=num_f))
else:
    st.info("💡 Por favor, sube un archivo o escribe tus notas arriba para empezar.")
