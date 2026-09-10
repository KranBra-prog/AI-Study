import os
import streamlit as st
import PyPDF2
from dotenv import load_dotenv
from openai import OpenAI

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Inicializar el cliente de OpenAI
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

# Configuración de la página en Streamlit
st.set_page_config(page_title="AI Study Buddy", page_icon="🤖", layout="centered")

# --- FUNCIONES AUXILIARES ---

def extract_text_from_pdf(pdf_file):
    """Extrae todo el texto de un archivo PDF subido."""
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def generate_summary(notes_text):
    """Genera un resumen estructurado a partir de los apuntes."""
    prompt = f"""You are a helpful teacher. Based on the following study notes, provide a concise, well-structured summary highlighting key concepts and main takeaways:

Notes:
{notes_text}
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content

def generate_quiz(notes_text, num_questions=5):
    """Genera un cuestionario de opción múltiple basado en los apuntes."""
    prompt = f"""You are a helpful teacher.
Based on the following notes, create {num_questions} multiple-choice questions with 4 options each (A, B, C, D).
Return the questions in a clear, structured format. Make sure to indicate the correct answer at the end of each question.

Notes:
{notes_text}
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

def generate_flashcards(notes_text, num_cards=5):
    """Genera fichas de estudio (Flashcards) en formato Término / Definición."""
    prompt = f"""You are a helpful study assistant. Create {num_cards} key flashcards based on the following notes.
Format each flashcard clearly as:
**Front (Concept/Question):** [Concept or Question]
**Back (Answer/Explanation):** [Short Explanation]

Notes:
{notes_text}
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6
    )
    return response.choices[0].message.content


# --- INTERFAZ DE USUARIO CON STREAMLIT ---

st.title("🤖 AI Study Buddy")
st.caption("Upload your notes and turn them into flashcards, quiz questions, and summaries.")

# Verificación de la API Key
if not api_key:
    st.error("⚠️ No se encontró la API Key de OpenAI. Asegúrate de definir OPENAI_API_KEY en tu archivo .env.")
    st.stop()

# Área de carga de archivos / texto
uploaded_file = st.file_uploader("Sube tus apuntes (PDF o TXT)", type=["pdf", "txt"])
text_input = st.text_area("O pega directamente tus notas aquí:", height=150)

notes_content = ""

# Determinar la fuente del texto
if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        notes_content = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "text/plain":
        notes_content = uploaded_file.read().decode("utf-8")
elif text_input.strip():
    notes_content = text_input.strip()

# Pestañas para las funcionalidades
if notes_content:
    st.success("✅ Apuntes cargados correctamente.")
    
    tab1, tab2, tab3 = st.tabs(["📌 Resumen", "❓ Quiz", "🎴 Flashcards"])

    with tab1:
        st.subheader("Resumen de apuntes")
        if st.button("Generar Resumen", type="primary"):
            with st.spinner("Procesando resumen..."):
                summary = generate_summary(notes_content)
                st.markdown(summary)

    with tab2:
        st.subheader("Cuestionario (Quiz)")
        num_q = st.slider("Número de preguntas:", min_value=1, max_value=10, value=5)
        if st.button("Generar Cuestionario", type="primary"):
            with st.spinner("Generando preguntas..."):
                quiz = generate_quiz(notes_content, num_questions=num_q)
                st.markdown(quiz)

    with tab3:
        st.subheader("Fichas de Estudio (Flashcards)")
        num_f = st.slider("Número de fichas:", min_value=1, max_value=10, value=5)
        if st.button("Generar Flashcards", type="primary"):
            with st.spinner("Generando flashcards..."):
                flashcards = generate_flashcards(notes_content, num_cards=num_f)
                st.markdown(flashcards)
else:
    st.info("💡 Por favor, sube un archivo o escribe tus notas arriba para empezar.")
    