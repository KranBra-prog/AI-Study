import os
import streamlit as st
import PyPDF2
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError

# Cargar variables de entorno
load_dotenv()

# Configuración de la API Key (prioriza Secrets de Streamlit Cloud, luego .env)
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

st.set_page_config(page_title="AI Study Buddy", page_icon="🤖", layout="centered")

# --- FUNCIONES AUXILIARES ---

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def safe_api_call(prompt, temperature=0.7):
    """Realiza la llamada a OpenAI manejando posibles errores de límites o saldo."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Modelo económico y rápido
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content
    except RateLimitError:
        return "⚠️ **Error de cuota/saldo:** Tu cuenta de OpenAI no tiene saldo suficiente o superó el límite de peticiones. Verifica tu facturación en [platform.openai.com/settings/billing](https://platform.openai.com/settings/billing)."
    except APIError as e:
        return f"⚠️ **Error de OpenAI:** {e.message}"
    except Exception as e:
        return f"⚠️ **Error inesperado:** {str(e)}"

def generate_summary(notes_text):
    prompt = f"You are a helpful teacher. Based on the following study notes, provide a concise, well-structured summary highlighting key concepts:\n\nNotes:\n{notes_text}"
    return safe_api_call(prompt, temperature=0.5)

def generate_quiz(notes_text, num_questions=5):
    prompt = f"You are a helpful teacher. Create {num_questions} multiple-choice questions with 4 options each (A, B, C, D) based on these notes. Return the questions in a clear, structured format indicating the correct answer:\n\nNotes:\n{notes_text}"
    return safe_api_call(prompt, temperature=0.7)

def generate_flashcards(notes_text, num_cards=5):
    prompt = f"You are a helpful study assistant. Create {num_cards} key flashcards based on these notes. Format each as:\n**Front (Concept/Question):** ...\n**Back (Answer/Explanation):** ...\n\nNotes:\n{notes_text}"
    return safe_api_call(prompt, temperature=0.6)

# --- INTERFAZ ---

st.title("🤖 AI Study Buddy")
st.caption("Upload your notes and turn them into flashcards, quiz questions, and summaries.")

if not api_key:
    st.error("⚠️ No se encontró la API Key de OpenAI. Configúrala en Secrets (Streamlit Cloud) o en tu archivo .env.")
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
