import os
import json
import time
import asyncio
from io import BytesIO
import streamlit as st
import streamlit.components.v1 as components
import pypdf
from dotenv import load_dotenv
from google import genai
import edge_tts

# Cargar variables de entorno
load_dotenv()

# Configuración de API Key
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="AI Study Buddy",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- INYECCIÓN DE CSS AVANZADO ---
st.markdown("""
    <style>
    /* Importar fuente moderna */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Estilo del Header */
    .header-container {
        text-align: center;
        padding: 1.5rem 1rem;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .header-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-top: 5px;
    }

    /* Tarjetas de Métricas y Contenedores */
    .metric-box {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 12px;
        text-align: center;
    }

    /* Botones y Cuestionarios */
    .stButton>button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
    }

    /* Pestañas estilizadas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
    }

    /* Ajustes móviles */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 0.8rem !important;
        }
        .header-title { font-size: 1.7rem; }
    }
    </style>
""", unsafe_allow_html=True)

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
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "summary_text" not in st.session_state:
    st.session_state.summary_text = ""
if "voice_enabled" not in st.session_state:
    st.session_state.voice_enabled = True
if "card_index" not in st.session_state:
    st.session_state.card_index = 0

# --- FUNCIONES AUXILIARES ---

def extract_text_from_pdf(pdf_file):
    """Extrae texto de PDF."""
    try:
        reader = pypdf.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    except pypdf.errors.PdfReadError:
        st.error("⚠️ El PDF está protegido con contraseña o dañado.")
        return ""
    except Exception as e:
        st.error(f"⚠️ Error al procesar el PDF: {e}")
        return ""

def safe_gemini_call(prompt):
    """Llamadas robustas con respaldo a Gemini."""
    models_to_try = ['gemini-2.5-flash', 'gemini-1.5-flash']
    for model_name in models_to_try:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                if response and response.text:
                    return response.text
            except Exception as e:
                error_str = str(e)
                if "503" in error_str or "UNAVAILABLE" in error_str:
                    time.sleep(1.5 * (attempt + 1))
                else:
                    break
    return "⚠️ **El servicio está experimentando alta demanda.** Reintenta en unos momentos."

def text_to_speech_bytes(text, voice="es-ES-AlvaroNeural", rate="+40%"):
    """Genera audio sintético en streaming."""
    try:
        clean_text = text.replace("*", "").replace("#", "").replace("`", "")
        async def _generate_audio():
            communicate = edge_tts.Communicate(clean_text, voice=voice, rate=rate)
            fp = BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    fp.write(chunk["data"])
            fp.seek(0)
            return fp
        return asyncio.run(_generate_audio())
    except Exception:
        return None

def generate_summary(notes_text):
    prompt = f"Eres un profesor experto. Crea un resumen didáctico, bien estructurado con títulos claros, listas y emojis clave:\n\n{notes_text}"
    return safe_gemini_call(prompt)

def generate_interactive_quiz(notes_text, num_questions=5):
    prompt = f"""
    Eres un profesor experto. Basándote en el siguiente texto, genera exactamente {num_questions} preguntas de opción múltiple.
    DEBES responder ÚNICAMENTE en formato JSON válido, sin Markdown adicional.

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
    Eres un profesor experto. Crea exactamente {num_cards} fichas de estudio (flashcards) clave sobre estos apuntes.
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

def render_flip_card(front_text, back_text, card_id):
    card_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
    .flip-card {{
      background-color: transparent;
      width: 100%;
      height: 240px;
      perspective: 1000px;
      margin: 10px 0;
      font-family: 'Inter', system-ui, sans-serif;
    }}
    .flip-card-inner {{
      position: relative;
      width: 100%;
      height: 100%;
      text-align: center;
      transition: transform 0.6s cubic-bezier(0.4, 0.2, 0.2, 1);
      transform-style: preserve-3d;
      cursor: pointer;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
      border-radius: 16px;
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
      border-radius: 16px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      padding: 20px;
      box-sizing: border-box;
    }}
    .flip-card-front {{
      background: linear-gradient(145deg, #1e293b, #0f172a);
      color: #f8fafc;
      border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    .flip-card-back {{
      background: linear-gradient(145deg, #0284c7, #0369a1);
      color: #ffffff;
      transform: rotateY(180deg);
    }}
    .hint {{
      font-size: 12px;
      opacity: 0.75;
      margin-top: 15px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    </style>
    </head>
    <body>
    <div class="flip-card" id="card-{card_id}" onclick="this.classList.toggle('flipped')">
      <div class="flip-card-inner">
        <div class="flip-card-front">
          <strong style="font-size: 17px; line-height: 1.4;">{front_text}</strong>
          <span class="hint">👆 Toca para ver la respuesta</span>
        </div>
        <div class="flip-card-back">
          <p style="font-size: 15px; line-height: 1.4; margin: 0;">{back_text}</p>
          <span class="hint">🔄 Toca para volver a la pregunta</span>
        </div>
      </div>
    </div>
    </body>
    </html>
    """
    components.html(card_html, height=265)

def render_certificate(student_name, score, total):
    cert_html = f"""
    <div style="border: 4px double #38bdf8; padding: 25px; text-align: center; background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-radius: 16px; margin-top: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.5);">
        <h2 style="color: #38bdf8; font-size: 1.5rem; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px;">📜 Certificado de Logro</h2>
        <p style="font-size: 14px; color: #94a3b8;">Otorgado con orgullo a:</p>
        <h3 style="color: #f8fafc; font-size: 1.6rem; margin: 15px 0; border-bottom: 2px solid #38bdf8; display: inline-block; padding-bottom: 5px;">{student_name}</h3>
        <p style="font-size: 14px; color: #94a3b8;">Por haber completado satisfactoriamente la evaluación sobre los apuntes de estudio.</p>
        <div style="font-size: 18px; font-weight: bold; color: #4ade80; margin: 15px 0; background: rgba(74, 222, 128, 0.1); padding: 8px; border-radius: 8px; display: inline-block;">
            Puntuación Obtenida: {score} / {total} ({(score/total)*100:.0f}%)
        </div>
        <p style="font-size: 11px; color: #64748b; margin-top: 15px;">Emitido automáticamente por AI Study Buddy</p>
    </div>
    """
    st.markdown(cert_html, unsafe_allow_html=True)

# --- CABECERA PRINCIPAL ---
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🎓 AI Study Buddy</h1>
        <div class="header-subtitle">Tu tutor personal inteligente de estudio paso a paso</div>
    </div>
""", unsafe_allow_html=True)

if not api_key:
    st.error("⚠️ No se encontró la GEMINI_API_KEY. Configúrala en Secrets o en tu archivo .env.")
    st.stop()

# --- PANEL DE CARGA ---
with st.expander("📄 **Cargar o Cambiar Apuntes**", expanded=not bool(st.session_state.notes_content)):
    uploaded_file = st.file_uploader("Sube tus apuntes (PDF o TXT)", type=["pdf", "txt"])
    
    with st.form("notes_form", clear_on_submit=False):
        text_input = st.text_area("O pega el texto directamente aquí:", height=120)
        submit_text = st.form_submit_button("Procesar Apuntes 🚀", use_container_width=True)

    if submit_text or uploaded_file is not None:
        previous_content = st.session_state.notes_content
        if uploaded_file is not None:
            if uploaded_file.type == "application/pdf":
                st.session_state.notes_content = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "text/plain":
                st.session_state.notes_content = uploaded_file.read().decode("utf-8")
        elif text_input.strip():
            st.session_state.notes_content = text_input.strip()

        if previous_content != st.session_state.notes_content and st.session_state.notes_content:
            st.session_state.chat_messages = [
                {"role": "assistant", "content": "¡Hola! 👋 Ya he analizado tus apuntes. ¿En qué tema o concepto quieres profundizar hoy?"}
            ]
            st.session_state.summary_text = ""
            st.session_state.quiz_data = []
            st.session_state.card_index = 0
            st.rerun()

# --- MÉTRICAS DE CONTENIDO & PESTAÑAS ---
if st.session_state.notes_content:
    words_count = len(st.session_state.notes_content.split())
    read_time = max(1, round(words_count / 200))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Palabras", f"{words_count:,}")
    with col2:
        st.metric("Caracteres", f"{len(st.session_state.notes_content):,}")
    with col3:
        st.metric("Lectura Estimada", f"~{read_time} min")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 Resumen", 
        "❓ Evaluación", 
        "🎴 Flashcards", 
        "💬 Tutor Interactivo"
    ])

    # 1. RESUMEN
    with tab1:
        st.subheader("Resumen Inteligente")
        if st.button("Generar Resumen Estructurado 📝", type="primary", use_container_width=True):
            with st.spinner("Sintetizando información clave..."):
                st.session_state.summary_text = generate_summary(st.session_state.notes_content)

        if st.session_state.summary_text:
            st.markdown(st.session_state.summary_text)
            st.download_button(
                label="📥 Descargar Resumen (.md)",
                data=st.session_state.summary_text,
                file_name="Resumen_Estudio.md",
                mime="text/markdown",
                use_container_width=True
            )

    # 2. CUESTIONARIO INTERACTIVO
    with tab2:
        st.subheader("Cuestionario Autoevaluativo")
        c1, c2 = st.columns(2)
        with c1:
            num_q = st.slider("Preguntas:", min_value=3, max_value=10, value=5)
        with c2:
            seconds_per_q = st.number_input("Segundos por pregunta:", min_value=10, max_value=120, value=30)

        if st.button("🚀 Comenzar Evaluación", type="primary", use_container_width=True):
            with st.spinner("Diseñando preguntas adaptadas..."):
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
                st.write(f"⏱️ **Tiempo Restante:** `{remaining} seg`")
                st.progress(percent_left)

                if remaining == 0:
                    st.warning("⏰ ¡Tiempo agotado! Evaluando respuestas...")
                    st.session_state.quiz_submitted = True
                    st.rerun()

            st.divider()

            with st.form("quiz_form"):
                for q_idx, q in enumerate(st.session_state.quiz_data):
                    st.write(f"**{q_idx + 1}. {q['question']}**")
                    selected_option = st.radio(
                        label=f"Opciones ({q_idx + 1}):",
                        options=q["options"],
                        key=f"q_{q_idx}",
                        disabled=st.session_state.quiz_submitted,
                        label_visibility="collapsed"
                    )
                    st.session_state.quiz_answers[q_idx] = selected_option
                    st.write("---")

                submit_quiz = st.form_submit_button("Entregar Cuestionario 📝", use_container_width=True)
                if submit_quiz:
                    st.session_state.quiz_submitted = True
                    st.rerun()

            if st.session_state.quiz_submitted:
                correct_count = 0
                total_q = len(st.session_state.quiz_data)
                st.subheader("📊 Resultados de la Evaluación")

                for q_idx, q in enumerate(st.session_state.quiz_data):
                    user_ans = st.session_state.quiz_answers.get(q_idx)
                    correct_ans = q["options"][q["correct_index"]]

                    if user_ans == correct_ans:
                        correct_count += 1
                        st.success(f"**Pregunta {q_idx + 1}: Correcta ✅**")
                    else:
                        st.error(f"**Pregunta {q_idx + 1}: Incorrecta ❌**")
                        st.write(f"👉 Tu respuesta: `{user_ans}`")
                        st.write(f"✅ Respuesta correcta: `{correct_ans}`")

                    st.info(f"💡 **Explicación:** {q['explanation']}")
                    st.write("---")

                score_percentage = (correct_count / total_q) * 100
                st.metric(label="Puntuación Final", value=f"{correct_count} / {total_q}", delta=f"{score_percentage:.0f}%")

                if score_percentage >= 70:
                    st.balloons()
                    student_name = st.text_input("Ingresa tu nombre para el certificado:", value="Estudiante")
                    if student_name:
                        render_certificate(student_name, correct_count, total_q)

    # 3. FLASHCARDS CON NAVEGACIÓN PASO A PASO
    with tab3:
        st.subheader("🎴 Modo de Estudio Interactivo")
        if "flashcards_data" not in st.session_state:
            num_f = st.slider("Número de tarjetas a generar:", min_value=3, max_value=10, value=5)
            if st.button("Generar Mazo de Flashcards 🎴", type="primary", use_container_width=True):
                with st.spinner("Creando tarjetas clave..."):
                    cards_data = generate_flashcards_json(st.session_state.notes_content, num_cards=num_f)
                    if cards_data:
                        st.session_state["flashcards_data"] = cards_data
                        st.session_state.card_index = 0
                        st.rerun()
        else:
            cards = st.session_state["flashcards_data"]
            idx = st.session_state.card_index
            
            # Indicador de progreso
            st.progress((idx + 1) / len(cards))
            st.caption(f"Tarjeta **{idx + 1}** de **{len(cards)}**")

            # Render tarjeta actual
            render_flip_card(cards[idx]["front"], cards[idx]["back"], card_id=idx)

            # Controles paso a paso
            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
            with btn_col1:
                if st.button("⬅️ Anterior", disabled=(idx == 0), use_container_width=True):
                    st.session_state.card_index -= 1
                    st.rerun()
            with btn_col2:
                if st.button("🔄 Reiniciar Mazo", use_container_width=True):
                    del st.session_state["flashcards_data"]
                    st.session_state.card_index = 0
                    st.rerun()
            with btn_col3:
                if st.button("Siguiente ➡️", disabled=(idx == len(cards) - 1), use_container_width=True):
                    st.session_state.card_index += 1
                    st.rerun()

            st.divider()
            csv_data = "Front,Back\n" + "\n".join([f'"{c["front"]}","{c["back"]}"' for c in cards])
            st.download_button(
                label="📥 Exportar Mazo para Anki / CSV",
                data=csv_data,
                file_name="flashcards_anki.csv",
                mime="text/csv",
                use_container_width=True
            )

    # 4. CHAT CON TUTOR E INTERACCIONES RÁPIDAS
    with tab4:
        st.subheader("💬 Chat con Buddy")
        st.session_state.voice_enabled = st.checkbox("🔊 Activar lectura de voz", value=st.session_state.voice_enabled)

        # Sugerencias Rápidas de Preguntas
        st.write("💡 **Sugerencias rápidas:**")
        sug_col1, sug_col2, sug_col3 = st.columns(3)
        quick_prompt = None
        with sug_col1:
            if st.button("📌 Puntos clave", use_container_width=True):
                quick_prompt = "¿Cuáles son los 3 puntos más importantes de estos apuntes?"
        with sug_col2:
            if st.button("💡 Simplifícalo", use_container_width=True):
                quick_prompt = "Explícame el concepto más complejo de las notas como si tuviera 10 años."
        with sug_col3:
            if st.button("❓ Posibles preguntas", use_container_width=True):
                quick_prompt = "¿Qué preguntas podría hacerme un profesor en un examen sobre este texto?"

        chat_container = st.container(height=380)

        with chat_container:
            for idx, msg in enumerate(st.session_state.chat_messages):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg["role"] == "assistant" and "audio" in msg and msg["audio"]:
                        is_last = (idx == len(st.session_state.chat_messages) - 1)
                        st.audio(msg["audio"], format="audio/mp3", autoplay=is_last)

        # Entrada por teclado limpia
        user_prompt = st.chat_input("Escribe tu duda...")
        final_prompt = quick_prompt or user_prompt

        if final_prompt:
            st.session_state.chat_messages.append({"role": "user", "content": final_prompt})

            system_context = f"""
            Eres Buddy, un tutor pedagógico amable y claro.
            Responde de forma concisa y directa utilizando únicamente la información de estos apuntes:

            APUNTES:
            {st.session_state.notes_content}

            PREGUNTA DEL ESTUDIANTE:
            {final_prompt}
            """
            
            response_text = safe_gemini_call(system_context)
            
            audio_fp = None
            if st.session_state.voice_enabled and not response_text.startswith("⚠️"):
                audio_fp = text_to_speech_bytes(response_text, voice="es-ES-AlvaroNeural", rate="+40%")

            st.session_state.chat_messages.append({
                "role": "assistant", 
                "content": response_text,
                "audio": audio_fp
            })

            st.rerun()

else:
    st.info("💡 Para comenzar, sube un archivo PDF / TXT o pega tus apuntes en el panel superior.")
