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
from st_mic_recorder import mic_recorder

# Cargar variables de entorno
load_dotenv()

# Configuración de API Key
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="AI Study Buddy",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- INYECCIÓN DE CSS (RESPONSIVO + MICRÓFONO INTEGRADO ESTILO GEMINI) ---
st.markdown("""
    <style>
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
            padding-top: 1rem !important;
        }
        
        h1 { font-size: 1.8rem !important; }
        h2 { font-size: 1.4rem !important; }
        
        .stButton button {
            width: 100% !important;
            min-height: 48px !important;
            font-size: 16px !important;
        }
        
        .stTabs [data-baseweb="tab-list"] { gap: 2px !important; }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 10px !important;
            font-size: 13px !important;
        }
    }
    
    div[data-testid="stForm"] {
        padding: 12px !important;
    }

    /* Ajuste para que el contenedor del chat soporte elementos absolutos dentro de su área */
    div[data-testid="stChatInput"] {
        position: relative !important;
    }

 /* Posicionar el micrófono a la izquierda del botón enter sin taparlo */
    iframe[title="audio_recorder_streamlit.audio_recorder"] {
        position: absolute !important;
        bottom: 25px !important;
        right: 55px !important; 
        z-index: 999999 !important;
        width: 32px !important;
        height: 32px !important;
        border: none !important;
        background: transparent !important;
    }

  

    /* Margen a la derecha del texto para que no tape los botones */
    div[data-testid="stChatInput"] textarea {
        padding-right: 90px !important;
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

# --- FUNCIONES AUXILIARES ---

def extract_text_from_pdf(pdf_file):
    try:
        reader = pypdf.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    except pypdf.errors.PdfReadError:
        st.error("⚠️ El archivo PDF está cifrado o requiere contraseña.")
        return ""
    except Exception as e:
        st.error(f"⚠️ Error al leer el archivo PDF: {e}")
        return ""

def safe_gemini_call(prompt):
    """Maneja reintentos y respaldos automáticos si Gemini está sobrecargado (503)."""
    models_to_try = ['gemini-3.6-flash', 'gemini-2.5-flash', 'gemini-1.5-flash']
    
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
                    
    return "⚠️ **El servicio está experimentando alta demanda.** Por favor, intenta de nuevo en unos momentos."

def text_to_speech_bytes(text, voice="es-ES-AlvaroNeural", rate="+50%"):
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
    except Exception as e:
        return None

def transcribe_audio_bytes(audio_bytes):
    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=[
                "Transcribe de manera exacta el siguiente audio en español. Devuelve ÚNICAMENTE el texto transcrito sin explicaciones ni comillas.",
                {"mime_type": "audio/wav", "data": audio_bytes}
            ]
        )
        return response.text.strip()
    except Exception as e:
        return None

def generate_summary(notes_text):
    prompt = f"Eres un profesor experto. Genera un resumen conciso, claro y bien estructurado de estos apuntes utilizando títulos y viñetas:\n\n{notes_text}"
    return safe_gemini_call(prompt)

def generate_interactive_quiz(notes_text, num_questions=5):
    prompt = f"""
    Eres un profesor experto. Basándote en el siguiente texto, genera exactamente {num_questions} preguntas de opción múltiple.
    DEBES responder ÚNICAMENTE en formato JSON válido, sin bloques de formato ni texto adicional.

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
    prompt = f"""
    Eres un diseñador de mapas conceptuales e infografías educativas.
    Analiza el texto facilitado y genera un código Graphviz en formato DOT válido.

    Instrucciones:
    - Usa 'digraph G {{ ... }}'.
    - Estilo de nodos: `node [shape=box, style="filled,rounded", color="#2b2d42", fontcolor=white, fontname="Helvetica"];`
    - Diseña relaciones conceptuales claras.
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
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
    .flip-card {{
      background-color: transparent;
      width: 100%;
      height: 220px;
      perspective: 1000px;
      margin-bottom: 15px;
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
      padding: 15px;
      box-sizing: border-box;
    }}
    .flip-card-front {{
      background-color: #2b2d42;
      color: #edf2f4;
      border: 2px solid #8d99ae;
    }}
    .flip-card-back {{
      background-color: #0077b6;
      color: #ffffff;
      transform: rotateY(180deg);
    }}
    .hint {{
      font-size: 11px;
      opacity: 0.8;
      margin-top: 10px;
    }}
    </style>
    </head>
    <body>
    <div class="flip-card" id="card-{card_id}" onclick="this.classList.toggle('flipped')">
      <div class="flip-card-inner">
        <div class="flip-card-front">
          <strong style="font-size: 15px;">{front_text}</strong>
          <span class="hint">👆 Toca para voltear</span>
        </div>
        <div class="flip-card-back">
          <p style="font-size: 14px; margin: 0;">{back_text}</p>
          <span class="hint">🔄 Toca para volver</span>
        </div>
      </div>
    </div>
    </body>
    </html>
    """
    components.html(card_html, height=240)

def render_certificate(student_name, score, total):
    cert_html = f"""
    <div style="border: 6px double #2b2d42; padding: 15px; text-align: center; background-color: #f8f9fa; border-radius: 12px; margin-top: 15px;">
        <h2 style="color: #0077b6; font-family: Georgia, serif; font-size: 1.3rem; margin-bottom: 5px;">📜 CERTIFICADO DE EXCELENCIA</h2>
        <p style="font-size: 13px; color: #555;">Otorgado a:</p>
        <h3 style="color: #2b2d42; text-transform: uppercase; font-size: 1.2rem; margin: 10px 0; border-bottom: 2px solid #0077b6; display: inline-block; padding-bottom: 3px;">{student_name}</h3>
        <p style="font-size: 13px; color: #555;">Por completar la evaluación en <strong>AI Study Buddy</strong>.</p>
        <div style="font-size: 16px; font-weight: bold; color: #28a745; margin: 10px 0;">
            Calificación: {score} / {total} ({(score/total)*100:.1f}%)
        </div>
        <p style="font-size: 10px; color: #888; margin-top: 15px;">Emitido por Buddy IA</p>
    </div>
    """
    st.markdown(cert_html, unsafe_allow_html=True)

# --- INTERFAZ PRINCIPAL ---

st.title("🤖 AI Study Buddy")
st.caption("Aprende a tu ritmo")

if not api_key:
    st.error("⚠️ No se encontró la GEMINI_API_KEY. Configúrala en Secrets o en tu archivo .env.")
    st.stop()

# --- CARGA DE TEXTO ---
st.subheader("📄 Carga tus apuntes")
uploaded_file = st.file_uploader("Sube tus apuntes (PDF o TXT)", type=["pdf", "txt"])

with st.form("notes_form", clear_on_submit=False):
    text_input = st.text_area("O pega directamente tus notas aquí:", height=120)
    submit_text = st.form_submit_button("Procesar Texto ↵")

if submit_text or uploaded_file is not None:
    previous_content = st.session_state.notes_content
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            st.session_state.notes_content = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "text/plain":
            st.session_state.notes_content = uploaded_file.read().decode("utf-8")
    elif text_input.strip():
        st.session_state.notes_content = text_input.strip()

    if previous_content != st.session_state.notes_content:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "¡Hola! 👋 Soy **Buddy**, tu tutor personal. Ya he leído tus apuntes. ¿Qué duda quieres resolver?"}
        ]
        st.session_state.summary_text = ""

# --- PESTAÑAS RESPONSIVAS ---
if st.session_state.notes_content:
    st.success("✅ Apuntes cargados.")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📌 Resumen", 
        "❓ Quiz", 
        "🎴 Flashcards", 
        "🗺️ Haz un Mapa Conceptual",
        "💬 Chat con Buddy"
    ])

    # 1. RESUMEN
    with tab1:
        st.subheader("Resumen de apuntes")
        if st.button("Generar Resumen 📝", type="primary", use_container_width=True):
            with st.spinner("Procesando..."):
                st.session_state.summary_text = generate_summary(st.session_state.notes_content)

        if st.session_state.summary_text:
            st.markdown(st.session_state.summary_text)
            st.download_button(
                label="📥 Descargar Resumen (.md)",
                data=st.session_state.summary_text,
                file_name="Resumen_Apuntes_Buddy.md",
                mime="text/markdown",
                use_container_width=True
            )

    # 2. QUIZ INTERACTIVO
    with tab2:
        st.subheader("Cuestionario Interactivo")
        num_q = st.slider("Número de preguntas:", min_value=1, max_value=10, value=5)
        seconds_per_q = st.number_input("Segundos por pregunta:", min_value=10, max_value=120, value=30)

        if st.button("🚀 Comenzar Quiz", type="primary", use_container_width=True):
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
                st.write(f"⏱️ **Tiempo restante:** `{remaining} s`")
                st.progress(percent_left)

                if remaining == 0:
                    st.warning("⏰ ¡Tiempo agotado!")
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

                submit_quiz = st.form_submit_button("Enviar Respuestas 📝", use_container_width=True)
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
                        st.success(f"**P{q_idx + 1}: CORRECTA ✅**")
                    else:
                        st.error(f"**P{q_idx + 1}: INCORRECTA ❌**")
                        st.write(f"👉 Tu respuesta: `{user_ans}`")
                        st.write(f"✅ Correcta: `{correct_ans}`")

                    st.info(f"💡 **Explicación:** {q['explanation']}")
                    st.write("---")

                score_percentage = (correct_count / total_q) * 100
                st.metric(label="Puntuación", value=f"{correct_count} / {total_q}", delta=f"{score_percentage:.1f}%")

                if score_percentage >= 70:
                    st.balloons()
                    student_name = st.text_input("Nombre para certificado:", value="Estudiante")
                    if student_name:
                        render_certificate(student_name, correct_count, total_q)

    # 3. FLASHCARDS
    with tab3:
        st.subheader("🎴 Fichas de Estudio")
        num_f = st.slider("Número de tarjetas:", min_value=1, max_value=10, value=5)

        if st.button("Generar Flashcards 🎴", type="primary", use_container_width=True):
            with st.spinner("Generando..."):
                cards_data = generate_flashcards_json(st.session_state.notes_content, num_cards=num_f)
                if cards_data:
                    st.session_state["flashcards_data"] = cards_data

        if "flashcards_data" in st.session_state:
            st.info("💡 Toca la tarjeta para ver el reverso.")
            for idx, card in enumerate(st.session_state["flashcards_data"]):
                render_flip_card(card["front"], card["back"], card_id=idx)

            csv_data = "Front,Back\n" + "\n".join([f'"{c["front"]}","{c["back"]}"' for c in st.session_state["flashcards_data"]])
            st.download_button(
                label="📥 Exportar Flashcards (.csv)",
                data=csv_data,
                file_name="flashcards_anki.csv",
                mime="text/csv",
                use_container_width=True
            )

    # 4. MAPA CONCEPTUAL
    with tab4:
        st.subheader("🗺️ Esquema Conceptual")

        if st.button("Generar Mapa 🗺️", type="primary", use_container_width=True):
            with st.spinner("Diseñando..."):
                dot_code = generate_concept_map_dot(st.session_state.notes_content)
                if dot_code:
                    st.session_state["concept_map_dot"] = dot_code

        if "concept_map_dot" in st.session_state:
            try:
                st.graphviz_chart(st.session_state["concept_map_dot"], use_container_width=True)
            except Exception:
                st.error("⚠️ No se pudo generar el esquema.")
                del st.session_state["concept_map_dot"]

   # 5. CHAT CON BUDDY
    with tab5:
        st.subheader("💬 Consulta a tu Tutor Buddy")
        st.session_state.voice_enabled = st.checkbox("🔊 Activar respuesta por voz (Masculina 1.5x)", value=st.session_state.voice_enabled)

        if not st.session_state.chat_messages:
            st.session_state.chat_messages = [
                {"role": "assistant", "content": "¡Hola! 👋 Soy **Buddy**. Pregúntame lo que quieras."}
            ]

        chat_container = st.container(height=380)

        with chat_container:
            for idx, msg in enumerate(st.session_state.chat_messages):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg["role"] == "assistant" and "audio" in msg and msg["audio"]:
                        is_last = (idx == len(st.session_state.chat_messages) - 1)
                        st.audio(msg["audio"], format="audio/mp3", autoplay=is_last)

        # Disposición con columnas
        col_input, col_mic = st.columns([0.85, 0.15])

        with col_mic:
            # Grabador de audio sin fondos ni recuadros oscuros
            audio = mic_recorder(
                start_prompt="🎙️",
                stop_prompt="⏹️",
                key='recorder',
                just_once=True,
                use_container_width=True
            )

        with col_input:
            user_prompt = st.chat_input("Escribe tu pregunta...")

        voice_prompt = None
        if audio and "bytes" in audio and audio["bytes"]:
            with st.spinner("Transcribiendo voz..."):
                voice_prompt = transcribe_audio_bytes(audio["bytes"])

        final_prompt = voice_prompt or user_prompt

        if final_prompt:
            st.session_state.chat_messages.append({"role": "user", "content": final_prompt})

            system_context = f"""
            Eres Buddy, un tutor de estudio amigable.
            Responde de forma clara y directa basándote EXCLUSIVAMENTE en el contenido de los apuntes.

            APUNTES:
            {st.session_state.notes_content}

            PREGUNTA:
            {final_prompt}
            """
            
            response_text = safe_gemini_call(system_context)
            
            audio_fp = None
            if st.session_state.voice_enabled and not response_text.startswith("⚠️"):
                audio_fp = text_to_speech_bytes(response_text, voice="es-ES-AlvaroNeural", rate="+50%")

            st.session_state.chat_messages.append({
                "role": "assistant", 
                "content": response_text,
                "audio": audio_fp
            })

            st.rerun()

else:
    st.info("💡 Sube un archivo o escribe tus notas arriba para empezar.")
