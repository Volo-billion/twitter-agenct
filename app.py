import os
import uuid
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_audio(video_path: Path, audio_path: Path) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "libmp3lame",
        "-ar", "16000",
        "-ac", "1",
        "-b:a", "64k",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr}")


def transcribe_audio(audio_path: Path) -> str:
    with open(audio_path, "rb") as audio_file:
        transcription = groq_client.audio.transcriptions.create(
            file=(audio_path.name, audio_file.read()),
            model="whisper-large-v3",
            response_format="text",
            language="es",
        )
    return transcription


def generate_posts(transcription: str) -> list[str]:
    system = """Eres el mejor ghostwriter de X (Twitter) en español de habla latina. Escribes posts que paran el scroll, generan miles de likes y convierten lectores en seguidores.

LEYES DE LA VIRALIDAD QUE APLICAS:
- Los primeros 8 palabras son todo. Si no enganchan, el post muere.
- Tensión + especificidad + formato correcto = viral
- Números > adjetivos siempre ("47%" > "casi la mitad", "3 años" > "mucho tiempo")
- La contradicción y lo inesperado detienen el dedo
- Las preguntas que incomodan generan replies — el algoritmo las recompensa
- Voz activa siempre. Nunca pasiva.

REGLAS ABSOLUTAS:
- Short Copy: máximo 120 caracteres
- Mid Copy: entre 120 y 220 caracteres
- Long Copy: entre 300 y 400 caracteres (usa cada palabra, cuenta una historia completa)
- Español latino directo. Sin relleno.
- 1-2 hashtags ultra-específicos del tema. NUNCA: #Éxito #Motivación #Emprendimiento #Liderazgo
- PROHIBIDO empezar con emoji — si usas uno, va al final del post
- PROHIBIDO: "Es importante", "No olvides", "Recuerda que", "En el mundo actual", "Sin duda", "Definitivamente"
- NO pongas comillas alrededor del post
- NO numeres los posts
- Separa cada post con exactamente esta línea: ---SEPARATOR---

EJEMPLOS DE CALIDAD (aprende el nivel):

[SHORT - One-Liner Contrarian]
Trabajar más horas no es productividad. Es ansiedad con sueldo. #DeepWork

[SHORT - Dato Desnudo]
El 92% de los proyectos "urgentes" del lunes son irrelevantes el viernes.

[MID - Hook + Prueba]
Dejé de revisar email antes de las 10am durante 90 días.
Resultado: terminé un 40% más trabajo importante por semana.
El email es la bandeja de entrada de las prioridades de otros. #GTD

[MID - Historia Micro]
Renuncié a mi trabajo con $800 en el banco.
Todos me dijeron que estaba loco.
Dos años después facturé más que mi ex-jefe. No era locura. Era claridad.

[LONG - Argumento Construido]
La mayoría fracasa en sus metas no por falta de disciplina. Fracasa por metas mal diseñadas.
Una meta vaga ("quiero ser fit") no activa el cerebro. Una meta específica sí ("bajar 8kg antes del 15 de marzo").
El problema no eres tú. Es el sistema que usas. Cambia el sistema. #Metas #Productividad"""

    user = f"""Genera exactamente 20 posts para X basándote en la transcripción. Usa CADA fórmula UNA sola vez, en este orden exacto:

═══ BLOQUE 1: SHORT COPY (posts 1-6) — máx 120 chars cada uno ═══

POST 1 — ONE-LINER CONTRARIAN
Una sola frase que contradiga lo que todos creen. Golpea y termina.

POST 2 — DATO DESNUDO
Solo el número o hecho más impactante del contenido. Sin explicación. La ambigüedad genera replies.

POST 3 — VERDAD INCÓMODA
Lo que todos piensan pero nadie se atreve a decir en voz alta sobre este tema.

POST 4 — EL REFRAME
Toma algo conocido del video y cámbialo de perspectiva en una línea. "X no es Y. Es Z."

POST 5 — DECLARACIÓN BOLD
Afirmación fuerte y polémica sin justificación. Corta. Genera debate.

POST 6 — EL CLIFFHANGER
Frase que abre un loop mental sin cerrarlo. Deja al lector queriendo más.

═══ BLOQUE 2: MID COPY (posts 7-14) — 120-220 chars cada uno ═══

POST 7 — HOOK + PRUEBA
Claim bold en línea 1. Dato concreto que lo respalda en línea 2. Implicación en línea 3.

POST 8 — ANTES / DESPUÉS
Estado A (1 línea) → qué cambió (1 línea) → Estado B con resultado medible (1 línea).

POST 9 — LOS 3 BULLETS
Intro de 1 línea + exactamente 3 insights del video en formato bullet "•"

POST 10 — PREGUNTA + RESPUESTA INESPERADA
Pregunta que parece tener respuesta obvia. Respuesta sorprendente o contraintuitiva.

POST 11 — LA ANALOGÍA
Explica el concepto más complejo del video usando una metáfora de la vida cotidiana.

POST 12 — FRAMEWORK SIMPLE
"Para [resultado del video]: • Paso 1 • Paso 2 • Paso 3" — accionable y concreto.

POST 13 — EL PROCESO REVELADO
"Cómo funciona [tema] en realidad:" + 2-3 líneas con lo que nadie explica así.

POST 14 — HISTORIA MICRO
Situación reconocible (1 línea) → Giro inesperado (1 línea) → Lección aplicable (1 línea).

═══ BLOQUE 3: LONG COPY (posts 15-20) — entre 300 y 400 chars cada uno ═══

POST 15 — HISTORIA COMPLETA
Setup (contexto, 1 línea) → Conflicto (el problema, 1 línea) → Resolución (qué pasó, 1 línea) → Lección (el takeaway, 1 línea).

POST 16 — ARGUMENTO CONSTRUIDO
Tesis polémica (1 línea) → Evidencia 1 del video → Evidencia 2 → Conclusión que cambia perspectiva.

POST 17 — THREAD STARTER
Post que funciona solo pero deja una pregunta abierta al final que invita a seguir. Termina con "Hilo →" o "Te explico:"

POST 18 — MINI CASE STUDY
Situación real del contenido → acción específica tomada → resultado con número exacto → qué aprender de eso.

POST 19 — LISTA DE VALOR
"[Número] cosas que [tema del video] enseña sobre [tema mayor de vida/negocios]:" + los items más poderosos.

POST 20 — MANIFIESTO PERSONAL
Postura personal, clara y sin miedo sobre el tema. Primera persona. Vulnerable y directo. Sin hashtags.

═══ TRANSCRIPCIÓN ═══
{transcription}

IMPORTANTE: Genera los 20 posts en orden. Separa cada uno con ---SEPARATOR--- Respeta los límites de caracteres por bloque."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=4096,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    raw = response.choices[0].message.content
    posts = [p.strip() for p in raw.split("---SEPARATOR---") if p.strip()]
    return posts[:20]


def cleanup(*paths: Path) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No se recibió ningún archivo."}), 400

    file = request.files["video"]

    if file.filename == "":
        return jsonify({"error": "Nombre de archivo vacío."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Formato no soportado. Usa mp4, mov, avi, mkv o webm."}), 400

    uid = uuid.uuid4().hex
    ext = file.filename.rsplit(".", 1)[1].lower()
    video_path = UPLOAD_FOLDER / f"{uid}.{ext}"
    audio_path = UPLOAD_FOLDER / f"{uid}.mp3"

    try:
        file.save(video_path)

        # Step 1 — extract audio
        try:
            extract_audio(video_path, audio_path)
        except FileNotFoundError:
            return jsonify({"error": "ffmpeg no está instalado en el servidor."}), 500
        except RuntimeError as e:
            return jsonify({"error": f"Error extrayendo audio: {str(e)}"}), 500

        # Step 2 — transcribe
        try:
            transcription = transcribe_audio(audio_path)
        except Exception as e:
            return jsonify({"error": f"Error transcribiendo audio: {str(e)}"}), 500

        if not transcription or not transcription.strip():
            return jsonify({"error": "No se detectó habla en el video."}), 422

        # Step 3 — generate posts
        try:
            posts = generate_posts(transcription)
        except Exception as e:
            return jsonify({"error": f"Error generando posts: {str(e)}"}), 500

        return jsonify({"transcription": transcription.strip(), "posts": posts})

    finally:
        cleanup(video_path, audio_path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
