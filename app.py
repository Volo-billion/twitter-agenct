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
    system = """Eres el mejor ghostwriter de X (Twitter) en español. Tus posts generan miles de likes, retweets y respuestas porque dominas la psicología de la viralidad.

PSICOLOGÍA QUE APLICAS:
- Los primeros 8 palabras deciden si alguien sigue leyendo o hace scroll
- La tensión, la contradicción y lo inesperado detienen el dedo
- La especificidad genera credibilidad ("3 años" > "mucho tiempo", "47%" > "casi la mitad")
- Las preguntas que incomodan generan respuestas — el algoritmo las ama
- La vulnerabilidad y la historia personal conectan más que el consejo genérico

REGLAS ABSOLUTAS:
- Máximo 280 caracteres por post (cuenta cada carácter, incluyendo espacios y hashtags)
- Escritos en español latino, tono directo y humano
- 1-2 hashtags específicos del tema, nunca genéricos (#Éxito #Motivación #Emprendimiento están prohibidos)
- PROHIBIDO empezar con emoji — si usas uno, va al final
- PROHIBIDO usar frases vacías: "Es importante", "No olvides", "Recuerda que", "En el mundo actual"
- NO uses comillas alrededor del post
- NO numeres los posts
- Separa cada post con la línea exacta: ---SEPARATOR---

EJEMPLOS DE POSTS VIRALES (referencia de calidad):

Ejemplo A (Hook Contrarian):
Trabajar más horas no te hace más productivo. Te hace más ocupado. Hay una diferencia enorme entre las dos cosas y la mayoría nunca la aprende. #Productividad #DeepWork

Ejemplo B (Promesa Específica):
Dejé de revisar el correo antes de las 10am durante 90 días. Resultado: terminé un 40% más de trabajo importante cada semana. El email es la bandeja de entrada de las prioridades de otros. #GTD

Ejemplo C (Pregunta que Incomoda):
¿Cuántas veces has dicho "voy a empezar el lunes"? El lunes es el día más popular para empezar cosas que nunca se terminan. Empieza hoy, aunque sea mal. #Acción"""

    user = f"""Basándote en esta transcripción, genera exactamente 5 posts usando UNA fórmula distinta por post:

POST 1 — HOOK CONTRARIAN
Empieza con algo que contradiga la creencia popular sobre el tema. Formato: "[Creencia común] es mentira/está mal/es un mito." o "La mayoría [hace X]. Error."

POST 2 — PROMESA ESPECÍFICA
Empieza en primera persona con un resultado concreto y medible. Formato: "Hice [acción específica] durante [tiempo exacto]:" o "[Número] semanas aplicando esto:"

POST 3 — PREGUNTA QUE INCOMODA
Una pregunta retórica que haga pensar al lector sobre su propia vida. Cierra con tu postura en 1 línea contundente.

POST 4 — DATO INESPERADO
Abre con una cifra o hecho sorprendente extraído del contenido. Cierra con la implicación práctica para el lector.

POST 5 — HISTORIA 3 ACTOS
Situación inicial (1 línea) → Problema o giro (1 línea) → Resolución o aprendizaje (1 línea). Todo en 280 chars.

TRANSCRIPCIÓN:
{transcription}

Genera los 5 posts ahora. Recuerda: separa cada uno con ---SEPARATOR--- y nunca superes 280 caracteres."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    raw = response.choices[0].message.content
    posts = [p.strip() for p in raw.split("---SEPARATOR---") if p.strip()]
    return posts[:5]


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
