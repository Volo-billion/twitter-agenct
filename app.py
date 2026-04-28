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
    prompt = f"""Eres un experto en redes sociales y copywriting para X (Twitter).

Basándote en la siguiente transcripción de un video, genera exactamente 5 posts para X/Twitter.

REGLAS ESTRICTAS:
- Máximo 280 caracteres por post (incluyendo hashtags)
- Cada post debe tener un ángulo DIFERENTE del contenido
- Incluye 2-3 hashtags relevantes en cada post
- Tono profesional pero cercano y humano
- Escritos en español
- Que generen engagement (preguntas, datos curiosos, reflexiones, etc.)
- NO uses comillas alrededor del post
- NO numeres los posts
- Separa cada post con la línea exacta: ---SEPARATOR---

ÁNGULOS SUGERIDOS (usa uno distinto por post):
1. El insight o aprendizaje principal
2. Una estadística o dato impactante del contenido
3. Una pregunta reflexiva para el público
4. Un consejo accionable
5. Una frase motivacional o de impacto relacionada al tema

TRANSCRIPCIÓN:
{transcription}

Genera los 5 posts ahora:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
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
