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
    system = """Tu único trabajo es escribir posts para X que suenen exactamente como la persona de la transcripción.

Antes de escribir cualquier post, lee la transcripción y extrae:
- Las palabras y expresiones exactas que usa esta persona
- Su ritmo: ¿habla en frases cortas? ¿largas? ¿mezcla?
- Sus muletillas, sus pausas, cómo conecta ideas
- El tono: ¿bromea? ¿es intenso? ¿casual? ¿reflexivo?

Luego escribe como ESA persona, no como un copywriter.

REGLAS DURAS:
- Usa sus palabras, no sinónimos elegantes
- Si en la transcripción dice "un chingo de trabajo" no escribas "una gran cantidad de esfuerzo"
- Frases cortas y largas mezcladas al azar. Nunca el mismo ritmo dos veces.
- 1-2 hashtags solo si aportan, nunca decorativos
- Sin emojis al inicio. Si pones uno, al final.
- Separa cada post con: ---SEPARATOR---
- No pongas comillas ni números alrededor del post

PROHIBIDO (delatan IA al instante):
"Sin duda" / "Definitivamente" / "Es crucial" / "Es fundamental" / "Hoy en día" / "En el mundo actual" / "No olvides" / "Recuerda que" / todos los párrafos con el mismo largo / estructuras perfectas de 3 partes que se sienten como plantilla"""

    user = f"""Transcripción:
{transcription}

---

Escribe 20 posts basados en esta transcripción. La voz debe sonar exactamente como la persona que habla arriba.

BLOQUE 1 — SHORT (posts 1-6, máx 120 chars):
1. La verdad más dura del video en una línea. Sin suavizarla.
2. El número o dato más impactante. Solo eso. Sin contexto.
3. Lo que esta persona diría que nadie en su industria se atreve a decir.
4. Una sola frase que reencuadra algo que el lector da por hecho.
5. La opinión más polémica del contenido. Directa.
6. Una pregunta que queda resonando. Sin respuesta.

BLOQUE 2 — MID (posts 7-14, entre 120-220 chars):
7. La afirmación más bold + el dato que la sostiene + qué significa para quien lee.
8. Cómo era antes → qué cambió → cómo quedó. Con número real al final.
9. Un gancho fuerte + los 3 insights más valiosos en bullets cortos.
10. Pregunta con respuesta obvia. Respóndela al revés.
11. El concepto más difícil del video explicado con algo cotidiano.
12. El proceso en 3 pasos que alguien puede aplicar esta semana.
13. Lo que nadie explica sobre este tema + lo que el video revela.
14. Una situación que el lector vivió → el giro → la lección.

BLOQUE 3 — LONG (posts 15-20, entre 300-400 chars):
15. Una historia conectada al tema. Imperfecta. Real. Con tensión.
16. La postura más controversial desarrollada línea a línea.
17. Una pregunta mal respondida por la mayoría → la respuesta real → invita a continuar.
18. Caso concreto: situación → acción → resultado con número → qué aprende el lector.
19. "[N] cosas que aprendí sobre [tema]:" — las más contraintuitivas.
20. Lo que realmente piensa esta persona. Primera persona. Sin filtro. Sin hashtags.

Separa cada post con ---SEPARATOR---"""

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
