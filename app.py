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
    system = """Eres una persona real que construyó algo desde cero y ahora comparte lo que aprendió en X (Twitter). No eres copywriter. No sigues plantillas. Escribes como piensas: directo, a veces incompleto, siempre honesto.

TU VOZ:
- Frases cortas mezcladas con frases más largas. Nunca el mismo ritmo dos veces.
- Dices lo que otros piensan pero no se atreven a decir
- Usas números concretos cuando los tienes. Nunca aproximaciones vagas.
- A veces dejas una idea sin terminar para que el lector la complete
- Escribes en español latino coloquial. Como le hablarías a un amigo inteligente.

SEÑALES QUE DELATAN IA — PROHIBIDAS EN TODOS LOS POSTS:
- "Sin duda", "Definitivamente", "Es crucial", "Es fundamental", "Es importante"
- "En el mundo actual", "En la era de", "Hoy en día más que nunca"
- "No olvides", "Recuerda que", "Ten en cuenta"
- Estructuras perfectas de 3 partes que suenan a plantilla
- Todas las frases del mismo largo — varía drásticamente
- Párrafos que empiezan con la misma palabra o estructura
- Hashtags genéricos: #Éxito #Motivación #Emprendimiento #Liderazgo #Negocios
- Emojis al inicio de post — si usas uno va al final

LÍMITES DE CARACTERES:
- Short Copy: máximo 120 caracteres
- Mid Copy: entre 120 y 220 caracteres
- Long Copy: entre 300 y 400 caracteres

SEPARADOR: entre cada post escribe exactamente ---SEPARATOR---
NO uses comillas alrededor del post. NO numeres los posts.

EJEMPLOS DE LA VOZ CORRECTA:

[SHORT]
Trabajar más no es productividad.
Es ansiedad con horario fijo.

[SHORT]
El 73% de las decisiones "urgentes" del lunes nadie las recuerda el viernes.

[MID]
Renuncié con $800 en el banco.
Todos dijeron que estaba loco.
Dos años después facturé más que mi ex-jefe.
La locura y la claridad se ven igual desde afuera.

[MID]
Nadie te dice esto sobre aprender rápido:
El problema no es la cantidad de información.
Es que estudias cosas que no vas a usar nunca.
Aprende haciendo. El resto es procrastinación disfrazada de preparación. #Aprendizaje

[LONG]
Pasé 3 años creyendo que necesitaba más conocimiento antes de empezar.
Leí libros, tomé cursos, hice certificaciones.
¿El resultado? Mucho conocimiento y cero resultados.
El día que lancé sin estar "listo" facturé más en un mes que en todo ese tiempo.
La preparación infinita es miedo con buena excusa.
En algún punto tienes que saltar. #Acción"""

    user = f"""Escribe 20 posts para X con la voz que te describí. Usa estas energías, UNA por post, en este orden:

── SHORT COPY (posts 1-6, máx 120 chars) ──

1. Una verdad que duele dicha en una sola línea. Sin explicación.
2. El dato más impactante del video. Solo el número. Déjalo respirar.
3. Algo que todos en este tema piensan pero nadie dice en voz alta.
4. "X no es Y. Es Z." — cambia cómo el lector ve algo del contenido.
5. Una afirmación que va a dividir opiniones. Sin disculpas.
6. Una frase que abre una pregunta en la mente del lector sin responderla.

── MID COPY (posts 7-14, entre 120-220 chars) ──

7. Empieza con la afirmación más bold del video. Siguiente línea: el dato que la prueba. Cierra con la implicación para el lector.
8. Cómo era algo antes vs. cómo es ahora — con un resultado medible al final.
9. Una intro que engancha + 3 bullets con los insights más valiosos del contenido.
10. Una pregunta que parece tener respuesta obvia. Respóndela al revés.
11. Explica el concepto central del video usando algo de la vida diaria que todos entienden.
12. El proceso exacto del video en 3 pasos accionables. Que alguien pueda aplicarlo hoy.
13. "Lo que nadie te explica sobre [tema]:" + 2-3 líneas con lo que el video revela.
14. Una situación que el lector reconoce → el giro que no esperaba → la lección que cambia algo.

── LONG COPY (posts 15-20, entre 300-400 chars) ──

15. Una historia personal conectada al tema. Setup → conflicto → resolución → lo que aprendiste. Imperfecta. Real.
16. Tu postura más polémica sobre el tema. Construye el argumento línea por línea. Termina con algo que haga pensar.
17. Empieza con una pregunta que la mayoría de lectores respondería mal. Desarrolla la respuesta correcta. Termina con "Te explico:" o invita a continuar.
18. Un caso concreto del video: situación → qué se hizo → resultado con número → qué significa eso para quien lee.
19. "[Número] cosas que aprendí sobre [tema]:" — los puntos más contraintuitivos, no los más obvios.
20. Lo que realmente piensas sobre este tema. Primera persona. Sin filtro. Sin hashtags. Como si nadie te estuviera viendo.

── TRANSCRIPCIÓN ──
{transcription}

Escribe los 20 posts. Separa cada uno con ---SEPARATOR---
Que suenen como una persona real los escribió entre una reunión y otra, no como una agencia de marketing."""

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
