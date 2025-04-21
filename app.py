from flask import Flask, request, jsonify, render_template
import uuid, os, subprocess
import speech_recognition as sr
import whisper

app = Flask(__name__)
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load Whisper model once (to avoid delay)
whisper_model = whisper.load_model("base")  # You can change to tiny/small/medium/large


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    method = request.args.get('method', 'speech_recognition')  # default: speech_recognition

    # Save and convert audio
    audio_file = request.files['audio']
    file_id = str(uuid.uuid4())
    webm_path = os.path.join(UPLOAD_DIR, f"{file_id}.webm")
    wav_path = os.path.join(UPLOAD_DIR, f"{file_id}.wav")
    audio_file.save(webm_path)

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", webm_path, "-ar", "16000", "-ac", "1", wav_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"FFmpeg conversion failed: {e.stderr.decode()}"}), 500

    try:
        if method == "whisper":
            result = whisper_model.transcribe(wav_path)
            text = result.get("text", "").strip()

        elif method == "speech_recognition":
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)

        else:
            return jsonify({"error": "Invalid method selected"}), 400

        return jsonify({
            "text": text,
            "method": method,
            "original_file": webm_path,
            "converted_file": wav_path
        })

    except Exception as e:
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
