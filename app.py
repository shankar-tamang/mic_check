from flask import Flask, request, jsonify, render_template
import uuid
import os
import subprocess
import speech_recognition as sr
import whisper
from google.cloud import speech

app = Flask(__name__)
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load Whisper model once (to avoid delay)
whisper_model = whisper.load_model("base")  # Change model size as needed

# Initialize Google Cloud Speech client once
gcs_client = speech.SpeechClient.from_service_account_file("metal-seeker-416407-35f733b9945b.json")

@app.route("/")
def index():
    return render_template("index.html")


def convert_webm_to_wav(webm_path, wav_path):
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", webm_path, "-ar", "16000", "-ac", "1", wav_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg conversion failed: {e.stderr.decode()}")


def transcribe_google_cloud(wav_path):
    with open(wav_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True,
    )

    response = gcs_client.recognize(config=config, audio=audio)
    transcripts = [result.alternatives[0].transcript for result in response.results]
    return " ".join(transcripts).strip()


@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    method = request.args.get('method', 'speech_recognition').lower()

    audio_file = request.files['audio']
    file_id = str(uuid.uuid4())
    webm_path = os.path.join(UPLOAD_DIR, f"{file_id}.webm")
    wav_path = os.path.join(UPLOAD_DIR, f"{file_id}.wav")
    audio_file.save(webm_path)

    try:
        convert_webm_to_wav(webm_path, wav_path)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    try:
        if method == "whisper":
            result = whisper_model.transcribe(wav_path)
            text = result.get("text", "").strip()

        elif method == "speech_recognition":
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)

        elif method == "google_cloud_speech":
            text = transcribe_google_cloud(wav_path)

        else:
            return jsonify({"error": "Invalid transcription method selected"}), 400

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
