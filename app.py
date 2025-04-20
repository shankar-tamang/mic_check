from flask import Flask, render_template, request, jsonify
from vosk import Model, KaldiRecognizer
import wave
import os
import subprocess
import uuid
import json

app = Flask(__name__)

# Load Vosk model
MODEL_PATH = "vosk-model-small-en-us-0.15"
if not os.path.exists(MODEL_PATH):
    raise Exception(f"Please download and unpack model to '{MODEL_PATH}'")
model = Model(MODEL_PATH)

# Directory for saving files
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    
    # Generate unique ID for the filename
    file_id = str(uuid.uuid4())
    webm_path = os.path.join(UPLOAD_DIR, f"{file_id}.webm")
    wav_path = os.path.join(UPLOAD_DIR, f"{file_id}.wav")

    # Save original WebM file
    audio_file.save(webm_path)

    # Convert WebM to WAV using ffmpeg
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', webm_path, wav_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"FFmpeg conversion failed: {e.stderr.decode()}"}), 500

    # Transcribe using Vosk
    try:
        wf = wave.open(wav_path, "rb")
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)

        results = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                results.append(rec.Result())
        results.append(rec.FinalResult())
        wf.close()
    except Exception as e:
        return jsonify({"error": f"Vosk processing failed: {str(e)}"}), 500

    # Parse text
    texts = []
    for r in results:
        j = json.loads(r)
        if 'text' in j:
            texts.append(j['text'])
    full_text = " ".join(texts).strip()

    return jsonify({
        "text": full_text,
        "original_file": webm_path,
        "converted_file": wav_path
    })

if __name__ == "__main__":
    app.run(debug=True)
