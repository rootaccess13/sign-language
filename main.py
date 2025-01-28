#FSL Translation
from flask import Flask, request, jsonify
import whisper
import io
import numpy as np

app = Flask(__name__)

model = whisper.load_model("turbo")  # You can use other models: tiny, small, medium, large

@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        audio_file = request.files["audio"]

        audio_data = io.BytesIO(audio_file.read())

        audio = whisper.load_audio(audio_data)  # This converts it to np.ndarray

        audio = whisper.pad_or_trim(audio)  # Whisper models usually require fixed-length input

        result = model.transcribe(audio)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)  # Listen on all network interfaces (0.0.0.0)
