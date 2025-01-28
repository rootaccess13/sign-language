from flask import Flask, request, jsonify
import whisper
import io
import numpy as np

app = Flask(__name__)

# Load Whisper model
model = whisper.load_model("turbo")  # You can use other models: tiny, small, medium, large

@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        # Get the uploaded audio file from the request
        audio_file = request.files["audio"]

        # Convert audio file to byte stream
        audio_data = io.BytesIO(audio_file.read())

        # Load audio data using Whisper's load_audio method
        audio = whisper.load_audio(audio_data)  # This converts it to np.ndarray

        # Preprocess the audio if necessary (e.g., resampling)
        audio = whisper.pad_or_trim(audio)  # Whisper models usually require fixed-length input

        # Use Whisper to transcribe the audio
        result = model.transcribe(audio)

        # Return the transcription in JSON format
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)  # Listen on all network interfaces (0.0.0.0)
