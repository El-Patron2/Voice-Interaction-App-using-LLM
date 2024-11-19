import os
import base64
from datetime import datetime
import time
from functools import wraps
import vertexai
from google.api_core import exceptions as google_exceptions
from vertexai.generative_models import GenerativeModel, Part, SafetySetting, GenerationConfig
from flask import Flask, request, render_template, jsonify, send_file
from pathlib import Path

app = Flask(__name__)

UPLOAD_FOLDER = Path(app.root_path) / 'output'
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Exponential backoff and retry configuration
MAX_RETRIES = 3
BASE_DELAY = 1  # seconds

def exponential_backoff(attempt):
    return BASE_DELAY * (2 ** attempt)

def retry_with_backoff(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except (google_exceptions.ResourceExhausted, google_exceptions.ServiceUnavailable) as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(exponential_backoff(attempt))
        return None
    return wrapper

@app.route('/')
def index():
    return render_template('index.html')

@app.errorhandler(404)
def page_not_found(e):
    return jsonify(error="404 Not Found: The requested URL was not found on the server."), 404

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify(error="500 Internal Server Error: An unexpected error occurred."), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audioFile' not in request.files:
        return jsonify(error="No file uploaded"), 400
    
    audio_file = request.files['audioFile']
    if audio_file.filename == '':
        return jsonify(error="No file selected"), 400

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f"analysis_{timestamp}"
    file_extension = os.path.splitext(audio_file.filename)[1].lower()
    temp_path = os.path.join(app.root_path, f'temp_{timestamp}{file_extension}')

    try:
        audio_file.save(temp_path)
        mime_type = "audio/mpeg" if file_extension == '.mp3' else "audio/wav"
        
        transcription = generate_transcription(temp_path, mime_type)
        final_result = generate_analysis(transcription)
        
        output_path = UPLOAD_FOLDER / f"{base_filename}.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_result)
        
        return jsonify({
            'result': final_result,
            'downloadPath': f'/download/{base_filename}.txt'
        })
    
    except google_exceptions.ResourceExhausted:
        return jsonify(error="API quota exceeded. Please try again later."), 429
    except Exception as e:
        return jsonify(error=f"An unexpected error occurred: {str(e)}"), 500
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/download/<filename>')
def download_file(filename):
    try:
        return send_file(UPLOAD_FOLDER / filename, as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify(error="File not found or unable to download."), 404

@retry_with_backoff
def generate_transcription(audio_path, mime_type):
    vertexai.init(project="voiceinteractionapp-436602", location="us-central1")
    model = GenerativeModel("gemini-1.5-pro")
    
    with open(audio_path, "rb") as audio_file:
        audio_content = audio_file.read()
    
    audio_base64 = base64.b64encode(audio_content).decode("utf-8")
    audio_part = Part.from_data(data=audio_base64, mime_type=mime_type)
    
    transcription_prompt = """Provide a detailed and accurate transcription of this audio recording. Focus on:
    1. Exact words spoken
    2. Capture all speech clearly and accurately
    3. Include natural pauses with "..." notation
    4. Note any significant background sounds in [brackets]
    5. Mark unclear segments with [unclear]
    6. Indicate speaker changes with "Speaker 1:", "Speaker 2:", etc.
    Transcribe the audio as naturally and accurately as possible."""

    config = GenerationConfig(temperature=0.2, top_p=0.95, top_k=40, max_output_tokens=2048)
    response = model.generate_content([transcription_prompt, audio_part], generation_config=config, safety_settings=safety_settings)
    
    return response.text if hasattr(response, 'text') else "Transcription completed but no text was generated."

@retry_with_backoff
def generate_analysis(transcription):
    model = GenerativeModel("gemini-1.5-pro")
    
    analysis_prompt = f"""Based on this transcription, provide a comprehensive analysis:

    Transcription:
    {transcription}

    Analyze and provide:
    1. Summary: Main topics, key points, context, and purpose
    2. Sentiment Analysis: Overall tone, emotion, speaker attitude, and confidence level
    3. Key Insights: Important quotes, action items, and notable speech patterns
    4. Additional Observations: Speech characteristics and relevant background context"""

    config = GenerationConfig(temperature=0.4, top_p=0.8, top_k=40, max_output_tokens=2048)
    response = model.generate_content(analysis_prompt, generation_config=config, safety_settings=safety_settings)
    
    return response.text if hasattr(response, 'text') else "Analysis completed but no text was generated."

safety_settings = [
    SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE),
    SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE),
    SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE),
    SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE),
]

if __name__ == '__main__':
    app.run(debug=True)
