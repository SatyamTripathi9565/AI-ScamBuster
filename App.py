import json
import re
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
import datetime
import google.generativeai as genai

# Gemini API key set kar
genai.configure(api_key="AIzaSyCeHeZ_idGr2OW415D4lcd2JlYkIuIj74Q")

gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# --- कॉन्फ़िगरेशन और इनिशियलाइज़ेशन ---
MAX_LEN = 200

# Flask ऐप और डेटाबेस को इनिशियलाइज़ करें
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///urls.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- मॉडल और वोकैब्युलरी लोड करें ---
model = load_model('model/url_cnn_lstm.h5')
with open('model/char_vocab.json', 'r') as f:
    char_to_int = json.load(f)

print("✅ मॉडल और वोकैब्युलरी सफलतापूर्वक लोड हो गए।")

# --- डेटाबेस मॉडल डेफिनिशन ---
class UrlLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(500), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f'<URL {self.url}>'

# --- URL वैलिडेशन फंक्शन ---
def is_valid_url(url):
    """
    यह function चेक करता है कि URL सही format में है या नहीं।
    जैसे: https://example.com या www.google.com
    """
    pattern = re.compile(r'^(https?:\/\/)?([\w\-]+\.)+[a-zA-Z]{2,}(\/.*)?$')
    return re.match(pattern, url) is not None

# --- प्रीप्रोसेसिंग फंक्शन ---
def preprocess_url(url):
    """URL स्ट्रिंग को मॉडल के लिए तैयार इनपुट में बदलता है।"""
    sequence = [char_to_int.get(char, 1) for char in url]  # 1 unknown chars के लिए
    padded_sequence = pad_sequences([sequence], maxlen=MAX_LEN, padding='post')
    return padded_sequence

@app.route('/')
def home():
    """फ्रंटएंड HTML पेज को रेंडर करता है।"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """URL पर भविष्यवाणी करता है और Gemini से जानकारी प्राप्त करता है।"""
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'URL नहीं दिया गया'}), 400

    user_url = data['url'].strip()

    # 1️⃣ Invalid URL check
    if not is_valid_url(user_url):
        return jsonify({
            'url': user_url,
            'prediction': 'Invalid Input',
            'confidence': 0.0,
            'explanation': 'The input does not appear to be a valid URL. Please enter a proper URL like https://example.com.'
        })

    # 2️⃣ Preprocess the URL for model
    processed_url = preprocess_url(user_url)

    # 3️⃣ Predict with model
    prediction_prob = float(model.predict(processed_url)[0][0])

# Confidence-based classification
    if 0.35 < prediction_prob < 0.65:
        result_text = "Uncertain"
    elif prediction_prob >= 0.65:
        result_text = "Malicious"
    else:
        result_text = "Benign"

    # 4️⃣ Gemini explanation
    try:
        prompt = f"""
        The system analyzed the URL: '{user_url}'.
        The model predicted that it is '{result_text}' with confidence {prediction_prob:.2f}.
        Please explain why this classification might be correct,
        and describe common characteristics of such URLs in simple terms.
        """
        gemini_response = gemini_model.generate_content(prompt)
        explanation = gemini_response.text
    except Exception as e:
        explanation = f"Gemini API error: {str(e)}"

    # 5️⃣ Save to DB
    new_log = UrlLog(url=user_url, prediction=result_text)
    db.session.add(new_log)
    db.session.commit()

    # 6️⃣ Return result
    return jsonify({
        'url': user_url,
        'prediction': result_text,
        'confidence': prediction_prob,
        'explanation': explanation
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()

    # ----- NGROK removed on purpose (run locally) -----
    # To run locally, open: http://127.0.0.1:5000 after starting this app.
    app.run(host="0.0.0.0", port=5000, debug=True)
