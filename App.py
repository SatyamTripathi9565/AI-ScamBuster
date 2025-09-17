import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
import datetime

# --- कॉन्फ़िगरेशन और इनिशियलाइज़ेशन ---

# मान लें कि मॉडल को 200 कैरेक्टर्स की लंबाई के लिए प्रशिक्षित किया गया था
MAX_LEN = 200

# Flask ऐप और डेटाबेस को इनिशियलाइज़ करें
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///urls.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- मॉडल और वोकैब्युलरी लोड करें ---

# अपनी मॉडल फ़ाइल का सही पाथ दें
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

# --- प्रीप्रोसेसिंग फंक्शन ---

def preprocess_url(url):
    """URL स्ट्रिंग को मॉडल के लिए तैयार इनपुट में बदलता है।"""
    # URL को कैरेक्टर IDs में बदलें
    sequence = [char_to_int.get(char, 1) for char in url] # 1 [UNK] के लिए है
    # सीक्वेंस को पैड करें
    padded_sequence = pad_sequences([sequence], maxlen=MAX_LEN, padding='post')
    return padded_sequence

# --- API रूट्स ---

@app.route('/')
def home():
    """फ्रंटएंड HTML पेज को रेंडर करता है।"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """URL पर भविष्यवाणी करता है।"""
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'URL नहीं दिया गया'}), 400

    user_url = data['url']
    
    # URL को प्रीप्रोसेस करें
    processed_url = preprocess_url(user_url)
    
    # भविष्यवाणी करें
    prediction_prob = model.predict(processed_url)[0][0]
    
    # परिणाम तय करें (मान लें कि 0.5 थ्रेसहोल्ड है)
    if prediction_prob > 0.5:
        result_text = "Malicious"
    else:
        result_text = "Benign"

    # डेटाबेस में लॉग सेव करें
    new_log = UrlLog(url=user_url, prediction=result_text)
    db.session.add(new_log)
    db.session.commit()
    
    # फ्रंटएंड को परिणाम भेजें
    return jsonify({
        'url': user_url,
        'prediction': result_text,
        'confidence': float(prediction_prob)
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # ऐप शुरू होने पर डेटाबेस टेबल बनाएं
    app.run(debug=True)