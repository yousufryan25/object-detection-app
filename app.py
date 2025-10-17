from flask import Flask, jsonify
import cv2
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
import os
import time
import threading
import requests
import tempfile
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Inisialisasi Firebase
def init_firebase():
    try:
        if not firebase_admin._apps:
            # Untuk Render, simpan credential sebagai environment variable atau file
            cred_path = os.path.join(os.path.dirname(__file__), 'firebase-key.json')
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://cdp1-1e393-default-rtdb.firebaseio.com'
            })
            logger.info("✅ Firebase initialized successfully")
    except Exception as e:
        logger.error(f"❌ Firebase init error: {e}")

# Load model YOLO
def load_model():
    try:
        model = YOLO("yolov8n.pt")  # Ganti dengan path model kamu
        logger.info("✅ YOLO model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"❌ Model loading error: {e}")
        return None

# Global variables
model = load_model()
init_firebase()

def detect_objects(image_url=None):
    """
    Fungsi untuk mendeteksi objek
    """
    try:
        if image_url:
            # Download gambar dari URL
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            # Simpan sementara
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name
        else:
            # Atau gunakan gambar lokal (sesuaikan path)
            temp_path = "static/image.jpg"
            if not os.path.exists(temp_path):
                return [], "Image file not found"

        # Baca gambar
        img = cv2.imread(temp_path)
        if img is None:
            return [], "Failed to read image"

        # Resize untuk optimasi
        img = cv2.resize(img, (640, 480))
        
        # Deteksi objek
        results = model(img, conf=0.25, iou=0.45)
        detected_objects = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    conf = box.conf[0].item()
                    detected_objects.append({
                        "label": label,
                        "confidence": round(conf, 2)
                    })

        # Cleanup
        if image_url and os.path.exists(temp_path):
            os.unlink(temp_path)

        return detected_objects, None

    except Exception as e:
        error_msg = f"Detection error: {str(e)}"
        logger.error(error_msg)
        return None, error_msg

def send_to_firebase(detections, error=None):
    """
    Fungsi untuk mengirim hasil deteksi ke Firebase
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if error:
            data = {
                "timestamp": timestamp,
                "status": "Error",
                "error": error,
                "detections_count": 0
            }
        elif not detections or len(detections) == 0:
            data = {
                "timestamp": timestamp,
                "status": "No objects detected",
                "detections_count": 0
            }
        else:
            data = {
                "timestamp": timestamp,
                "status": "Detected",
                "detections_count": len(detections),
                "first_label": detections[0]['label'],
                "all_detections": detections
            }

        # Kirim ke Firebase
        db.reference('/detections/latest').set(data)
        db.reference('/detections/history').push(data)
        
        logger.info(f"✅ Data sent to Firebase: {data['status']}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to send to Firebase: {e}")
        return False

def detection_worker():
    """
    Worker thread untuk deteksi berkala
    """
    logger.info("🔄 Starting detection worker thread")
    
    # URL gambar dari ESP32-CAM (ganti dengan URL kamu)
    ESP32_CAM_URL = "http://your-esp32-ip/capture"  # Ganti dengan URL kamera
    
    while True:
        try:
            logger.info("🔍 Running detection cycle...")
            
            # Lakukan deteksi
            detections, error = detect_objects(ESP32_CAM_URL)
            
            # Kirim ke Firebase
            send_to_firebase(detections, error)
            
            # Log hasil
            if error:
                logger.warning(f"⚠️ Detection error: {error}")
            else:
                logger.info(f"✅ Detected {len(detections)} objects")
            
            # Tunggu 30 detik sebelum deteksi berikutnya
            time.sleep(30)
            
        except Exception as e:
            logger.error(f"❌ Worker thread error: {e}")
            time.sleep(60)  # Tunggu lebih lama jika error

# Routes untuk Flask
@app.route('/')
def home():
    return """
    <h1>🚀 Object Detection Service</h1>
    <p>Service is running on Render.com</p>
    <p>Endpoints:</p>
    <ul>
        <li><a href="/health">/health</a> - Health check</li>
        <li><a href="/detect">/detect</a> - Manual detection</li>
        <li><a href="/status">/status</a> - Service status</li>
    </ul>
    """

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/detect')
def manual_detect():
    """Endpoint untuk deteksi manual"""
    detections, error = detect_objects()
    send_to_firebase(detections, error)
    
    if error:
        return jsonify({"status": "error", "message": error})
    else:
        return jsonify({
            "status": "success", 
            "detections": detections,
            "count": len(detections)
        })

@app.route('/status')
def status():
    return jsonify({
        "service": "Object Detection",
        "status": "running",
        "model_loaded": model is not None,
        "firebase_connected": len(firebase_admin._apps) > 0,
        "timestamp": datetime.now().isoformat()
    })

# Start worker thread ketika app berjalan
@app.before_first_request
def start_worker():
    thread = threading.Thread(target=detection_worker)
    thread.daemon = True  # Thread akan berhenti ketika main process berhenti
    thread.start()
    logger.info("🎬 Worker thread started")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)