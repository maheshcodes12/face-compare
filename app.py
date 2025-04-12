from flask import Flask, request, jsonify
from deepface import DeepFace
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/compare-faces', methods=['POST'])
def compare_faces():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({"error": "Both image1 and image2 files are required"}), 400
    
    file1 = request.files['image1']
    file2 = request.files['image2']
    
    if file1.filename == '' or file2.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename):
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        
        file1.save(filepath1)
        file2.save(filepath2)
        
        try:
            # Use DeepFace for verification
            result = DeepFace.verify(
                img1_path=filepath1,
                img2_path=filepath2,
                model_name="Facenet",
                detector_backend="opencv",
                enforce_detection=False
            )
            
            # Clean up uploaded files
            os.remove(filepath1)
            os.remove(filepath2)
            
            return jsonify({
                "verified": bool(result["verified"]),
                "similarity": float(1-result["distance"])*100,
                "threshold": float(result["threshold"]),
                "model": "Facenet",
                "message": "The images are {} with {:.2f}% similarity".format(
                    "similar" if result["verified"] else "different",
                    (1 - result["distance"]) * 100
                )
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Invalid file type"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)