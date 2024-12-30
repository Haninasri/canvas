import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import base64
import pytesseract

app = Flask(__name__)

# Function to detect pearls using HoughCircles
def detect_pearls(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise Exception("Image file could not be read.")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detecting circles (pearls) using HoughCircles
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
            param1=50, param2=30, minRadius=20, maxRadius=50
        )
        
        pearls = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for circle in circles:
                x, y, r = circle
                pearls.append({'x': int(x), 'y': int(y), 'radius': int(r)})
        
        return img, pearls
    except Exception as e:
        return None, str(e)

# Function to detect and color text in the image using pytesseract
def detect_and_color_text(image_path, color):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect text using pytesseract
    boxes = pytesseract.image_to_boxes(gray)
    
    for box in boxes.splitlines():
        b = box.split()
        x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        
        # Draw a colored rectangle over detected text
        cv2.rectangle(img, (x, img.shape[0] - y), (w, img.shape[0] - h), color, -1)
    
    return img

# Main route to render the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload
@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Save the uploaded image file
        image_path = 'uploads/' + image_file.filename
        os.makedirs('uploads', exist_ok=True)
        image_file.save(image_path)

        # Detect pearls from the uploaded image
        img, pearls = detect_pearls(image_path)
        if img is None:
            return jsonify({'error': pearls}), 500

        # Convert the image to base64
        _, buffer = cv2.imencode('.png', img)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({'image': img_str, 'pearls': pearls})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route to handle saving image
@app.route('/save_image', methods=['POST'])
def save_image():
    try:
        data = request.json
        image_data = base64.b64decode(data['image'])
        save_dir = data['folder']  # The selected folder name (not path)
        save_file_name = data['file_name']  # The file name to save as

        # Ensure directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Save the image to the provided path
        save_path = os.path.join(save_dir, save_file_name)
        with open(save_path, 'wb') as f:
            f.write(image_data)

        return jsonify({'message': f'Image saved successfully to {save_path}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# Route to handle coloring the text
@app.route('/color_text', methods=['POST'])
def color_text():
    try:
        data = request.json
        image_path = data['image_path']
        color = tuple(int(data['color'][i:i+2], 16) for i in (1, 3, 5))  # Hex to RGB

        # Color the text in the image
        colored_img = detect_and_color_text(image_path, color)
        
        # Save or return the colored image
        _, buffer = cv2.imencode('.png', colored_img)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({'image': img_str})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
