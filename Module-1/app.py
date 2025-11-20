from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Store uploaded image temporarily
current_image = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Handle image upload"""
    global current_image

    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Read image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400

        # Store image and convert to base64 for display
        current_image = img
        height, width = img.shape[:2]

        # Convert image to base64 for display
        _, buffer = cv2.imencode('.png', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        image_data = f'data:image/png;base64,{img_base64}'

        return jsonify({
            'success': True,
            'width': int(width),
            'height': int(height),
            'image_data': image_data,
            'message': 'Image uploaded successfully'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/calculate_dimensions', methods=['POST'])
def calculate_dimensions():
    """Calculate real-world dimensions using perspective projection from two points"""
    global current_image

    try:
        data = request.json

        # Extract two points (top-left and bottom-right)
        x1 = int(data['x1'])
        y1 = int(data['y1'])
        x2 = int(data['x2'])
        y2 = int(data['y2'])

        # Camera intrinsic parameters
        fx = float(data['fx'])  # Focal length in x (pixels)
        fy = float(data['fy'])  # Focal length in y (pixels)
        cx = float(data['cx'])  # Principal point x (pixels)
        cy = float(data['cy'])  # Principal point y (pixels)
        Z = float(data['Z'])    # Depth from camera to object (mm)

        # Perspective Projection Equations (Inverse):
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy

        # Calculate 3D coordinates for both points
        X1 = (x1 - cx) * Z / fx
        Y1 = (y1 - cy) * Z / fy

        X2 = (x2 - cx) * Z / fx
        Y2 = (y2 - cy) * Z / fy

        # Calculate real-world dimensions
        width_mm = abs(X2 - X1)
        height_mm = abs(Y2 - Y1)

        # Convert to different units
        width_cm = width_mm / 10
        height_cm = height_mm / 10
        width_inch = width_mm / 25.4
        height_inch = height_mm / 25.4

        # Calculate diagonal
        diagonal_mm = np.sqrt(width_mm**2 + height_mm**2)
        diagonal_cm = diagonal_mm / 10
        diagonal_inch = diagonal_mm / 25.4

        # Calculate pixel dimensions
        pixel_width = abs(x2 - x1)
        pixel_height = abs(y2 - y1)

        # Draw rectangle and points on image
        annotated_image = None
        if current_image is not None:
            img_copy = current_image.copy()

            # Ensure correct corner coordinates
            top_left = (min(x1, x2), min(y1, y2))
            bottom_right = (max(x1, x2), max(y1, y2))

            # Draw rectangle
            cv2.rectangle(img_copy, top_left, bottom_right, (0, 255, 0), 3)

            # Draw points
            cv2.circle(img_copy, (x1, y1), 6, (255, 0, 0), -1)
            cv2.circle(img_copy, (x2, y2), 6, (255, 0, 0), -1)

            # Add point labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_copy, 'P1', (x1 - 20, y1 - 10), font, 0.6, (255, 0, 0), 2)
            cv2.putText(img_copy, 'P2', (x2 + 10, y2 + 20), font, 0.6, (255, 0, 0), 2)

            # Add dimension labels
            cv2.putText(img_copy, f'W: {width_mm:.2f} mm',
                        (top_left[0] + 5, top_left[1] - 10),
                        font, 0.6, (0, 255, 0), 2)
            cv2.putText(img_copy, f'H: {height_mm:.2f} mm',
                        (top_left[0] + 5, bottom_right[1] + 25),
                        font, 0.6, (0, 255, 0), 2)

            # Convert to base64
            _, buffer = cv2.imencode('.png', img_copy)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            annotated_image = f'data:image/png;base64,{img_base64}'

        return jsonify({
            'success': True,
            'dimensions': {
                'width_mm': round(width_mm, 2),
                'height_mm': round(height_mm, 2),
                'width_cm': round(width_cm, 2),
                'height_cm': round(height_cm, 2),
                'width_inch': round(width_inch, 3),
                'height_inch': round(height_inch, 3),
                'diagonal_mm': round(diagonal_mm, 2),
                'diagonal_cm': round(diagonal_cm, 2),
                'diagonal_inch': round(diagonal_inch, 3),
                'pixel_width': pixel_width,
                'pixel_height': pixel_height
            },
            'annotated_image': annotated_image,
            'calculations': {
                'X1': round(X1, 2),
                'Y1': round(Y1, 2),
                'X2': round(X2, 2),
                'Y2': round(Y2, 2),
                'formula': 'X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy'
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/load_calibration', methods=['POST'])
def load_calibration():
    """Load camera calibration from JSON file"""
    try:
        if 'calibration' not in request.files:
            return jsonify({'error': 'No calibration file provided'}), 400

        file = request.files['calibration']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        import json
        calib_data = json.load(file)

        return jsonify({
            'success': True,
            'fx': calib_data.get('fx', 0),
            'fy': calib_data.get('fy', 0),
            'cx': calib_data.get('cx', 0),
            'cy': calib_data.get('cy', 0)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Flask Point Selection Dimension Calculator")
    print("=" * 60)
    print("Server starting on http://127.0.0.1:5001")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5001)