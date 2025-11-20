from flask import Flask, render_template, request
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Folder to store uploaded and processed images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# -------------------- Image Processing Functions --------------------

def detect_edge_keypoints(img_gray, mag_thresh=40):
    """
    Simple EDGE keypoint detector:

      1. Compute image gradients using Sobel (∂I/∂x, ∂I/∂y).
      2. Compute gradient magnitude.
      3. Threshold the magnitude to select strong edge pixels.
      4. Mark those pixels as 'edge keypoints' (green dots).
    """
    # Gradients
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

    mag = cv2.magnitude(gx, gy)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Threshold to select strong edges (lower threshold so we see more)
    _, edge_mask = cv2.threshold(mag_norm, mag_thresh, 255, cv2.THRESH_BINARY)

    # Overlay on image for visualization
    overlay_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    ys, xs = np.where(edge_mask > 0)

    # Draw slightly larger green points for edge keypoints
    for (x, y) in zip(xs, ys):
        cv2.circle(overlay_bgr, (int(x), int(y)), 2, (0, 255, 0), -1)

    return edge_mask, overlay_bgr


def detect_corner_keypoints(img_gray,
                            block_size=3,
                            ksize=3,
                            k=0.04,
                            thresh_rel=0.005):
    """
    Simple CORNER keypoint detector using Harris response:

      1. Use cv2.cornerHarris() to compute Harris response R.
      2. Threshold R to select strong corners.
      3. Mark those pixels as 'corner keypoints' (red circles).
    """
    gray_f32 = np.float32(img_gray)

    # Harris corner response
    harris = cv2.cornerHarris(gray_f32, block_size, ksize, k)

    # Normalize for visualization (optional, not used for overlay)
    response_norm = cv2.normalize(harris, None, 0, 255, cv2.NORM_MINMAX)
    response_norm = np.uint8(response_norm)

    # Threshold relative to max response (slightly lower than before)
    thresh = thresh_rel * harris.max()

    overlay_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    ys, xs = np.where(harris > thresh)

    # Draw corner keypoints as red circles
    for (x, y) in zip(xs, ys):
        cv2.circle(overlay_bgr, (int(x), int(y)), 4, (0, 0, 255), 2)

    return response_norm, overlay_bgr


# -------------------- Flask Routes --------------------

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Upload up to 10 images of the same object (from different angles/distances).

    For each image:
      - Save original color image.
      - Detect and mark EDGE keypoints (gradient-based).
      - Detect and mark CORNER keypoints (Harris-based).

    Results are displayed in index1.html.
    """
    results = []

    if request.method == "POST":
        try:
            files = request.files.getlist("files")
            if not files or len(files) == 0:
                return render_template("index1.html",
                                       error="No files selected")

            # Optional: cap dataset at 10 images
            if len(files) > 10:
                files = files[:10]

            for file in files:
                if file and file.filename:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                    file.save(filepath)

                    img_color = cv2.imread(filepath)
                    if img_color is None:
                        continue

                    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

                    # --- EDGE keypoints ---
                    edge_mask, edge_overlay = detect_edge_keypoints(img_gray)

                    # --- CORNER keypoints ---
                    corner_resp, corner_overlay = detect_corner_keypoints(img_gray)

                    # Save images
                    orig_path = os.path.join(app.config["UPLOAD_FOLDER"],
                                             "original_" + filename)
                    edge_overlay_path = os.path.join(app.config["UPLOAD_FOLDER"],
                                                     "edges_overlay_" + filename)
                    corner_overlay_path = os.path.join(app.config["UPLOAD_FOLDER"],
                                                       "corners_overlay_" + filename)

                    cv2.imwrite(orig_path, img_color)
                    cv2.imwrite(edge_overlay_path, edge_overlay)
                    cv2.imwrite(corner_overlay_path, corner_overlay)

                    results.append({
                        "original": "/" + orig_path,
                        "edges": "/" + edge_overlay_path,
                        "corners": "/" + corner_overlay_path,
                        "filename": filename
                    })

            if not results:
                return render_template("index1.html",
                                       error="No valid images processed")

            return render_template("index1.html", results=results)

        except Exception as e:
            return render_template("index1.html",
                                   error=f"Processing error: {str(e)}")

    # GET
    return render_template("index1.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5006, threaded=True)
