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

def compute_gradients(img_gray):
    """
    Compute gradient magnitude and direction (angle) using Sobel filters.

    This corresponds to:
      - ∂L/∂x and ∂L/∂y from the image L
      - Gradient magnitude:  sqrt(Gx^2 + Gy^2)
      - Gradient angle:      arctan2(Gy, Gx)  (in degrees)
    """
    # First derivatives in x and y
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

    # Magnitude & angle
    magnitude = cv2.magnitude(sobelx, sobely)
    angle = cv2.phase(sobelx, sobely, angleInDegrees=True)

    # Normalize both to [0,255] for visualization
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    angle_norm = cv2.normalize(angle, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return magnitude_norm, angle_norm


def compute_log(img_gray, sigma=1.4):
    """
    Compute Laplacian of Gaussian (LoG).

    Steps:
      1. Smooth the image with a Gaussian (scale controlled by sigma).
      2. Apply Laplacian operator to detect regions of rapid intensity change.

    Result is normalized for display, so it can be visually compared
    with gradient magnitude images.
    """
    # Kernel size chosen from sigma (~6*sigma+1 is a common rule)
    k = int(6 * sigma + 1)
    if k % 2 == 0:
        k += 1

    blurred = cv2.GaussianBlur(img_gray, (k, k), sigmaX=sigma, sigmaY=sigma)
    log = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)

    # Take absolute value (edges can be positive/negative) and normalize
    log_abs = np.abs(log)
    log_norm = cv2.normalize(log_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return log_norm


# -------------------- Flask Routes --------------------

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Upload up to 10 images (same object from different angles/distances).

    For each image in the dataset:
      - Save original
      - Compute and save gradient magnitude
      - Compute and save gradient angle
      - Compute and save Laplacian-of-Gaussian (LoG) filtered image

    The HTML template can display these side by side for comparison.
    """
    results = []  # To store results for each uploaded image

    if request.method == "POST":
        try:
            files = request.files.getlist("files")  # multiple files input: name="files"
            if not files or len(files) == 0:
                return render_template("index.html", error="No files selected")

            # Optional: enforce dataset size ≈ 10 images
            if len(files) > 10:
                files = files[:10]

            for file in files:
                if file and file.filename:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                    file.save(filepath)

                    # Read image
                    img_color = cv2.imread(filepath)
                    if img_color is None:
                        continue

                    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

                    # --- Compute gradient magnitude & angle ---
                    grad_mag, grad_angle = compute_gradients(img_gray)

                    # --- Compute Laplacian of Gaussian ---
                    log_img = compute_log(img_gray, sigma=1.4)

                    # Save outputs
                    orig_path = os.path.join(app.config["UPLOAD_FOLDER"], "original_" + filename)
                    mag_path = os.path.join(app.config["UPLOAD_FOLDER"], "gradient_magnitude_" + filename)
                    angle_path = os.path.join(app.config["UPLOAD_FOLDER"], "gradient_angle_" + filename)
                    log_path = os.path.join(app.config["UPLOAD_FOLDER"], "log_" + filename)

                    cv2.imwrite(orig_path, img_color)
                    cv2.imwrite(mag_path, grad_mag)
                    cv2.imwrite(angle_path, grad_angle)
                    cv2.imwrite(log_path, log_img)

                    # Store paths for template rendering
                    results.append({
                        "original": "/" + orig_path,
                        "grad_mag": "/" + mag_path,
                        "grad_angle": "/" + angle_path,
                        "log_img": "/" + log_path,
                        "filename": filename
                    })

            if not results:
                return render_template("index.html", error="No valid images processed")

            # Render page showing all dataset images + their gradient & LoG versions
            return render_template("index.html", results=results)

        except Exception as e:
            return render_template("index.html", error=f"Processing error: {str(e)}")

    # GET
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5005, threaded=True)
