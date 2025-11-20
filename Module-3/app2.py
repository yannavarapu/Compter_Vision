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


# -------------------- Image Processing Function --------------------

def find_object_boundary(img_color):
    """
    Find the boundary of the main object in the image using classic CV:

      1. Convert to grayscale and blur.
      2. Run Canny edge detector.
      3. Close gaps using morphology.
      4. Find contours and pick the largest external contour (object).
      5. Draw the boundary on top of the original image.
      6. Also produce a masked version (only object region).

    Returns:
      edges_vis      : Canny edge image (for debugging/visualization)
      boundary_overlay : original image with object boundary drawn
      object_only      : original image masked to the detected object
    """
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # 1. Smooth slightly to reduce noise
    blur = cv2.GaussianBlur(img_gray, (5, 5), 1.0)

    # 2. Canny edges
    edges = cv2.Canny(blur, 50, 150)

    # 3. Morphological closing to connect fragmented edges
    kernel = np.ones((3, 3), np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4. Find contours
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Prepare outputs
    boundary_overlay = img_color.copy()
    object_only = np.zeros_like(img_color)

    if contours:
        # Assume the object of interest is the largest external contour
        main_contour = max(contours, key=cv2.contourArea)

        # 5. Draw boundary overlay (green contour)
        cv2.drawContours(boundary_overlay, [main_contour], -1, (0, 255, 0), 3)

        # 6. Create mask and masked object image
        mask = np.zeros(img_gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [main_contour], -1, 255, thickness=-1)
        object_only = cv2.bitwise_and(img_color, img_color, mask=mask)
    else:
        # No contour found, keep edges_closed as diagnostic
        pass

    # For visualization, show the processed edges (after closing)
    edges_vis = edges_closed

    return edges_vis, boundary_overlay, object_only


# -------------------- Flask Routes --------------------

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Upload up to 10 images.

    For each image:
      - Save original.
      - Detect object boundary and draw contour overlay.
      - Show object-only view (background suppressed).
    """
    results = []

    if request.method == "POST":
        try:
            files = request.files.getlist("files")
            if not files or len(files) == 0:
                return render_template("index2.html",
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

                    # --- Boundary detection ---
                    edges_vis, boundary_overlay, object_only = find_object_boundary(img_color)

                    # Save images
                    orig_path = os.path.join(app.config["UPLOAD_FOLDER"],
                                             "original_" + filename)
                    edges_path = os.path.join(app.config["UPLOAD_FOLDER"],
                                              "edges_" + filename)
                    boundary_path = os.path.join(app.config["UPLOAD_FOLDER"],
                                                 "boundary_overlay_" + filename)
                    object_path = os.path.join(app.config["UPLOAD_FOLDER"],
                                               "object_only_" + filename)

                    cv2.imwrite(orig_path, img_color)
                    cv2.imwrite(edges_path, edges_vis)
                    cv2.imwrite(boundary_path, boundary_overlay)
                    cv2.imwrite(object_path, object_only)

                    results.append({
                        "original": "/" + orig_path,
                        "edges": "/" + edges_path,
                        "boundary": "/" + boundary_path,
                        "object_only": "/" + object_path,
                        "filename": filename
                    })

            if not results:
                return render_template("index2.html",
                                       error="No valid images processed")

            return render_template("index2.html", results=results)

        except Exception as e:
            return render_template("index2.html",
                                   error=f"Processing error: {str(e)}")

    # GET
    return render_template("index2.html")


if __name__ == "__main__":
    # Use a different port from your other apps (e.g., 5007)
    app.run(debug=True, host="0.0.0.0", port=5007, threaded=True)
