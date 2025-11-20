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


# -------------------- ArUco-based Segmentation --------------------

def get_aruco_dict_and_params():
    """
    Handle both old and new OpenCV ArUco APIs.

    Returns:
        aruco_dict, parameters, use_detector_class (bool)
    """
    try:
        aruco = cv2.aruco
    except AttributeError:
        raise RuntimeError("OpenCV ArUco module not found. Install opencv-contrib-python.")

    # Dictionary getter (newer OpenCV)
    if hasattr(aruco, "getPredefinedDictionary"):
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    # Older OpenCV API
    elif hasattr(aruco, "Dictionary_get"):
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    else:
        raise RuntimeError("ArUco dictionary functions not found in your OpenCV build.")

    # Detector parameters
    if hasattr(aruco, "DetectorParameters_create"):
        parameters = aruco.DetectorParameters_create()
    else:
        parameters = aruco.DetectorParameters()

    # Newer API: ArucoDetector class
    use_detector_class = hasattr(aruco, "ArucoDetector")

    return aruco, aruco_dict, parameters, use_detector_class


def segment_object_with_aruco(img_color):
    """
    Segment a NON-RECTANGULAR object using ArUco markers placed on its boundary.

    Steps:
      1. Detect ArUco markers in the image.
      2. Collect all marker corner points.
      3. Compute the convex hull of all corners to approximate the object's boundary.
      4. Draw this hull on top of the image (boundary overlay).
      5. Create a filled polygon mask from the hull and use it to extract the object region.

    Returns:
      aruco_overlay   : image with detected ArUco markers drawn
      boundary_overlay: image with approximate object boundary (green hull)
      object_only     : original image masked to the object region
    """
    img_bgr = img_color.copy()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    aruco, aruco_dict, parameters, use_detector_class = get_aruco_dict_and_params()

    # --- Detect markers (handle both old & new APIs) ---
    if use_detector_class:
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # ArUco overlay: draw detected markers
    aruco_overlay = img_bgr.copy()
    if ids is not None and len(ids) > 0:
        aruco.drawDetectedMarkers(aruco_overlay, corners, ids)
    else:
        # No markers found – return original as all outputs
        return aruco_overlay, img_bgr, img_bgr

    # --- Collect all corner points ---
    pts = []
    for c in corners:
        # c has shape (1,4,2)
        for p in c[0]:
            pts.append(p)

    pts = np.array(pts, dtype=np.float32)
    if pts.shape[0] < 3:
        # Need at least 3 pts to form a polygon
        return aruco_overlay, img_bgr, img_bgr

    # --- Convex hull as boundary approximation ---
    hull = cv2.convexHull(pts).astype(np.int32)
    hull_for_draw = hull.reshape(-1, 1, 2)

    boundary_overlay = img_bgr.copy()
    cv2.polylines(boundary_overlay, [hull_for_draw], isClosed=True,
                  color=(0, 255, 0), thickness=3)

    # --- Mask and extract object-only region ---
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
    object_only = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

    return aruco_overlay, boundary_overlay, object_only


# -------------------- Flask Routes --------------------

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Upload up to 10 images of a NON-RECTANGULAR object with ArUco markers
    on its boundary (taken from different angles / distances).

    For each image:
      - Save original.
      - Detect ArUco markers and draw them.
      - Build a convex hull of marker corners as the object boundary.
      - Extract the object-only segmentation from that boundary.
    """
    results = []

    if request.method == "POST":
        try:
            files = request.files.getlist("files")
            if not files or len(files) == 0:
                return render_template("index3.html",
                                       error="No files selected")

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

                    try:
                        aruco_overlay, boundary_overlay, object_only = segment_object_with_aruco(img_color)
                    except RuntimeError as e:
                        return render_template("index3.html", error=str(e))

                    # Save images
                    orig_path = os.path.join(app.config["UPLOAD_FOLDER"],
                                             "original_" + filename)
                    aruco_path = os.path.join(app.config["UPLOAD_FOLDER"],
                                              "aruco_" + filename)
                    boundary_path = os.path.join(app.config["UPLOAD_FOLDER"],
                                                 "boundary_overlay_" + filename)
                    object_path = os.path.join(app.config["UPLOAD_FOLDER"],
                                               "object_only_" + filename)

                    cv2.imwrite(orig_path, img_color)
                    cv2.imwrite(aruco_path, aruco_overlay)
                    cv2.imwrite(boundary_path, boundary_overlay)
                    cv2.imwrite(object_path, object_only)

                    results.append({
                        "original": "/" + orig_path,
                        "aruco": "/" + aruco_path,
                        "boundary": "/" + boundary_path,
                        "object_only": "/" + object_path,
                        "filename": filename
                    })

            if not results:
                return render_template("index3.html",
                                       error="No valid images processed")

            return render_template("index3.html", results=results)

        except Exception as e:
            return render_template("index3.html",
                                   error=f"Processing error: {str(e)}")

    # GET
    return render_template("index3.html")


if __name__ == "__main__":
    # Different port from your other apps
    app.run(debug=True, host="0.0.0.0", port=5008, threaded=True)
