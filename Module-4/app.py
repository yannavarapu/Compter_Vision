from flask import Flask, render_template, request, jsonify, url_for
import numpy as np
import cv2
import cv2 as cv
import base64
import json
import os
from datetime import datetime
from werkzeug.utils import secure_filename
import math


app = Flask(__name__)

# =========================================================
# HOME / NAVIGATOR
# =========================================================

@app.route("/")
def home():
    # Main dropdown page (index.html)
    return render_template("index.html")


# =========================================================
# MODULE 1 – Question 1 & 2
#   Page:        /module1           -> Module1.html
#   Endpoints:   /module1/upload_image
#                /module1/calculate_dimensions
#                /module1/load_calibration
# =========================================================

current_image = None   # store uploaded image for module 1


@app.route("/module1")
def module1_page():
    return render_template("Module1.html")


@app.route("/module1/upload_image", methods=["POST"])
def module1_upload_image():
    """Handle image upload (Module 1)"""
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


@app.route("/module1/calculate_dimensions", methods=["POST"])
def module1_calculate_dimensions():
    """Calculate real-world dimensions using perspective projection (Module 1)"""
    global current_image

    try:
        data = request.json

        # Extract two points (top-left and bottom-right)
        x1 = int(data['x1'])
        y1 = int(data['y1'])
        x2 = int(data['x2'])
        y2 = int(data['y2'])

        # Camera intrinsic parameters
        fx = float(data['fx'])
        fy = float(data['fy'])
        cx = float(data['cx'])
        cy = float(data['cy'])
        Z = float(data['Z'])   # depth (mm)

        # Inverse perspective projection:
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        X1 = (x1 - cx) * Z / fx
        Y1 = (y1 - cy) * Z / fy
        X2 = (x2 - cx) * Z / fx
        Y2 = (y2 - cy) * Z / fy

        width_mm = abs(X2 - X1)
        height_mm = abs(Y2 - Y1)

        width_cm = width_mm / 10
        height_cm = height_mm / 10
        width_inch = width_mm / 25.4
        height_inch = height_mm / 25.4

        diagonal_mm = np.sqrt(width_mm ** 2 + height_mm ** 2)
        diagonal_cm = diagonal_mm / 10
        diagonal_inch = diagonal_mm / 25.4

        pixel_width = abs(x2 - x1)
        pixel_height = abs(y2 - y1)

        annotated_image = None
        if current_image is not None:
            img_copy = current_image.copy()

            top_left = (min(x1, x2), min(y1, y2))
            bottom_right = (max(x1, x2), max(y1, y2))

            # Draw rectangle
            cv2.rectangle(img_copy, top_left, bottom_right, (0, 255, 0), 3)

            # Draw points
            cv2.circle(img_copy, (x1, y1), 6, (255, 0, 0), -1)
            cv2.circle(img_copy, (x2, y2), 6, (255, 0, 0), -1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_copy, 'P1', (x1 - 20, y1 - 10), font, 0.6, (255, 0, 0), 2)
            cv2.putText(img_copy, 'P2', (x2 + 10, y2 + 20), font, 0.6, (255, 0, 0), 2)

            cv2.putText(
                img_copy,
                f'W: {width_mm:.2f} mm',
                (top_left[0] + 5, top_left[1] - 10),
                font, 0.6, (0, 255, 0), 2
            )
            cv2.putText(
                img_copy,
                f'H: {height_mm:.2f} mm',
                (top_left[0] + 5, bottom_right[1] + 25),
                font, 0.6, (0, 255, 0), 2
            )

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


@app.route("/module1/load_calibration", methods=["POST"])
def module1_load_calibration():
    """Load camera calibration JSON (Module 1)"""
    try:
        if 'calibration' not in request.files:
            return jsonify({'error': 'No calibration file provided'}), 400

        file = request.files['calibration']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

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


# =========================================================
# MODULE 2 – Q1 & Q3 (Template Matching + Blur)
#   Page:        /module2/q1        -> M2Q1.html
#   Endpoints:   /module2/upload_source
#                /module2/upload_template
#                /module2/process_all
#                /module2/health
# =========================================================

SOURCE_FOLDER = 'source_images'
TEMPLATE_FOLDER = 'template_images'
RESULTS_FOLDER = 'results'

for folder in [SOURCE_FOLDER, TEMPLATE_FOLDER, RESULTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB


def m2_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg']


def m2_image_to_base64(image):
    try:
        _, buffer = cv.imencode('.jpg', image, [cv.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None


def rotate_keep_all(gray, angle):
    """Rotate image by `angle` degrees, expanding canvas so nothing is clipped."""
    rows, cols = gray.shape[:2]
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nW = int(rows * sin + cols * cos)
    nH = int(rows * cos + cols * sin)
    M[0, 2] += (nW / 2) - cols / 2
    M[1, 2] += (nH / 2) - rows / 2
    return cv.warpAffine(
        gray, M, (nW, nH),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_REPLICATE
    )


def blur_regions(img_bgr, matches, ksize=31, sigma=15):
    """Blur only the matched rectangles in img_bgr."""
    out = img_bgr.copy()
    if not matches:
        return out

    k = ksize | 1  # ensure odd
    for m in matches:
        x, y, w, h = m['x'], m['y'], m['width'], m['height']
        x = max(0, x)
        y = max(0, y)
        w = max(1, min(w, out.shape[1] - x))
        h = max(1, min(h, out.shape[0] - y))

        roi = out[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        blurred_roi = cv.GaussianBlur(roi, (k, k), sigma)
        out[y:y + h, x:x + w] = blurred_roi

    return out


@app.route("/module2/q1")
def module2_q1_page():
    return render_template("M2Q1.html")


@app.route("/module2/upload_source", methods=["POST"])
def upload_source():
    """Upload source image for Module 2"""
    try:
        if 'source' not in request.files:
            return jsonify({'error': 'No source file provided'}), 400

        file = request.files['source']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not m2_allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Use JPG, JPEG, or PNG'}), 400

        filepath = os.path.join(SOURCE_FOLDER, 'source.jpg')
        file.save(filepath)

        img = cv.imread(filepath)
        if img is None:
            return jsonify({'error': 'Failed to read image file'}), 400

        h, w = img.shape[:2]
        img_base64 = m2_image_to_base64(img)

        return jsonify({
            'success': True,
            'filename': 'source.jpg',
            'width': int(w),
            'height': int(h),
            'image': f'data:image/jpeg;base64,{img_base64}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/module2/upload_template", methods=["POST"])
def upload_template():
    """Upload template image for Module 2"""
    try:
        if 'template' not in request.files:
            return jsonify({'error': 'No template file provided'}), 400

        file = request.files['template']
        template_id = request.form.get('template_id', '1')

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not m2_allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Use JPG, JPEG, or PNG'}), 400

        filename = f'template_{template_id}.jpg'
        filepath = os.path.join(TEMPLATE_FOLDER, filename)
        file.save(filepath)

        img = cv.imread(filepath)
        if img is None:
            return jsonify({'error': 'Failed to read template image'}), 400

        h, w = img.shape[:2]
        img_base64 = m2_image_to_base64(img)

        return jsonify({
            'success': True,
            'template_id': template_id,
            'filename': filename,
            'width': int(w),
            'height': int(h),
            'image': f'data:image/jpeg;base64,{img_base64}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/module2/process_all", methods=["POST"])
def process_all():
    """Multi-template detection with blur (Module 2 Q1 & Q3)"""
    try:
        data = request.json or {}
        SCORE_THRESH = float(data.get('threshold', 0.60))

        METHOD = cv.TM_CCOEFF_NORMED
        SCALES = np.linspace(0.5, 1.4, 19)
        ANGLES = [0, 180]

        source_path = os.path.join(SOURCE_FOLDER, 'source.jpg')
        if not os.path.exists(source_path):
            return jsonify({'error': 'Please upload source image first'}), 400

        img_gray = cv.imread(source_path, cv.IMREAD_GRAYSCALE)
        if img_gray is None:
            return jsonify({'error': 'Failed to read source image'}), 500

        img_gray = cv.GaussianBlur(img_gray, (3, 3), 0)
        H, W = img_gray.shape[:2]

        base_vis = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)

        template_files = []
        for i in range(1, 11):
            tpath = os.path.join(TEMPLATE_FOLDER, f'template_{i}.jpg')
            if os.path.exists(tpath):
                template_files.append((i, tpath))

        if len(template_files) == 0:
            return jsonify({'error': 'Please upload at least one template'}), 400

        colors = [
            (0, 255, 0), (0, 180, 255), (255, 160, 0), (255, 0, 120),
            (120, 255, 120), (160, 120, 255), (200, 200, 0), (0, 220, 180),
            (255, 200, 200), (200, 255, 200)
        ]

        results = []

        for idx, (template_id, tpath) in enumerate(template_files):
            tpl = cv.imread(tpath, cv.IMREAD_GRAYSCALE)
            if tpl is None:
                continue

            tpl = cv.GaussianBlur(tpl, (3, 3), 0)
            rows, cols = tpl.shape[:2]

            best_score = -1.0
            best = None
            best_res = None

            for ang in ANGLES:
                tpl_rot = rotate_keep_all(tpl, ang)
                for s in SCALES:
                    tw = max(5, int(tpl_rot.shape[1] * s))
                    th = max(5, int(tpl_rot.shape[0] * s))
                    if tw >= W or th >= H:
                        continue

                    tpl_scaled = cv.resize(tpl_rot, (tw, th), interpolation=cv.INTER_AREA)
                    res = cv.matchTemplate(img_gray, tpl_scaled, METHOD)
                    _, max_val, _, max_loc = cv.minMaxLoc(res)

                    if max_val > best_score:
                        best_score = float(max_val)
                        best = (max_loc, (tw, th), float(s), ang)
                        best_res = res

            name = os.path.basename(tpath)
            name_noext = os.path.splitext(name)[0]

            if best is None:
                matches = []
                num_matches = 0
                correlation = np.zeros((10, 10), dtype=np.float32)
            else:
                (x, y), (w, h), s, ang = best

                matches = []
                if best_score >= SCORE_THRESH:
                    matches.append({
                        'x': int(x),
                        'y': int(y),
                        'width': int(w),
                        'height': int(h),
                        'confidence': float(best_score),
                        'scale': float(s),
                        'angle': int(ang)
                    })
                    num_matches = 1
                else:
                    num_matches = 0

                correlation = best_res if best_res is not None else np.zeros((10, 10), dtype=np.float32)

            result_img = base_vis.copy()
            if len(matches) > 0:
                m = matches[0]
                color = colors[idx % len(colors)]
                x, y, w, h = m['x'], m['y'], m['width'], m['height']
                cv.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
                cv.putText(
                    result_img,
                    f"{name_noext} {m['confidence']:.2f}",
                    (x, max(15, y - 6)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    1,
                    cv.LINE_AA
                )

            blurred_img = blur_regions(base_vis, matches, ksize=31, sigma=15)

            if correlation is not None and correlation.size > 0:
                max_corr = float(correlation.max())
                min_corr = float(correlation.min())
                mean_corr = float(correlation.mean())
            else:
                max_corr = min_corr = mean_corr = 0.0

            tpl_bgr = cv.imread(tpath)
            tpl_b64 = m2_image_to_base64(tpl_bgr) if tpl_bgr is not None else None
            res_b64 = m2_image_to_base64(result_img)
            blur_b64 = m2_image_to_base64(blurred_img)

            results.append({
                'template_id': template_id,
                'template_name': f'Template_{template_id}',
                'template_image': f'data:image/jpeg;base64,{tpl_b64}' if tpl_b64 else None,
                'num_matches': num_matches,
                'matches': matches,
                'result_image': f'data:image/jpeg;base64,{res_b64}',
                'heatmap_image': f'data:image/jpeg;base64,{blur_b64}',
                'max_correlation': max_corr,
                'min_correlation': min_corr,
                'mean_correlation': mean_corr
            })

        response = {
            'success': True,
            'method': 'TM_CCOEFF_NORMED',
            'threshold': SCORE_THRESH,
            'total_templates': len(results),
            'total_matches': sum(r['num_matches'] for r in results),
            'results': results
        }

        return jsonify(response)

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route("/module2/health", methods=["GET"])
def module2_health():
    return jsonify({'status': 'ok'})


# =========================================================
# SHARED UPLOAD FOLDER (Module 2 Q2 + Module 3)
# =========================================================

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# =========================================================
# MODULE 2 – Q2 (Gaussian Blur + Fourier Deblur)
#   Page:  /module2/q2       -> M2Q2.html
# =========================================================

SIGMA = 3.0
KERNEL_SIZE = 19
MODE = "wiener"          # "wiener" or "inverse"
K_WIENER = 0.001


def ensure_odd(k):
    k = int(k)
    return k if k % 2 == 1 else k + 1


def gaussian_psf(ksize, sigma):
    ax = np.arange(ksize) - (ksize - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    psf /= psf.sum()
    return psf


def psf_to_otf(psf, shapeHW):
    H, W = shapeHW
    pad = np.zeros((H, W), np.float32)
    psf_shifted = np.fft.ifftshift(psf)
    pad[:psf.shape[0], :psf.shape[1]] = psf_shifted
    return np.fft.fft2(pad)


def wiener_deconv(G, H, K):
    return (np.conj(H) / (np.abs(H) ** 2 + K)) * G


def inverse_deconv(G, H, eps=1e-6):
    return G / (H + eps)


def to_float01(img):
    return img.astype(np.float32) / 255.0


def to_uint8(x):
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def gaussian_blur(img_color, ksize=KERNEL_SIZE, sigma=SIGMA):
    L = to_float01(img_color)
    H, W = L.shape[:2]

    ksize = ensure_odd(ksize)
    psf = gaussian_psf(ksize, sigma)
    OTF = psf_to_otf(psf, (H, W))

    L_b = np.zeros_like(L, dtype=np.float32)
    for c in range(3):
        F = np.fft.fft2(L[:, :, c])
        G = F * OTF
        L_b[:, :, c] = np.fft.ifft2(G).real

    return to_uint8(L_b)


def fourier_deblur_color(blurred, sigma=SIGMA, K=K_WIENER, mode=MODE):
    L_b = to_float01(blurred)
    H, W = L_b.shape[:2]

    ksize = ensure_odd(KERNEL_SIZE)
    psf = gaussian_psf(ksize, sigma)
    OTF = psf_to_otf(psf, (H, W))

    L_rec = np.zeros_like(L_b, dtype=np.float32)
    for c in range(3):
        G = np.fft.fft2(L_b[:, :, c])
        if mode == "wiener":
            Fhat = wiener_deconv(G, OTF, K)
        else:
            Fhat = inverse_deconv(G, OTF, 1e-6)
        rec_c = np.fft.ifft2(Fhat).real
        L_rec[:, :, c] = np.clip(rec_c, 0.0, 1.0)

    return to_uint8(L_rec)


@app.route("/module2/q2", methods=["GET", "POST"])
def module2_q2_page():
    if request.method == "POST":
        try:
            file = request.files.get('file', None)
            if not file or file.filename == "":
                return render_template("M2Q2.html", error="No file selected")

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = cv2.imread(filepath, cv2.IMREAD_COLOR)
            if img is None:
                return render_template("M2Q2.html", error="Invalid image format")

            blurred = gaussian_blur(img)
            recovered = fourier_deblur_color(blurred)

            original_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original_' + filename)
            blurred_path = os.path.join(app.config['UPLOAD_FOLDER'], 'blurred_' + filename)
            recovered_path = os.path.join(app.config['UPLOAD_FOLDER'], 'recovered_' + filename)

            cv2.imwrite(original_path, img)
            cv2.imwrite(blurred_path, blurred)
            cv2.imwrite(recovered_path, recovered)

            return render_template(
                "M2Q2.html",
                original='/' + original_path,
                blurred='/' + blurred_path,
                recovered='/' + recovered_path,
                sigma=SIGMA,
                kernel_size=ensure_odd(KERNEL_SIZE),
                mode=MODE,
                K=K_WIENER
            )

        except Exception as e:
            return render_template("M2Q2.html", error=f"Processing error: {str(e)}")

    return render_template("M2Q2.html")


# =========================================================
# MODULE 3 – Q1 (Gradient & LoG Dataset Visualizer)
#   Page:  /module3/q1   -> M3Q1.html
# =========================================================

def compute_gradients(img_gray):
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(sobelx, sobely)
    angle = cv2.phase(sobelx, sobely, angleInDegrees=True)

    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    angle_norm = cv2.normalize(angle, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return magnitude_norm, angle_norm


def compute_log(img_gray, sigma=1.4):
    k = int(6 * sigma + 1)
    if k % 2 == 0:
        k += 1

    blurred = cv2.GaussianBlur(img_gray, (k, k), sigmaX=sigma, sigmaY=sigma)
    log = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)

    log_abs = np.abs(log)
    log_norm = cv2.normalize(log_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return log_norm


@app.route("/module3/q1", methods=["GET", "POST"])
def module3_q1_page():
    results = []

    if request.method == "POST":
        try:
            files = request.files.getlist("files")
            if not files or len(files) == 0:
                return render_template("M3Q1.html", error="No files selected")

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

                    grad_mag, grad_angle = compute_gradients(img_gray)
                    log_img = compute_log(img_gray, sigma=1.4)

                    orig_path = os.path.join(app.config["UPLOAD_FOLDER"], "original_" + filename)
                    mag_path = os.path.join(app.config["UPLOAD_FOLDER"], "gradient_magnitude_" + filename)
                    angle_path = os.path.join(app.config["UPLOAD_FOLDER"], "gradient_angle_" + filename)
                    log_path = os.path.join(app.config["UPLOAD_FOLDER"], "log_" + filename)

                    cv2.imwrite(orig_path, img_color)
                    cv2.imwrite(mag_path, grad_mag)
                    cv2.imwrite(angle_path, grad_angle)
                    cv2.imwrite(log_path, log_img)

                    results.append({
                        "original": "/" + orig_path,
                        "grad_mag": "/" + mag_path,
                        "grad_angle": "/" + angle_path,
                        "log_img": "/" + log_path,
                        "filename": filename
                    })

            if not results:
                return render_template("M3Q1.html", error="No valid images processed")

            return render_template("M3Q1.html", results=results)

        except Exception as e:
            return render_template("M3Q1.html", error=f"Processing error: {str(e)}")

    return render_template("M3Q1.html")


# =========================================================
# MODULE 3 – Q2 (Edge & Corner Keypoint Detector)
#   Page:  /module3/q2   -> M3Q2.html
# =========================================================

def detect_edge_keypoints(img_gray, mag_thresh=40):
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

    mag = cv2.magnitude(gx, gy)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, edge_mask = cv2.threshold(mag_norm, mag_thresh, 255, cv2.THRESH_BINARY)

    overlay_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    ys, xs = np.where(edge_mask > 0)
    for (x, y) in zip(xs, ys):
        cv2.circle(overlay_bgr, (int(x), int(y)), 2, (0, 255, 0), -1)

    return edge_mask, overlay_bgr


def detect_corner_keypoints(img_gray,
                            block_size=3,
                            ksize=3,
                            k=0.04,
                            thresh_rel=0.005):
    gray_f32 = np.float32(img_gray)
    harris = cv2.cornerHarris(gray_f32, block_size, ksize, k)

    response_norm = cv2.normalize(harris, None, 0, 255, cv2.NORM_MINMAX)
    response_norm = np.uint8(response_norm)

    thresh = thresh_rel * harris.max()

    overlay_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    ys, xs = np.where(harris > thresh)
    for (x, y) in zip(xs, ys):
        cv2.circle(overlay_bgr, (int(x), int(y)), 4, (0, 0, 255), 2)

    return response_norm, overlay_bgr


@app.route("/module3/q2", methods=["GET", "POST"])
def module3_q2_page():
    results = []

    if request.method == "POST":
        try:
            files = request.files.getlist("files")
            if not files or len(files) == 0:
                return render_template("M3Q2.html", error="No files selected")

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

                    _, edge_overlay = detect_edge_keypoints(img_gray)
                    _, corner_overlay = detect_corner_keypoints(img_gray)

                    orig_path = os.path.join(app.config["UPLOAD_FOLDER"], "original_" + filename)
                    edge_overlay_path = os.path.join(app.config["UPLOAD_FOLDER"], "edges_overlay_" + filename)
                    corner_overlay_path = os.path.join(app.config["UPLOAD_FOLDER"], "corners_overlay_" + filename)

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
                return render_template("M3Q2.html", error="No valid images processed")

            return render_template("M3Q2.html", results=results)

        except Exception as e:
            return render_template("M3Q2.html", error=f"Processing error: {str(e)}")

    return render_template("M3Q2.html")


# =========================================================
# MODULE 3 – Q3 (Object Boundary via Contours)
#   Page:  /module3/q3   -> M3Q3.html
# =========================================================

def find_object_boundary(img_color):
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(img_gray, (5, 5), 1.0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boundary_overlay = img_color.copy()
    object_only = np.zeros_like(img_color)

    if contours:
        main_contour = max(contours, key=cv2.contourArea)

        cv2.drawContours(boundary_overlay, [main_contour], -1, (0, 255, 0), 3)

        mask = np.zeros(img_gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [main_contour], -1, 255, thickness=-1)
        object_only = cv2.bitwise_and(img_color, img_color, mask=mask)

    edges_vis = edges_closed
    return edges_vis, boundary_overlay, object_only


@app.route("/module3/q3", methods=["GET", "POST"])
def module3_q3_page():
    results = []

    if request.method == "POST":
        try:
            files = request.files.getlist("files")
            if not files or len(files) == 0:
                return render_template("M3Q3.html", error="No files selected")

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

                    edges_vis, boundary_overlay, object_only = find_object_boundary(img_color)

                    orig_path = os.path.join(app.config["UPLOAD_FOLDER"], "original_" + filename)
                    edges_path = os.path.join(app.config["UPLOAD_FOLDER"], "edges_" + filename)
                    boundary_path = os.path.join(app.config["UPLOAD_FOLDER"], "boundary_overlay_" + filename)
                    object_path = os.path.join(app.config["UPLOAD_FOLDER"], "object_only_" + filename)

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
                return render_template("M3Q3.html", error="No valid images processed")

            return render_template("M3Q3.html", results=results)

        except Exception as e:
            return render_template("M3Q3.html", error=f"Processing error: {str(e)}")

    return render_template("M3Q3.html")


# =========================================================
# MODULE 3 – Q4 (ArUco-based Object Segmentation)
#   Page:  /module3/q4   -> M3Q4.html
# =========================================================

def get_aruco_dict_and_params():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "Your OpenCV build has no 'aruco' module. "
            "Install with: pip install opencv-contrib-python"
        )

    aruco = cv2.aruco

    if hasattr(aruco, "getPredefinedDictionary"):
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    else:
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

    if hasattr(aruco, "DetectorParameters_create"):
        parameters = aruco.DetectorParameters_create()
    else:
        parameters = aruco.DetectorParameters()

    def set_if_has(obj, name, value):
        if hasattr(obj, name):
            setattr(obj, name, value)

    set_if_has(parameters, "adaptiveThreshWinSizeMin", 3)
    set_if_has(parameters, "adaptiveThreshWinSizeMax", 23)
    set_if_has(parameters, "adaptiveThreshWinSizeStep", 10)
    set_if_has(parameters, "adaptiveThreshConstant", 7)

    set_if_has(parameters, "minMarkerPerimeterRate", 0.02)
    set_if_has(parameters, "maxMarkerPerimeterRate", 4.0)
    set_if_has(parameters, "polygonalApproxAccuracyRate", 0.03)
    set_if_has(parameters, "minCornerDistanceRate", 0.05)
    set_if_has(parameters, "minOtsuStdDev", 5.0)
    set_if_has(parameters, "perspectiveRemovePixelPerCell", 8)

    if hasattr(aruco, "CORNER_REFINE_SUBPIX"):
        set_if_has(parameters, "cornerRefinementMethod", aruco.CORNER_REFINE_SUBPIX)

    use_detector_class = hasattr(aruco, "ArucoDetector")

    return aruco, aruco_dict, parameters, use_detector_class


def segment_object_with_aruco(img_color):
    img_bgr = img_color.copy()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    aruco, aruco_dict, parameters, use_detector_class = get_aruco_dict_and_params()

    if use_detector_class:
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    aruco_overlay = img_bgr.copy()
    boundary_overlay = img_bgr.copy()

    if ids is None or len(ids) == 0:
        return aruco_overlay, boundary_overlay, img_bgr

    aruco.drawDetectedMarkers(aruco_overlay, corners, ids)

    centers = []
    for c in corners:
        pts = c[0]
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        centers.append([cx, cy])

    centers = np.array(centers, dtype=np.float32)
    if centers.shape[0] < 3:
        return aruco_overlay, boundary_overlay, img_bgr

    hull = cv2.convexHull(centers).astype(np.int32)
    hull_for_draw = hull.reshape(-1, 1, 2)

    cv2.polylines(boundary_overlay, [hull_for_draw],
                  isClosed=True, color=(0, 255, 0), thickness=3)

    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    mask = cv2.medianBlur(mask, 7)

    object_only = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

    return aruco_overlay, boundary_overlay, object_only


@app.route("/module3/q4", methods=["GET", "POST"])
def module3_q4_page():
    results = []

    if request.method == "POST":
        files = request.files.getlist("files")
        if not files or len(files) == 0:
            return render_template("M3Q4.html", error="No files selected.")

        files = files[:10]

        # optional: clear previous outputs so only current set is visible
        for f in os.listdir(UPLOAD_FOLDER):
            fp = os.path.join(UPLOAD_FOLDER, f)
            try:
                os.remove(fp)
            except OSError:
                pass

        for file in files:
            if not file or file.filename.strip() == "":
                continue

            filename = secure_filename(file.filename)
            raw_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(raw_path)

            img_color = cv2.imread(raw_path)
            if img_color is None:
                continue

            try:
                aruco_overlay, boundary_overlay, object_only = \
                    segment_object_with_aruco(img_color)
            except Exception as e:
                return render_template("M3Q4.html",
                                       error=f"Processing error: {e}")

            orig_name = "original_" + filename
            aruco_name = "aruco_" + filename
            boundary_name = "boundary_" + filename
            object_name = "object_" + filename

            orig_path = os.path.join(UPLOAD_FOLDER, orig_name)
            aruco_path = os.path.join(UPLOAD_FOLDER, aruco_name)
            boundary_path = os.path.join(UPLOAD_FOLDER, boundary_name)
            object_path = os.path.join(UPLOAD_FOLDER, object_name)

            cv2.imwrite(orig_path, img_color)
            cv2.imwrite(aruco_path, aruco_overlay)
            cv2.imwrite(boundary_path, boundary_overlay)
            cv2.imwrite(object_path, object_only)

            results.append({
                "filename": filename,
                "original": url_for("static", filename=f"uploads/{orig_name}"),
                "aruco": url_for("static", filename=f"uploads/{aruco_name}"),
                "boundary": url_for("static", filename=f"uploads/{boundary_name}"),
                "object_only": url_for("static", filename=f"uploads/{object_name}")
            })

        if not results:
            return render_template("M3Q4.html", error="No valid images processed.")

        return render_template("M3Q4.html", results=results)

    return render_template("M3Q4.html")

# ------------------------------------------------------------
# MODULE 4 – Image stitching & panorama comparison
# URL: /module4
# HTML: templates/M4Q1.html
# ------------------------------------------------------------

@app.route("/module4", methods=["GET", "POST"])
def module4_stitching():
    """
    Module 4:
    - Upload at least 4 (landscape) or 8 (portrait) images of a scene.
    - Optionally upload a phone-generated panorama for comparison.
    - Use OpenCV's Stitcher to create a panorama from the uploaded images.
    """
    stitched_url = None
    phone_pano_url = None
    input_images = []
    error = None

    if request.method == "POST":
        try:
            # 1) Get the list of images for stitching
            files = request.files.getlist("images")
            if not files or len(files) == 0:
                error = "Please select at least 4 images for stitching."
                return render_template(
                    "M4Q1.html",
                    error=error,
                    stitched_url=stitched_url,
                    phone_pano_url=phone_pano_url,
                    input_images=input_images,
                )

            # enforce minimum count (4 images)
            if len(files) < 4:
                error = f"You selected only {len(files)} images. Please upload at least 4."
                return render_template(
                    "M4Q1.html",
                    error=error,
                    stitched_url=stitched_url,
                    phone_pano_url=phone_pano_url,
                    input_images=input_images,
                )

            # 2) Save uploaded images to static/uploads and keep paths
            img_paths = []
            for file in files:
                if not file or file.filename.strip() == "":
                    continue

                filename = secure_filename(file.filename)
                save_name = "m4_" + filename  # prefix so we know it's module-4
                path = os.path.join(app.config["UPLOAD_FOLDER"], save_name)
                file.save(path)
                img_paths.append(path)
                input_images.append("/" + path)  # for displaying thumbnails

            if len(img_paths) < 2:
                error = "Unable to read the uploaded images."
                return render_template(
                    "M4Q1.html",
                    error=error,
                    stitched_url=stitched_url,
                    phone_pano_url=phone_pano_url,
                    input_images=input_images,
                )

            # 3) Load images as OpenCV arrays
            imgs = []
            for p in img_paths:
                img = cv2.imread(p)
                if img is None:
                    continue
                # optional: resize large images to speed up stitching
                max_dim = 1200
                h, w = img.shape[:2]
                scale = min(max_dim / max(h, w), 1.0)
                if scale < 1.0:
                    img = cv2.resize(
                        img,
                        (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                imgs.append(img)

            if len(imgs) < 2:
                error = "Not enough valid images after loading. Check your files."
                return render_template(
                    "M4Q1.html",
                    error=error,
                    stitched_url=stitched_url,
                    phone_pano_url=phone_pano_url,
                    input_images=input_images,
                )

            # 4) Create the stitcher
            # Newer OpenCV
            if hasattr(cv2, "Stitcher_create"):
                stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
            else:  # older OpenCV fallback
                stitcher = cv2.createStitcher(False)

            (status, pano) = stitcher.stitch(imgs)

            if status != cv2.Stitcher_OK:
                error = f"Stitching failed (status code: {status}). " \
                        "Try more overlap, good texture, or fewer images."
                return render_template(
                    "M4Q1.html",
                    error=error,
                    stitched_url=stitched_url,
                    phone_pano_url=phone_pano_url,
                    input_images=input_images,
                )

            # 5) Save stitched panorama
            pano_name = "module4_stitched_panorama.jpg"
            pano_path = os.path.join(app.config["UPLOAD_FOLDER"], pano_name)
            cv2.imwrite(pano_path, pano)
            stitched_url = "/" + pano_path

            # 6) Optional: save phone panorama for comparison (if provided)
            phone_file = request.files.get("phone_pano")
            if phone_file and phone_file.filename.strip() != "":
                phone_name = secure_filename("module4_phone_" + phone_file.filename)
                phone_path = os.path.join(app.config["UPLOAD_FOLDER"], phone_name)
                phone_file.save(phone_path)
                phone_pano_url = "/" + phone_path

        except Exception as e:
            error = f"Processing error: {str(e)}"

    # GET or POST with error/success
    return render_template(
        "M4Q1.html",
        error=error,
        stitched_url=stitched_url,
        phone_pano_url=phone_pano_url,
        input_images=input_images,
    )
# ------------------------------------------------------------
# MODULE 4 – Q2: very simple SIFT-like feature extractor
# ------------------------------------------------------------

def m4q2_custom_sift(gray, max_keypoints=800, dog_thresh=0.001):
    """
    Very simplified SIFT-like feature detector + descriptor.

    - Primary detector: DoG extrema in a tiny 3-scale pyramid
    - If DoG finds too few points, fall back to Harris corners
    - 128-D SIFT-style descriptor (4x4 cells, 8-bin orientation histograms)
    """
    g = gray.astype(np.float32) / 255.0
    h, w = g.shape[:2]

    # ---------- 1. Gaussian pyramid + DoG ----------
    sigmas = [1.0, 2.0, 4.0]
    gaussians = [cv2.GaussianBlur(g, (0, 0), sigmaX=s) for s in sigmas]
    dogs = [gaussians[i + 1] - gaussians[i] for i in range(len(sigmas) - 1)]

    cand_pts = []
    cand_resp = []

    # look for local extrema in each DoG level
    for s_idx, dog in enumerate(dogs):
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                val = dog[y, x]
                if abs(val) < dog_thresh:   # very low threshold → many candidates
                    continue

                patch = dog[y - 1:y + 2, x - 1:x + 2]
                if (val == patch.max()) or (val == patch.min()):
                    cand_pts.append((x, y, s_idx))
                    cand_resp.append(abs(val))

    # ---------- 2. If DoG is empty / too small, fall back to Harris ----------
    if len(cand_pts) < 50:     # heuristic: “too few” → use Harris corners
        corners = cv2.goodFeaturesToTrack(
            (g * 255).astype(np.uint8),
            maxCorners=max_keypoints,
            qualityLevel=0.01,
            minDistance=5,
            blockSize=3,
        )
        cand_pts = []
        cand_resp = []
        if corners is not None:
            for c in corners:
                x, y = c.ravel()
                x, y = int(x), int(y)
                if 1 <= x < w - 1 and 1 <= y < h - 1:
                    cand_pts.append((x, y, 0))  # scale index 0
                    patch = g[y - 1:y + 2, x - 1:x + 2]
                    cand_resp.append(float(patch.var()))

    if not cand_pts:
        return [], None

    # ---------- 3. Keep strongest points, avoid duplicates ----------
    idxs = np.argsort(cand_resp)[::-1][:max_keypoints]
    keypoints = []
    used = set()

    for idx in idxs:
        x, y, s_idx = cand_pts[idx]
        if (x, y) in used:
            continue
        used.add((x, y))

        size = 4.0 * (s_idx + 1)
        response = float(cand_resp[idx])
        kp = cv2.KeyPoint(float(x), float(y), size, -1, response, s_idx, -1)
        keypoints.append(kp)

    # ---------- 4. Build 128-D SIFT-like descriptors ----------
    descs = []
    for kp in keypoints:
        x = int(round(kp.pt[0]))
        y = int(round(kp.pt[1]))
        radius = 8  # 16×16 patch

        if (x - radius < 0 or x + radius >= w or
                y - radius < 0 or y + radius >= h):
            continue

        patch = g[y - radius:y + radius, x - radius:x + radius]

        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=False)

        descriptor = []
        cell_size = 4          # 4×4 cells in a 16×16 patch
        num_bins = 8
        bin_width = 2 * math.pi / num_bins

        for cy in range(4):
            for cx in range(4):
                cell_mag = mag[cy * cell_size:(cy + 1) * cell_size,
                               cx * cell_size:(cx + 1) * cell_size]
                cell_ang = ang[cy * cell_size:(cy + 1) * cell_size,
                               cx * cell_size:(cx + 1) * cell_size]

                hist = np.zeros(num_bins, dtype=np.float32)
                for yy in range(cell_mag.shape[0]):
                    for xx in range(cell_mag.shape[1]):
                        m = cell_mag[yy, xx]
                        a = cell_ang[yy, xx]
                        bin_idx = int(a / bin_width) % num_bins
                        hist[bin_idx] += m

                descriptor.extend(hist.tolist())

        desc = np.array(descriptor, dtype=np.float32)
        # SIFT-style normalize → clip → renormalize
        n = np.linalg.norm(desc) + 1e-7
        desc /= n
        desc = np.clip(desc, 0, 0.2)
        n = np.linalg.norm(desc) + 1e-7
        desc /= n

        descs.append(desc)

    if not descs:
        return [], None

    descs = np.stack(descs).astype(np.float32)
    keypoints = keypoints[:descs.shape[0]]
    return keypoints, descs


# ------------------------------------------------------------
# MODULE 4 – Q2: SIFT-from-scratch + RANSAC + OpenCV SIFT comparison
# URL: /module4/q2
# Template: M4Q2.html
# ------------------------------------------------------------

# ------------------------------------------------------------
# MODULE 4 – Q2: SIFT-from-scratch + RANSAC + OpenCV SIFT comparison
# URL: /module4/q2
# Template: M4Q2.html
# ------------------------------------------------------------

@app.route("/module4/q2", methods=["GET", "POST"])
def module4_q2_sift():
    error = None

    img1_url = None
    img2_url = None

    custom_matches_url = None
    custom_warp_url = None

    sift_matches_url = None
    sift_warp_url = None

    # stats
    custom_kp1 = custom_kp2 = custom_inliers = 0
    sift_kp1 = sift_kp2 = sift_inliers = 0
    sift_available = False

    if request.method == "POST":
        try:
            file1 = request.files.get("image1")
            file2 = request.files.get("image2")

            if (not file1 or file1.filename.strip() == "" or
                    not file2 or file2.filename.strip() == ""):
                error = "Please upload both Image 1 and Image 2."
            else:
                # ----- Save inputs -----
                fname1 = secure_filename("m4q2_img1_" + file1.filename)
                fname2 = secure_filename("m4q2_img2_" + file2.filename)

                path1 = os.path.join(app.config["UPLOAD_FOLDER"], fname1)
                path2 = os.path.join(app.config["UPLOAD_FOLDER"], fname2)
                file1.save(path1)
                file2.save(path2)

                img1_color = cv2.imread(path1)
                img2_color = cv2.imread(path2)

                if img1_color is None or img2_color is None:
                    error = "Failed to read one or both images. Check the formats."
                else:
                    img1_url = "/" + path1
                    img2_url = "/" + path2

                    # small helper: resize if very large
                    def resize_if_large(img, max_dim=800):
                        h, w = img.shape[:2]
                        scale = min(max_dim / max(h, w), 1.0)
                        if scale < 1.0:
                            img = cv2.resize(
                                img,
                                (int(w * scale), int(h * scale)),
                                interpolation=cv2.INTER_AREA,
                            )
                        return img

                    img1 = resize_if_large(img1_color)
                    img2 = resize_if_large(img2_color)

                    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

                    # =====================================================
                    # A) CUSTOM SIFT + RANSAC
                    # =====================================================
                    kp1_custom, des1_custom = m4q2_custom_sift(gray1)
                    kp2_custom, des2_custom = m4q2_custom_sift(gray2)
                    custom_kp1 = len(kp1_custom)
                    custom_kp2 = len(kp2_custom)
                    custom_inliers = 0

                    if des1_custom is not None and des2_custom is not None:
                        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                        matches = bf.knnMatch(des1_custom, des2_custom, k=2)

                        good_custom = []
                        for m, n in matches:
                            if m.distance < 0.75 * n.distance:
                                good_custom.append(m)

                        if len(good_custom) >= 4:
                            src_pts = np.float32(
                                [kp1_custom[m.queryIdx].pt for m in good_custom]
                            ).reshape(-1, 1, 2)
                            dst_pts = np.float32(
                                [kp2_custom[m.trainIdx].pt for m in good_custom]
                            ).reshape(-1, 1, 2)

                            H_custom, mask_custom = cv2.findHomography(
                                src_pts, dst_pts, cv2.RANSAC, 5.0
                            )

                            if H_custom is not None and mask_custom is not None:
                                custom_inliers = int(mask_custom.ravel().sum())

                                matches_mask = mask_custom.ravel().tolist()
                                draw_params = dict(
                                    matchColor=(0, 255, 0),
                                    singlePointColor=(255, 0, 0),
                                    matchesMask=matches_mask,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                                )

                                custom_match_img = cv2.drawMatches(
                                    img1, kp1_custom,
                                    img2, kp2_custom,
                                    good_custom, None, **draw_params
                                )

                                custom_match_name = "m4q2_custom_matches.jpg"
                                custom_match_path = os.path.join(
                                    app.config["UPLOAD_FOLDER"], custom_match_name
                                )
                                cv2.imwrite(custom_match_path, custom_match_img)
                                custom_matches_url = "/" + custom_match_path

                                # warp image 2 into image 1 frame
                                h1, w1 = img1.shape[:2]
                                custom_warp = cv2.warpPerspective(
                                    img2, H_custom,
                                    (w1 + img2.shape[1], max(h1, img2.shape[0]))
                                )
                                custom_warp[0:h1, 0:w1] = img1

                                custom_warp_name = "m4q2_custom_warp.jpg"
                                custom_warp_path = os.path.join(
                                    app.config["UPLOAD_FOLDER"], custom_warp_name
                                )
                                cv2.imwrite(custom_warp_path, custom_warp)
                                custom_warp_url = "/" + custom_warp_path

                    # =====================================================
                    # B) OpenCV SIFT + RANSAC (for comparison)
                    # =====================================================
                    if hasattr(cv2, "SIFT_create"):
                        sift_available = True
                        sift = cv2.SIFT_create()
                        kp1_sift, des1_sift = sift.detectAndCompute(gray1, None)
                        kp2_sift, des2_sift = sift.detectAndCompute(gray2, None)
                        sift_kp1 = len(kp1_sift)
                        sift_kp2 = len(kp2_sift)
                        sift_inliers = 0

                        if des1_sift is not None and des2_sift is not None:
                            bf2 = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                            matches2 = bf2.knnMatch(des1_sift, des2_sift, k=2)

                            good_sift = []
                            for m, n in matches2:
                                if m.distance < 0.75 * n.distance:
                                    good_sift.append(m)

                            if len(good_sift) >= 4:
                                src_pts2 = np.float32(
                                    [kp1_sift[m.queryIdx].pt for m in good_sift]
                                ).reshape(-1, 1, 2)
                                dst_pts2 = np.float32(
                                    [kp2_sift[m.trainIdx].pt for m in good_sift]
                                ).reshape(-1, 1, 2)

                                H_sift, mask_sift = cv2.findHomography(
                                    src_pts2, dst_pts2, cv2.RANSAC, 5.0
                                )

                                if H_sift is not None and mask_sift is not None:
                                    sift_inliers = int(mask_sift.ravel().sum())

                                    matches_mask2 = mask_sift.ravel().tolist()
                                    draw_params2 = dict(
                                        matchColor=(0, 255, 0),
                                        singlePointColor=(255, 0, 0),
                                        matchesMask=matches_mask2,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                                    )

                                    sift_match_img = cv2.drawMatches(
                                        img1, kp1_sift,
                                        img2, kp2_sift,
                                        good_sift, None, **draw_params2
                                    )

                                    sift_match_name = "m4q2_sift_matches.jpg"
                                    sift_match_path = os.path.join(
                                        app.config["UPLOAD_FOLDER"], sift_match_name
                                    )
                                    cv2.imwrite(sift_match_path, sift_match_img)
                                    sift_matches_url = "/" + sift_match_path

                                    # warp image 2 using SIFT homography
                                    h1, w1 = img1.shape[:2]
                                    sift_warp = cv2.warpPerspective(
                                        img2, H_sift,
                                        (w1 + img2.shape[1], max(h1, img2.shape[0]))
                                    )
                                    sift_warp[0:h1, 0:w1] = img1

                                    sift_warp_name = "m4q2_sift_warp.jpg"
                                    sift_warp_path = os.path.join(
                                        app.config["UPLOAD_FOLDER"], sift_warp_name
                                    )
                                    cv2.imwrite(sift_warp_path, sift_warp)
                                    sift_warp_url = "/" + sift_warp_path
                    else:
                        sift_available = False

        except Exception as e:
            error = f"Processing error: {str(e)}"

    return render_template(
        "M4Q2.html",
        error=error,
        img1_url=img1_url,
        img2_url=img2_url,
        custom_kp1=custom_kp1,
        custom_kp2=custom_kp2,
        custom_inliers=custom_inliers,
        sift_kp1=sift_kp1,
        sift_kp2=sift_kp2,
        sift_inliers=sift_inliers,
        custom_matches_url=custom_matches_url,
        custom_warp_url=custom_warp_url,
        sift_matches_url=sift_matches_url,
        sift_warp_url=sift_warp_url,
        sift_available=sift_available,
    )

# ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)
