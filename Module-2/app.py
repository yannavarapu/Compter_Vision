"""
Template Matching Web App
Uses the same logic as:

# Multi Detected
# Assignment 2 - Question 1

- TM_CCOEFF_NORMED
- SCALES = 0.5..1.4 (19 steps)
- ANGLES = [0, 180]
- One best detection per template (if score >= threshold)

For the assignment requirement:
Once the object is detected, the detected region is blurred
(with a Gaussian blur) instead of showing a heatmap.
"""

from flask import Flask, render_template, request, jsonify
import cv2 as cv
import numpy as np
import os
import base64
from datetime import datetime

app = Flask(__name__)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
SOURCE_FOLDER = 'source_images'
TEMPLATE_FOLDER = 'template_images'
RESULTS_FOLDER = 'results'

for folder in [SOURCE_FOLDER, TEMPLATE_FOLDER, RESULTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg']


def image_to_base64(image):
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
    return cv.warpAffine(gray, M, (nW, nH),
                         flags=cv.INTER_LINEAR,
                         borderMode=cv.BORDER_REPLICATE)


def blur_regions(img_bgr, matches, ksize=31, sigma=15):
    """
    Blur only the matched rectangles in img_bgr.

    - img_bgr: BGR image
    - matches: list of dicts with x, y, width, height
    """
    out = img_bgr.copy()
    if not matches:
        return out

    # ensure odd kernel size
    k = ksize | 1
    for m in matches:
        x, y, w, h = m['x'], m['y'], m['width'], m['height']
        # Clamp to image boundaries
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


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload_source', methods=['POST'])
def upload_source():
    """Upload source image"""
    try:
        print("\n[UPLOAD SOURCE] Request received")

        if 'source' not in request.files:
            return jsonify({'error': 'No source file provided'}), 400

        file = request.files['source']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Use JPG, JPEG, or PNG'}), 400

        filepath = os.path.join(SOURCE_FOLDER, 'source.jpg')
        file.save(filepath)
        print(f"[UPLOAD SOURCE] Saved to: {filepath}")

        img = cv.imread(filepath)
        if img is None:
            return jsonify({'error': 'Failed to read image file'}), 400

        h, w = img.shape[:2]
        print(f"[UPLOAD SOURCE] Image size: {w}x{h}")

        img_base64 = image_to_base64(img)

        return jsonify({
            'success': True,
            'filename': 'source.jpg',
            'width': int(w),
            'height': int(h),
            'image': f'data:image/jpeg;base64,{img_base64}'
        })

    except Exception as e:
        print(f"[UPLOAD SOURCE] Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/upload_template', methods=['POST'])
def upload_template():
    """Upload template image"""
    try:
        print("\n[UPLOAD TEMPLATE] Request received")

        if 'template' not in request.files:
            return jsonify({'error': 'No template file provided'}), 400

        file = request.files['template']
        template_id = request.form.get('template_id', '1')

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Use JPG, JPEG, or PNG'}), 400

        filename = f'template_{template_id}.jpg'
        filepath = os.path.join(TEMPLATE_FOLDER, filename)
        file.save(filepath)
        print(f"[UPLOAD TEMPLATE] Saved template {template_id} to: {filepath}")

        img = cv.imread(filepath)
        if img is None:
            return jsonify({'error': 'Failed to read template image'}), 400

        h, w = img.shape[:2]
        print(f"[UPLOAD TEMPLATE] Template {template_id} size: {w}x{h}")

        img_base64 = image_to_base64(img)

        return jsonify({
            'success': True,
            'template_id': template_id,
            'filename': filename,
            'width': int(w),
            'height': int(h),
            'image': f'data:image/jpeg;base64,{img_base64}'
        })

    except Exception as e:
        print(f"[UPLOAD TEMPLATE] Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/process_all', methods=['POST'])
def process_all():
    """
    Multi-template detection using the exact logic of:

      # Multi Detected  (Assignment 2 - Q1)

    After detecting the best match per template (if score >= threshold),
    the detected region is blurred with a Gaussian filter.
    """
    try:
        print("\n" + "=" * 70)
        print("PROCESSING ALL TEMPLATES (Multi Detected + Blur)")
        print("=" * 70)

        data = request.json or {}
        SCORE_THRESH = float(data.get('threshold', 0.60))

        METHOD = cv.TM_CCOEFF_NORMED
        SCALES = np.linspace(0.5, 1.4, 19)
        ANGLES = [0, 180]

        # -------- load source --------
        source_path = os.path.join(SOURCE_FOLDER, 'source.jpg')
        if not os.path.exists(source_path):
            print("ERROR: Source image not found!")
            return jsonify({'error': 'Please upload source image first'}), 400

        img_gray = cv.imread(source_path, cv.IMREAD_GRAYSCALE)
        if img_gray is None:
            print("ERROR: Failed to read source image")
            return jsonify({'error': 'Failed to read source image'}), 500

        img_gray = cv.GaussianBlur(img_gray, (3, 3), 0)
        H, W = img_gray.shape[:2]
        print(f"Source size: {W}x{H}")

        base_vis = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)

        # -------- gather templates --------
        template_files = []
        for i in range(1, 11):
            tpath = os.path.join(TEMPLATE_FOLDER, f'template_{i}.jpg')
            if os.path.exists(tpath):
                template_files.append((i, tpath))

        if len(template_files) == 0:
            return jsonify({'error': 'Please upload at least one template'}), 400

        print("Templates:", [p for _, p in template_files])

        colors = [
            (0, 255, 0), (0, 180, 255), (255, 160, 0), (255, 0, 120),
            (120, 255, 120), (160, 120, 255), (200, 200, 0), (0, 220, 180),
            (255, 200, 200), (200, 255, 200)
        ]

        results = []

        for idx, (template_id, tpath) in enumerate(template_files):
            print(f"\n--- Template {template_id} ---")
            tpl = cv.imread(tpath, cv.IMREAD_GRAYSCALE)
            if tpl is None:
                print(f"[skip] Can't read template: {tpath}")
                continue

            tpl = cv.GaussianBlur(tpl, (3, 3), 0)
            rows, cols = tpl.shape[:2]
            print(f"Template size: {cols}x{rows}")

            best_score = -1.0
            best = None          # (loc, (tw,th), scale, angle)
            best_res = None      # correlation map for completeness (still used for stats)

            # --- main search (same as your script) ---
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
                print(f"[warn] No valid scales for {name}.")
                matches = []
                num_matches = 0
                correlation = np.zeros((10, 10), dtype=np.float32)
            else:
                (x, y), (w, h), s, ang = best
                print(f"Detected '{name_noext}' at {x},{y} "
                      f"(score={best_score:.3f}, scale={s:.2f}, angle={ang}°)")

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

            # --- draw detection image (with rectangle & label) ---
            result_img = base_vis.copy()
            if len(matches) > 0:
                m = matches[0]
                color = colors[idx % len(colors)]
                x, y, w, h = m['x'], m['y'], m['width'], m['height']
                cv.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
                cv.putText(result_img,
                           f"{name_noext} {m['confidence']:.2f}",
                           (x, max(15, y - 6)),
                           cv.FONT_HERSHEY_SIMPLEX,
                           0.55,
                           color,
                           1,
                           cv.LINE_AA)

            # --- BLUR: build blurred image from detected region(s) ---
            # (This replaces the old heatmap visualization.)
            blurred_img = blur_regions(base_vis, matches, ksize=31, sigma=15)

            # we still keep correlation stats for display if you want them
            if correlation is not None and correlation.size > 0:
                max_corr = float(correlation.max())
                min_corr = float(correlation.min())
                mean_corr = float(correlation.mean())
            else:
                max_corr = min_corr = mean_corr = 0.0

            # template image base64
            tpl_bgr = cv.imread(tpath)
            tpl_b64 = image_to_base64(tpl_bgr) if tpl_bgr is not None else None
            res_b64 = image_to_base64(result_img)
            blur_b64 = image_to_base64(blurred_img)

            results.append({
                'template_id': template_id,
                'template_name': f'Template_{template_id}',
                'template_image': f'data:image/jpeg;base64,{tpl_b64}' if tpl_b64 else None,
                'num_matches': num_matches,
                'matches': matches,
                'result_image': f'data:image/jpeg;base64,{res_b64}',
                # For backward compatibility, we reuse the same key:
                # the frontend can rename "Correlation Heatmap" -> "Blurred Region".
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

        print("\n" + "=" * 70)
        print(f"PROCESSING COMPLETE: {len(results)} templates processed")
        print("=" * 70 + "\n")

        return jsonify(response)

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print("\n" + "=" * 70)
        print("ERROR:")
        print("=" * 70)
        print(error_msg)
        print("=" * 70 + "\n")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("TEMPLATE MATCHING WEB APPLICATION (Multi Detected + Blur)")
    print("=" * 70)
    print("Server URL: http://127.0.0.1:5002")
    print("Press Ctrl+C to stop")
    print("=" * 70 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5002, threaded=True)
