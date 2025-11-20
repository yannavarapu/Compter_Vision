from flask import Flask, render_template, request
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ----------------- CONFIG -----------------
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Assignment-style params (similar to deblur_fourier_fixed.py)
SIGMA = 3.0              # Gaussian blur sigma
KERNEL_SIZE = 19         # should be odd; ~6*SIGMA+1 is a good rule
MODE = "wiener"          # "wiener" or "inverse"
K_WIENER = 0.001         # smaller K -> closer to inverse, usually sharper
# ------------------------------------------


# ---------- Helper functions (Fourier logic) ----------
def ensure_odd(k):
    k = int(k)
    return k if k % 2 == 1 else k + 1


def gaussian_psf(ksize, sigma):
    """Create 2D Gaussian PSF (kernel) normalized to sum = 1."""
    ax = np.arange(ksize) - (ksize - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2)).astype(np.float32)
    psf /= psf.sum()
    return psf


def psf_to_otf(psf, shapeHW):
    """
    Embed PSF into an array of size shapeHW and FFT to get Optical Transfer Function.

    We do an ifftshift on psf so its center goes to (0,0) before FFT.
    This matches circular convolution when we multiply in Fourier domain.
    """
    H, W = shapeHW
    pad = np.zeros((H, W), np.float32)
    psf_shifted = np.fft.ifftshift(psf)  # center -> (0,0)
    pad[:psf.shape[0], :psf.shape[1]] = psf_shifted
    return np.fft.fft2(pad)


def wiener_deconv(G, H, K):
    """Wiener deconvolution in frequency domain."""
    return (np.conj(H) / (np.abs(H)**2 + K)) * G


def inverse_deconv(G, H, eps=1e-6):
    """Simple inverse filtering (with small epsilon for stability)."""
    return G / (H + eps)


def to_float01(img):
    return img.astype(np.float32) / 255.0


def to_uint8(x):
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


# ---------- Gaussian Blur via Fourier (L -> L_b) ----------
def gaussian_blur(img_color, ksize=KERNEL_SIZE, sigma=SIGMA):
    """
    Apply Gaussian blur using a PSF and Fourier-domain multiplication.

    This ensures the blur model exactly matches what we later invert with Wiener
    deconvolution, so the recovered image can be very close to the original.
    """
    L = to_float01(img_color)
    H, W = L.shape[:2]

    ksize = ensure_odd(ksize)
    psf = gaussian_psf(ksize, sigma)
    OTF = psf_to_otf(psf, (H, W))

    L_b = np.zeros_like(L, dtype=np.float32)
    for c in range(3):
        F = np.fft.fft2(L[:, :, c])
        G = F * OTF                     # blur in frequency domain
        L_b[:, :, c] = np.fft.ifft2(G).real

    return to_uint8(L_b)


# ---------- Fourier-based Deblurring (L_b -> approx L) ----------
def fourier_deblur_color(blurred, sigma=SIGMA, K=K_WIENER, mode=MODE):
    """
    Recover image using Fourier-domain deconvolution
    (Wiener or inverse), using the SAME PSF/OTF as in gaussian_blur().
    """
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


# ---------- ROUTES ----------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            file = request.files.get('file', None)
            if not file or file.filename == "":
                return render_template("index1.html", error="No file selected")

            # Save uploaded image
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Load in color (BGR)
            img = cv2.imread(filepath, cv2.IMREAD_COLOR)
            if img is None:
                return render_template("index1.html", error="Invalid image format")

            # Step 1: Gaussian blur (L -> L_b)
            blurred = gaussian_blur(img)

            # Step 2: Fourier-based deblurring (L_b -> approx L)
            recovered = fourier_deblur_color(blurred)

            # Step 3: Save outputs
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original_' + filename)
            blurred_path = os.path.join(app.config['UPLOAD_FOLDER'], 'blurred_' + filename)
            recovered_path = os.path.join(app.config['UPLOAD_FOLDER'], 'recovered_' + filename)

            cv2.imwrite(original_path, img)
            cv2.imwrite(blurred_path, blurred)
            cv2.imwrite(recovered_path, recovered)

            return render_template(
                "index1.html",
                original='/' + original_path,
                blurred='/' + blurred_path,
                recovered='/' + recovered_path,
                sigma=SIGMA,
                kernel_size=ensure_odd(KERNEL_SIZE),
                mode=MODE,
                K=K_WIENER
            )

        except Exception as e:
            return render_template("index1.html", error=f"Processing error: {str(e)}")

    # GET
    return render_template("index1.html")


if __name__ == "__main__":
    # Run on port 5003 as you had
    app.run(host="0.0.0.0", port=5003, debug=True, threaded=True)
