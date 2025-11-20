For Question-1,3:

(1) Run using: python app.py

(2) Detects 10 real-world objects in a scene using OpenCV’s correlation-based template matching (TM_CCOEFF_NORMED).

(3) Each template is captured from a different scene, ensuring no cropping from the test image.

(4) Performs multi-scale (0.5–1.4) and multi-angle (0° and 180°) matching to handle size and orientation changes.

(5) For each detected object, the system draws a bounding box and then blurs the detected region (required by the assignment).

(6) The Flask web interface allows users to upload 1 source image from source_images folder + 10 templates from template_images folder, and displays matches with correlation scores.


For Question-2:

(1)Run using: python app1.py

(2) Loads a user-provided image L and applies a Gaussian blur using a controlled PSF (σ, kernel size).

(3) Converts the blur model into the frequency domain (OTF) to accurately simulate convolution using FFT.

(4) Recovers the original image L from the blurred image L_b using Fourier-domain inverse / Wiener filtering.

(5) Produces three outputs: the original image, blurred image, and the Fourier-domain reconstructed image.

(6) The web interface allows to upload the upload a template image and then processing, and visualization of results for easy comparison.