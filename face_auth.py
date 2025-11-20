from flask import Flask, render_template, request, jsonify
import cv2
import face_recognition
import numpy as np
import pickle
import os
import traceback
import base64
from io import BytesIO
from PIL import Image
import sys

# explicitly say templates folder
app = Flask(__name__, template_folder="templates")

# =============================================================================
# FACE AUTHENTICATOR
# =============================================================================

class FaceAuthenticator:
    def __init__(self, encodings_file="face_encodings.pkl"):
        self.encodings_file = encodings_file
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_encodings()

    def load_encodings(self):
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, "rb") as f:
                    data = pickle.load(f)
                self.known_face_encodings = data["encodings"]
                self.known_face_names = data["names"]
                print(f"✓ Loaded {len(self.known_face_encodings)} encoding(s)")
            except Exception as e:
                print("⚠ Error loading encodings:", e)
        else:
            print("⚠ No face encodings found yet.")

    def save_encodings(self):
        data = {
            "encodings": self.known_face_encodings,
            "names": self.known_face_names,
        }
        with open(self.encodings_file, "wb") as f:
            pickle.dump(data, f)
        print(f"✓ Encodings saved to {self.encodings_file}")

    def register_from_frame(self, frame, name="User"):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb)

        if not locs:
            return False, "No face detected"

        encs = face_recognition.face_encodings(rgb, locs)
        if not encs:
            return False, "Encoding failed"

        self.known_face_encodings.append(encs[0])
        self.known_face_names.append(name)
        self.save_encodings()
        return True, None

    def authenticate_from_frame(self, frame,
                                min_face_size=150,
                                confidence_threshold=0.5):
        if not self.known_face_encodings:
            return False, "No registered faces"

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, locs)

        if not locs:
            return False, "No face detected"

        for (top, right, bottom, left), enc in zip(locs, encs):
            w = right - left
            h = bottom - top

            if w < min_face_size or h < min_face_size:
                return False, "Move closer — face too small/cropped"

            matches = face_recognition.compare_faces(
                self.known_face_encodings, enc, tolerance=confidence_threshold
            )
            dist = face_recognition.face_distance(self.known_face_encodings, enc)

            if len(dist) == 0:
                continue

            idx = np.argmin(dist)
            confidence = 1 - dist[idx]

            if matches[idx] and confidence >= (1 - confidence_threshold):
                name = self.known_face_names[idx]
                print(f"✓ Authenticated as {name} (confidence: {confidence*100:.1f}%)")
                return True, name

        return False, "Face not recognized"

    def register_via_webcam(self, name="User"):
        print("\n" + "=" * 50)
        print("FACE REGISTRATION (CLI)")
        print("=" * 50)
        print(f"Registering face for: {name}")
        print("Instructions:")
        print("  - Look directly at the camera")
        print("  - Ensure good lighting")
        print("  - Press SPACE to capture")
        print("  - Press ESC to cancel")
        print("=" * 50 + "\n")

        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            print("✗ Error: Could not access webcam")
            return False

        face_captured = False

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            display_frame = frame.copy()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(
                    display_frame,
                    "Press SPACE to capture",
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            if not face_locations:
                cv2.putText(
                    display_frame,
                    "No face detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow(
                "Face Registration - SPACE: capture, ESC: cancel",
                display_frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                print("Registration cancelled")
                break
            elif key == 32 and face_locations:
                face_encodings = face_recognition.face_encodings(
                    rgb_frame, face_locations
                )
                if face_encodings:
                    self.known_face_encodings.append(face_encodings[0])
                    self.known_face_names.append(name)
                    self.save_encodings()
                    print(f"✓ Face registered successfully for {name}!")
                    face_captured = True
                    break

        video_capture.release()
        cv2.destroyAllWindows()
        return face_captured

# =============================================================================
# Helpers & Routes
# =============================================================================

def convert_base64_to_frame(b64data):
    img_data = b64data.split(",")[1]
    img = Image.open(BytesIO(base64.b64decode(img_data)))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

@app.route("/")
def home():
    print("Serving login.html")
    return render_template("login.html")

@app.route("/register", methods=["POST", "OPTIONS"])
def register_route():
    if request.method == "OPTIONS":
        # allow preflight if Safari/other sends it
        return ("", 204)

    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"status": "failed", "message": "No image data received"}), 400

        frame = convert_base64_to_frame(data["image"])
        auth = FaceAuthenticator()
        ok, msg = auth.register_from_frame(frame, name="User")

        if ok:
            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "failed", "message": msg})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/authenticate", methods=["POST", "OPTIONS"])
def authenticate_route():
    if request.method == "OPTIONS":
        # allow preflight if sent
        return ("", 204)

    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"status": "failed", "message": "No image data received"}), 400

        frame = convert_base64_to_frame(data["image"])
        auth = FaceAuthenticator()
        ok, result = auth.authenticate_from_frame(frame)

        if ok:
            return jsonify({"status": "success", "user": result})
        else:
            return jsonify({"status": "failed", "message": result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/dashboard")
def dashboard():
    print("Serving index.html (dashboard)")
    return render_template("index.html")

# =============================================================================
# Main entry
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--setup":
        auth = FaceAuthenticator()
        name = input("Enter your name: ").strip() or "User"
        success = auth.register_via_webcam(name)
        if success:
            print("\n✅ Setup complete. You can now run the web app and authenticate.")
        else:
            print("\n⚠ Setup did not complete.")
    else:
        app.run(port=5004, debug=True)
