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


# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)
