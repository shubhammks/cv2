from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_image():
    if "image" not in request.files:
        return "No file uploaded", 400

    file = request.files["image"]
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return "Failed to process image", 400

    # Resize image
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    inverted=cv2.bitwise_not(gray)
    inverted_blur=cv2.GaussianBlur(inverted,(199,199),0)
    skech=cv2.divide(gray,255-inverted_blur,scale=256)
    kernal_sharpen=np.array([[-1,-1,-1],
                            [-1,9,-1],
                            [-1,-1,-1]])
    sharp_skech=cv2.filter2D(skech,-1,kernal_sharpen)

    # Save and return processed image
    processed_path = "processed.png"
    cv2.imwrite(processed_path, sharp_skech)

    return send_file(processed_path, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True)
