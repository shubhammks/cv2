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
    h, w, _ = img.shape
    img = cv2.resize(img, (w, h))

    # Apply Canny Edge Detection
    edges = cv2.Canny(img, 20, 50)
    edges = cv2.bitwise_not(edges)
    new_img=cv2.bitwise_and(img,img,mask=edges)
    h1,w2,_=new_img.shape
    new_img=cv2.resize(new_img,(w2,h1))

    # Save and return processed image
    processed_path = "processed.png"
    cv2.imwrite(processed_path, new_img)

    return send_file(processed_path, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True)
