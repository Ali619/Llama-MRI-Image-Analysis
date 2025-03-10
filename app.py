import base64
import json
import os
import sqlite3
import tempfile
from datetime import datetime
from io import BytesIO

import cv2
import numpy as np
import pydicom
import requests
from flask import Flask, jsonify, request
from flask_sslify import SSLify
from PIL import Image

app = Flask(__name__)
sslify = SSLify(app)


# Database setup
def init_db():
    conn = sqlite3.connect("mri_analysis.db")
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_history
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         filename TEXT,
         analysis_type TEXT,
         results TEXT,
         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
    """
    )
    conn.commit()
    conn.close()


# Modified Ollama API interaction with streaming
def analyze_image_with_ollama(image_base64, prompt):
    api_url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.2-vision",
        "prompt": prompt,
        "images": [image_base64],
        "stream": True,  # Enable streaming response
    }

    try:
        response = requests.post(api_url, json=payload, stream=True)
        response.raise_for_status()

        full_response = ""
        for line in response.iter_lines():
            if line:
                json_response = json.loads(line)
                if "response" in json_response:
                    full_response += json_response["response"]
                if json_response.get("done", False):
                    break

        return {"analysis": full_response}
    except requests.exceptions.RequestException as e:
        return {"error": f"API Request Error: {str(e)}"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON Decode Error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected Error: {str(e)}"}


def process_dicom(dicom_file):
    ds = pydicom.dcmread(dicom_file)
    if hasattr(ds, "PixelData"):
        image = ds.pixel_array
        if len(image.shape) == 2:
            return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            return [
                cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                for frame in image
            ]
    return None


def save_analysis_result(filename, analysis_type, results):
    conn = sqlite3.connect("mri_analysis.db")
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO analysis_history (filename, analysis_type, results)
        VALUES (?, ?, ?)
    """,
        (filename, analysis_type, json.dumps(results)),
    )
    conn.commit()
    conn.close()


@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files or "analysis_type" not in request.form:
        return jsonify({"error": "Missing required parameters"}), 400

    file = request.files["file"]
    analysis_type = request.form["analysis_type"]
    file_extension = file.filename.split(".")[-1].lower()

    if file_extension == "dcm":
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            image_data = process_dicom(tmp_file.name)
        os.unlink(tmp_file.name)
    else:
        image = Image.open(file)
        image_data = np.array(image)

    img_bytes = BytesIO()
    Image.fromarray(image_data).save(img_bytes, format="PNG")
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")

    # prompts = {
    #     "General_Description": "Summarize the purpose of analyzing this MRI image, provide a brief description of its key features, analyze the content, and conclude with a short summary of the findings.",
    #     "Anomaly_Detection": "Summarize the goal of detecting anomalies in this MRI image, briefly analyze potential irregularities, and provide a concise summary of detected anomalies.",
    #     "Segmentation": "Summarize the segmentation process for this MRI image, briefly analyze the different segments, and conclude with a short summary of the segmented areas.",
    #     "Condition_Identification": "Summarize the objective of identifying medical conditions in this MRI image, briefly analyze potential conditions, and provide a concise summary of the findings.",
    # }
    prompts = {
        "General_Description": "هدف از تحلیل این تصویر MRI را به‌طور خلاصه بیان کنید، ویژگی‌های کلیدی آن را به‌اختصار شرح دهید، محتوای تصویر را تجزیه‌وتحلیل کنید و در پایان، خلاصه‌ای از یافته‌ها ارائه دهید.",
        "Anomaly_Detection": "هدف از شناسایی ناهنجاری‌ها در این تصویر MRI را به‌طور خلاصه توضیح دهید، ناهنجاری‌های احتمالی را تجزیه‌وتحلیل کنید و در نهایت، خلاصه‌ای مختصر از ناهنجاری‌های شناسایی‌شده ارائه نمایید.",
        "Segmentation": "فرآیند بخش‌بندی در این تصویر MRI را به‌طور خلاصه توضیح دهید، بخش‌های مختلف تصویر را مورد تجزیه‌وتحلیل قرار دهید و در پایان، خلاصه‌ای از نواحی بخش‌بندی‌شده ارائه دهید.",
        "Condition_Identification": "هدف از شناسایی وضعیت‌های پزشکی در این تصویر MRI را به‌طور خلاصه بیان کنید، شرایط احتمالی موجود را تجزیه‌وتحلیل کنید و در نهایت، خلاصه‌ای از یافته‌های شناسایی‌شده ارائه دهید.",
    }

    if analysis_type not in prompts:
        return jsonify({"error": "Invalid analysis type"}), 400

    result = analyze_image_with_ollama(img_base64, prompts[analysis_type])
    save_analysis_result(file.filename, analysis_type, result)
    # return result["analysis"]
    return jsonify(result)


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, ssl_context=("cert.pem", "key.pem"))
