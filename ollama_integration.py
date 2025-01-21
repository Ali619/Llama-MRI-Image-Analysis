import base64
import json
import os
import sqlite3
import tempfile
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import pydicom
import requests
import streamlit as st
from PIL import Image


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


# Modified Ollama API interaction
def analyze_image_with_ollama(image_base64, prompt):
    api_url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.2-vision",
        "prompt": prompt,
        "images": [image_base64],
        "stream": True,  # Set to False to get complete response
    }

    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Handle streaming response
        full_response = ""
        for line in response.iter_lines():
            if line:
                json_response = json.loads(line)
                if "response" in json_response:
                    full_response += json_response["response"]

                # Check if this is the last message
                if json_response.get("done", False):
                    break

        return {"analysis": full_response}

    except requests.exceptions.RequestException as e:
        return {"error": f"API Request Error: {str(e)}"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON Decode Error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected Error: {str(e)}"}


# Image processing functions
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


def main():
    st.title("MRI Image Analysis System")
    init_db()

    # Sidebar for analysis options
    st.sidebar.header("Analysis Options")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        [
            "General_Description",
            "Anomaly_Detection",
            "Segmentation",
            "Condition_Identification",
        ],
    )

    # File upload
    uploaded_file = st.file_uploader(
        "Upload MRI Image (DICOM, PNG, JPG, GIF)",
        type=["dcm", "png", "jpg", "jpeg", "gif"],
    )

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()

        # Process the uploaded file
        if file_extension == "dcm":
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                image_data = process_dicom(tmp_file.name)
            os.unlink(tmp_file.name)
        else:
            image = Image.open(uploaded_file)
            if file_extension == "gif":
                frames = []
                try:
                    while True:
                        frames.append(np.array(image.copy()))
                        image.seek(len(frames))
                except EOFError:
                    pass
                image_data = frames
            else:
                image_data = np.array(image)

        # Display the image(s)
        if isinstance(image_data, list):
            # Handle multi-frame images
            frame_idx = st.slider("Select frame", 0, len(image_data) - 1, 0)
            st.image(image_data[frame_idx], caption=f"Frame {frame_idx+1}")
            current_frame = image_data[frame_idx]
        else:
            st.image(image_data, caption="Uploaded MRI Image")
            current_frame = image_data

        # Prepare image for Ollama
        img_bytes = BytesIO()
        Image.fromarray(current_frame).save(img_bytes, format="PNG")
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")

        # Analysis prompts based on type
        prompts = {
            "General_Description": "Provide a detailed description of this MRI image, including the visible anatomical structures and any notable features.",
            "Anomaly_Detection": "Analyze this MRI image and identify any potential anomalies or unusual patterns. Focus on areas that appear different from normal tissue.",
            "Segmentation": "Identify and describe the different segments and regions visible in this MRI image, including tissue types and anatomical structures.",
            "Condition_Identification": "Based on this MRI image, identify any potential medical conditions or pathologies that might be present. List any concerning features.",
        }

        if st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                result = analyze_image_with_ollama(img_base64, prompts[analysis_type])

                # Display results
                st.subheader("Analysis Results")
                st.write(result)

                # Save to database
                save_analysis_result(uploaded_file.name, analysis_type, result)

        # Show history
        st.subheader("Analysis History")
        conn = sqlite3.connect("mri_analysis.db")
        history = pd.read_sql_query(
            "SELECT * FROM analysis_history ORDER BY timestamp DESC LIMIT 5", conn
        )
        conn.close()
        st.dataframe(history)


if __name__ == "__main__":
    main()
