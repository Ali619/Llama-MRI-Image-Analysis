import numpy as np
import pydicom
from PIL import Image


def load_image(uploaded_file):
    if uploaded_file.type == "application/dicom":
        dicom_data = pydicom.dcmread(uploaded_file)
        image = dicom_data.pixel_array
    else:
        image = Image.open(uploaded_file)
    return image


def preprocess_image(image):
    # Add any preprocessing steps here (e.g., resizing, normalization)
    return image
