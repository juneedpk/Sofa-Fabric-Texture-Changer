# Sofa Fabric Texture Changer

An interactive web application that allows users to change the fabric texture of sofas in images. Built with Streamlit and OpenCV.

## Features

- Upload sofa and fabric texture images
- Adjust fabric pattern scale
- Control image resolution (DPI)
- Adjust brightness
- Download processed images
- High-quality texture mapping
- Perspective-aware fabric application

## Installation

```bash
pip install -r requirements.txt
```

## Local Development

Run the app locally:

```bash
streamlit run app.py
```

## Deployment

This app is deployed on Streamlit Cloud. Visit [https://sofa-fabric-changer.streamlit.app](https://sofa-fabric-changer.streamlit.app) to use the application.

## Usage

1. Upload a clear image of your sofa
2. Upload a fabric texture image
3. Adjust the parameters:
   - Fabric Pattern Scale: Controls the size of the fabric pattern
   - Image Resolution: Higher DPI for better quality
   - Brightness: Adjust the brightness of the result
4. Click 'Apply Fabric Texture'
5. Download the result

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- Pillow
- Streamlit
- Matplotlib
