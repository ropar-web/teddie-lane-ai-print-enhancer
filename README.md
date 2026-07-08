
# Teddie & Lane Real-ESRGAN Print App

This is the stronger local version of the print-ready enhancer.

## What this app does

- Uses **Real-ESRGAN** for actual AI image upscaling.
- Works better than a browser-only HTML file for clarity improvement.
- Accepts **PNG, JPG, WEBP, TIFF, BMP and PDF**.
- Exports exact-size **PDF, PNG, TIFF and JPEG**.
- Calculates the **real pixel dimensions** required for the selected physical print size.
- Supports **A4, A3, A5, US Letter, 8 × 10, 8.5 × 8.5, and custom size**.
- Embeds **300–600 DPI** metadata.
- Lets you choose:
  - RealESRGAN x2 Plus
  - RealESRGAN x4 Plus
  - or no AI upscaling

## Why this version is better

The earlier HTML/browser builds were limited by browser memory and weaker browser-safe AI models.

This version runs **locally in Python** and uses the proper Real-ESRGAN models, which are much better for:
- soft images
- low-resolution templates
- edge detail
- illustration clarity
- print enlargement

## Install

### 1. Create a virtual environment (recommended)

#### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Mac / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install requirements
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

## Notes

- The first time you run the AI models, the app will download the required `.pth` model weights automatically.
- If you do not have a GPU, it will still run on CPU, just slower.
- For templates and worksheets, **RealESRGAN x2 Plus** is usually the best first choice.
- For general prints or very small originals, try **RealESRGAN x4 Plus**.

## Recommended settings for your use

### Teddie & Lane templates
- Size: A4
- DPI: 300
- AI model: RealESRGAN x2 Plus
- AI upscale amount: 2.0
- Fit mode: Keep whole design + add margins
- Export: PDF + PNG

## Windows quick start

Double-click `run_app.bat`
