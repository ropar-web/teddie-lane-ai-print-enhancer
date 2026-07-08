
# Teddie & Lane Cloud AI Print Enhancer

This build is designed for Streamlit Community Cloud.

## Why this package is different

It does **not** install:

- basicsr
- torchvision
- realesrgan Python wrapper
- facexlib
- gfpgan
- OpenCV

Those older packages frequently conflict with current Python and torchvision releases.

Instead, the app includes the Real-ESRGAN x2 Plus network architecture directly and downloads the official model weights when AI enhancement is used for the first time.

## Deploy

1. Replace the old repository files with the files in this package.
2. Commit the changes.
3. In Streamlit Community Cloud, reboot the app.
4. If the existing app still uses a different Python version, delete it and redeploy.
5. Use Python 3.13 with this package.

Repository:
`YOUR-USERNAME/teddie-lane-ai-print-enhancer`

Branch:
`main`

Main file:
`app.py`

## First test

- A5
- 300 DPI
- Real-ESRGAN AI enabled
- Tile size 96
- PNG only

Then test A4 at 300 DPI.

## Limitations

- Community Cloud has limited CPU and RAM, so AI processing is slower than a local GPU.
- The cloud-safe PDF input processes the first page only.
- The first AI run downloads the official RealESRGAN x2 Plus model.


## New additions

- Export formats now include **JPG** and **SVG**.
- SVG export is an **embedded-raster SVG** at the exact physical size.
- Added ratio preset canvas options: **3:4, 4:5, 11:14, and 2:3**.
- In ratio mode, enter the width and the app calculates the matching height automatically.


## Added paper sizes

This update adds standard page size presets for **A1, A2, A3, and A6** in addition to the existing size options.
