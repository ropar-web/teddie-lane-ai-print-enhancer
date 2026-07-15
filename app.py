
from __future__ import annotations

import io
import math
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz
import numpy as np
import streamlit as st
from PIL import Image, ImageCms, ImageFilter, ImageOps

Image.MAX_IMAGE_PIXELS = 180_000_000

APP_TITLE = "Teddie & Lane Cloud AI Print Enhancer"

RATIO_PRESETS = {
    "3:4": (3, 4),
    "4:5": (4, 5),
    "11:14": (11, 14),
    "2:3": (2, 3),
}
MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.1/RealESRGAN_x2plus.pth"
)
MODEL_FILENAME = "RealESRGAN_x2plus.pth"
MAX_OUTPUT_PIXELS = 30_000_000

PAGE_PRESETS = {
    "A1 — 594 × 841 mm": ("mm", 594.0, 841.0),
    "A2 — 420 × 594 mm": ("mm", 420.0, 594.0),
    "A3 — 297 × 420 mm": ("mm", 297.0, 420.0),
    "A4 — 210 × 297 mm": ("mm", 210.0, 297.0),
    "A5 — 148 × 210 mm": ("mm", 148.0, 210.0),
    "A6 — 105 × 148 mm": ("mm", 105.0, 148.0),
    "US Letter — 8.5 × 11 in": ("in", 8.5, 11.0),
    "Square — 8.5 × 8.5 in": ("in", 8.5, 8.5),
    "Art print — 8 × 10 in": ("in", 8.0, 10.0),
    "Custom size": ("mm", 210.0, 297.0),
}

BACKGROUNDS = {
    "White": "#FFFFFF",
    "Warm cream": "#FDF6EA",
    "Soft beige": "#E8D9BF",
}


@dataclass
class OutputSpec:
    width_in: float
    height_in: float
    dpi: int
    fit_mode: str
    background: str
    sharpen: float
    output_formats: list[str]
    use_ai: bool
    tile_size: int

    @property
    def target_px(self) -> tuple[int, int]:
        return (
            max(1, round(self.width_in * self.dpi)),
            max(1, round(self.height_in * self.dpi)),
        )


def safe_name(filename: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).stem).strip("._")
    return name or "print_ready"


def to_inches(unit: str, width: float, height: float) -> tuple[float, float]:
    if unit == "mm":
        return width / 25.4, height / 25.4
    if unit == "cm":
        return width / 2.54, height / 2.54
    return width, height


def normalize_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode == "RGBA":
        return image
    if image.mode == "P" and "transparency" in image.info:
        return image.convert("RGBA")
    return image.convert("RGB")


def open_image_bytes(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as opened:
        opened.load()
        return normalize_image(opened.copy())


def flatten_image(image: Image.Image, background: str = "#FFFFFF") -> Image.Image:
    if image.mode == "RGB":
        return image
    rgba = image.convert("RGBA")
    base = Image.new("RGB", rgba.size, background)
    base.paste(rgba, mask=rgba.getchannel("A"))
    return base


def fit_image(
    image: Image.Image,
    target_size: tuple[int, int],
    fit_mode: str,
    background: str,
) -> Image.Image:
    src_w, src_h = image.size
    dst_w, dst_h = target_size

    if fit_mode == "Fill page and crop":
        scale = max(dst_w / src_w, dst_h / src_h)
        size = (max(1, round(src_w * scale)), max(1, round(src_h * scale)))
        resized = image.resize(size, Image.Resampling.LANCZOS, reducing_gap=3.0)
        left = max(0, (resized.width - dst_w) // 2)
        top = max(0, (resized.height - dst_h) // 2)
        return resized.crop((left, top, left + dst_w, top + dst_h))

    scale = min(dst_w / src_w, dst_h / src_h)
    size = (max(1, round(src_w * scale)), max(1, round(src_h * scale)))
    resized = image.resize(size, Image.Resampling.LANCZOS, reducing_gap=3.0)

    mode = "RGBA" if image.mode == "RGBA" else "RGB"
    canvas = Image.new(mode, target_size, background)
    x = (dst_w - resized.width) // 2
    y = (dst_h - resized.height) // 2
    if resized.mode == "RGBA":
        canvas.paste(resized, (x, y), resized.getchannel("A"))
    else:
        canvas.paste(resized, (x, y))
    return canvas


# ---------------- Real-ESRGAN x2 network ----------------

def pixel_unshuffle(x, scale: int):
    import torch.nn.functional as F
    return F.pixel_unshuffle(x, scale)


def build_rrdbnet():
    import torch.nn as nn
    import torch.nn.functional as F

    class ResidualDenseBlock(nn.Module):
        def __init__(self, num_feat=64, num_grow_ch=32):
            super().__init__()
            self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(__import__("torch").cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(__import__("torch").cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(__import__("torch").cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(__import__("torch").cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, num_feat=64, num_grow_ch=32):
            super().__init__()
            self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

        def forward(self, x):
            out = self.rdb1(x)
            out = self.rdb2(out)
            out = self.rdb3(out)
            return out * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(
            self,
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        ):
            super().__init__()
            self.scale = scale
            internal_in_ch = num_in_ch * 4 if scale == 2 else num_in_ch
            self.conv_first = nn.Conv2d(internal_in_ch, num_feat, 3, 1, 1)
            self.body = nn.Sequential(
                *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch) for _ in range(num_block)]
            )
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            if self.scale == 2:
                feat = pixel_unshuffle(x, scale=2)
            else:
                feat = x
            feat = self.conv_first(feat)
            body_feat = self.conv_body(self.body(feat))
            feat = feat + body_feat
            feat = self.lrelu(
                self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
            )
            feat = self.lrelu(
                self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
            )
            out = self.conv_last(self.lrelu(self.conv_hr(feat)))
            return out

    return RRDBNet(scale=2)


def get_model_path() -> Path:
    cache_dir = Path.home() / ".cache" / "teddie_lane_ai"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / MODEL_FILENAME
    if not model_path.exists():
        urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


@st.cache_resource(show_spinner=False)
def load_ai_model():
    import torch

    torch.set_num_threads(2)
    model = build_rrdbnet()
    model_path = get_model_path()

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "params_ema" in checkpoint:
        state = checkpoint["params_ema"]
    elif isinstance(checkpoint, dict) and "params" in checkpoint:
        state = checkpoint["params"]
    else:
        state = checkpoint

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def pil_to_tensor(image: Image.Image):
    import torch

    rgb = flatten_image(image).convert("RGB")
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
    return tensor.contiguous()


def tensor_to_pil(tensor) -> Image.Image:
    data = (
        tensor.squeeze(0)
        .detach()
        .clamp_(0, 1)
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )
    return Image.fromarray(np.round(data * 255.0).astype(np.uint8), "RGB")


def tiled_ai_x2(image: Image.Image, tile_size: int = 128, tile_pad: int = 12) -> Image.Image:
    import torch
    import torch.nn.functional as F

    model = load_ai_model()
    input_tensor = pil_to_tensor(image)
    _, _, height, width = input_tensor.shape
    output = torch.empty((1, 3, height * 2, width * 2), dtype=torch.float32)

    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)
    progress = st.progress(0, text="Starting Real-ESRGAN AI…")
    total = tiles_x * tiles_y
    completed = 0

    with torch.inference_mode():
        for y in range(tiles_y):
            for x in range(tiles_x):
                start_x = x * tile_size
                end_x = min(start_x + tile_size, width)
                start_y = y * tile_size
                end_y = min(start_y + tile_size, height)

                pad_start_x = max(start_x - tile_pad, 0)
                pad_end_x = min(end_x + tile_pad, width)
                pad_start_y = max(start_y - tile_pad, 0)
                pad_end_y = min(end_y + tile_pad, height)

                tile = input_tensor[
                    :,
                    :,
                    pad_start_y:pad_end_y,
                    pad_start_x:pad_end_x,
                ]

                original_tile_h = tile.shape[-2]
                original_tile_w = tile.shape[-1]
                extra_bottom = original_tile_h % 2
                extra_right = original_tile_w % 2

                if extra_bottom or extra_right:
                    tile = F.pad(
                        tile,
                        (0, extra_right, 0, extra_bottom),
                        mode="replicate",
                    )

                enhanced = model(tile)
                enhanced = enhanced[
                    :,
                    :,
                    : original_tile_h * 2,
                    : original_tile_w * 2,
                ]

                out_start_x = start_x * 2
                out_end_x = end_x * 2
                out_start_y = start_y * 2
                out_end_y = end_y * 2

                tile_start_x = (start_x - pad_start_x) * 2
                tile_end_x = tile_start_x + (end_x - start_x) * 2
                tile_start_y = (start_y - pad_start_y) * 2
                tile_end_y = tile_start_y + (end_y - start_y) * 2

                output[
                    :,
                    :,
                    out_start_y:out_end_y,
                    out_start_x:out_end_x,
                ] = enhanced[
                    :,
                    :,
                    tile_start_y:tile_end_y,
                    tile_start_x:tile_end_x,
                ].cpu()

                del enhanced, tile
                completed += 1
                progress.progress(
                    completed / total,
                    text=f"AI tile {completed} of {total}",
                )

    progress.empty()
    return tensor_to_pil(output)


def create_ai_print_image(image: Image.Image, spec: OutputSpec) -> Image.Image:
    target_w, target_h = spec.target_px

    if not spec.use_ai:
        result = fit_image(image, (target_w, target_h), spec.fit_mode, spec.background)
    else:
        # Build a half-resolution print layout first; x2 AI then produces the
        # requested print resolution directly and predictably.
        work_w = math.ceil(target_w / 2)
        work_h = math.ceil(target_h / 2)
        if work_w % 2:
            work_w += 1
        if work_h % 2:
            work_h += 1

        working = fit_image(
            flatten_image(image, spec.background),
            (work_w, work_h),
            spec.fit_mode,
            spec.background,
        )
        result = tiled_ai_x2(working, tile_size=spec.tile_size)
        result = result.crop((0, 0, target_w, target_h))

    if spec.sharpen > 0:
        result = result.filter(
            ImageFilter.UnsharpMask(
                radius=0.7,
                percent=int(45 + 90 * spec.sharpen),
                threshold=3,
            )
        )
    return result


def encode_image(image: Image.Image, fmt: str, dpi: int) -> bytes:
    out = io.BytesIO()
    profile = None
    try:
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    except Exception:
        pass

    if fmt == "PNG":
        kwargs = {"format": "PNG", "dpi": (dpi, dpi), "compress_level": 6}
        if profile:
            kwargs["icc_profile"] = profile
        image.save(out, **kwargs)
    elif fmt == "JPEG":
        kwargs = {
            "format": "JPEG",
            "quality": 96,
            "subsampling": 0,
            "optimize": True,
            "dpi": (dpi, dpi),
        }
        if profile:
            kwargs["icc_profile"] = profile
        flatten_image(image).save(out, **kwargs)
    else:
        raise ValueError(f"Unsupported output format: {fmt}")
    return out.getvalue()




def image_to_svg(image: Image.Image, width_in: float, height_in: float, dpi: int) -> bytes:
    """Create an SVG containing the raster artwork at the exact physical size.
    This preserves true size, but it is not a fully editable vector trace.
    """
    import base64

    png_bytes = encode_image(image, "PNG", dpi)
    b64 = base64.b64encode(png_bytes).decode("ascii")
    svg = (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width_in:.4f}in" height="{height_in:.4f}in" '
        f'viewBox="0 0 {image.width} {image.height}" preserveAspectRatio="none">\n'
        f'  <image x="0" y="0" width="{image.width}" height="{image.height}" '
        f'href="data:image/png;base64,{b64}" />\n'
        f'</svg>'
    )
    return svg.encode("utf-8")


def image_to_pdf(image: Image.Image, width_in: float, height_in: float, dpi: int) -> bytes:
    png = encode_image(flatten_image(image), "PNG", dpi)
    doc = fitz.open()
    page = doc.new_page(width=width_in * 72, height=height_in * 72)
    page.insert_image(page.rect, stream=png, keep_proportion=False)
    data = doc.tobytes(garbage=4, deflate=True)
    doc.close()
    return data


def render_pdf_first_page(data: bytes, dpi: int = 120) -> Image.Image:
    doc = fitz.open(stream=data, filetype="pdf")
    if len(doc) == 0:
        doc.close()
        raise ValueError("The PDF has no pages.")
    pix = doc[0].get_pixmap(dpi=dpi, alpha=False)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return image



def process_file(filename: str, data: bytes, spec: OutputSpec):
    if filename.lower().endswith(".pdf"):
        source = render_pdf_first_page(data, dpi=150)
        pdf_warning = "Only the first PDF page was processed in this cloud-safe build."
    else:
        source = open_image_bytes(data)
        pdf_warning = ""

    original_size = source.size
    output = create_ai_print_image(source, spec)
    stem = safe_name(filename)
    files: list[tuple[str, bytes]] = []

    for fmt in spec.output_formats:
        if fmt == "PDF":
            files.append((f"{stem}_AI_PRINT_READY_{spec.dpi}dpi.pdf", image_to_pdf(output, spec.width_in, spec.height_in, spec.dpi)))
        elif fmt == "PNG":
            files.append((f"{stem}_AI_PRINT_READY_{spec.dpi}dpi.png", encode_image(output, "PNG", spec.dpi)))
        elif fmt == "JPG":
            files.append((f"{stem}_AI_PRINT_READY_{spec.dpi}dpi.jpg", encode_image(output, "JPEG", spec.dpi)))
        elif fmt == "SVG":
            files.append((f"{stem}_AI_PRINT_READY_{spec.dpi}dpi.svg", image_to_svg(output, spec.width_in, spec.height_in, spec.dpi)))

    report = {
        "Original pixels": f"{original_size[0]:,} × {original_size[1]:,}",
        "Final pixels": f"{output.width:,} × {output.height:,}",
        "Physical size": f"{spec.width_in:.2f} × {spec.height_in:.2f} in",
        "DPI": spec.dpi,
        "AI": "Real-ESRGAN x2 Plus" if spec.use_ai else "Off",
        "SVG note": "SVG export uses an embedded raster image at true physical size.",
        "PDF note": pdf_warning,
    }
    return files, report, output


def make_zip(files: Iterable[tuple[str, bytes]]) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files:
            archive.writestr(name, data)
    return out.getvalue()


def style():
    st.markdown(
        """
        <style>
        :root {
          color-scheme: light !important;
        }

        html, body, .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
          background: #FDF6EA !important;
          color: #3A2C20 !important;
        }

        [data-testid="stHeader"] {
          background: rgba(253, 246, 234, 0.96) !important;
        }

        h1, h2, h3, h4, h5, h6,
        p, span, label,
        .stMarkdown,
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] *,
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] *,
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"],
        .stRadio label,
        .stCheckbox label {
          color: #3A2C20 !important;
        }

        /* Select boxes, number inputs and text inputs */
        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div,
        [data-baseweb="base-input"],
        input,
        textarea {
          background: #FFFFFF !important;
          color: #3A2C20 !important;
          border-color: #D8C7AA !important;
        }

        [data-baseweb="select"] span,
        [data-baseweb="select"] div,
        [data-baseweb="input"] input {
          color: #3A2C20 !important;
        }

        [data-baseweb="popover"],
        [role="listbox"],
        [role="option"] {
          background: #FFFFFF !important;
          color: #3A2C20 !important;
        }

        [role="option"]:hover {
          background: #FFF4E6 !important;
        }

        /* Multiselect tags */
        [data-baseweb="tag"] {
          background: #CF6F4A !important;
          color: #FFFFFF !important;
        }

        [data-baseweb="tag"] span,
        [data-baseweb="tag"] svg {
          color: #FFFFFF !important;
          fill: #FFFFFF !important;
        }

        /* Radio and checkbox controls */
        [data-testid="stRadio"] label,
        [data-testid="stCheckbox"] label {
          color: #3A2C20 !important;
        }

        /* Sliders */
        [data-testid="stSlider"] [role="slider"] {
          background: #CF6F4A !important;
        }

        div[data-testid="stMetric"] {
          background: #FFF4E6 !important;
          border: 1px solid #E8D9BF !important;
          border-radius: 14px !important;
          padding: 10px !important;
        }

        div[data-testid="stMetric"] * {
          color: #3A2C20 !important;
        }

        div.stButton > button,
        div.stDownloadButton > button {
          background: #7F946F !important;
          color: #FFFFFF !important;
          border: 0 !important;
          border-radius: 12px !important;
          font-weight: 700 !important;
        }

        div.stButton > button:hover,
        div.stDownloadButton > button:hover {
          background: #627556 !important;
          color: #FFFFFF !important;
        }

        .note {
          background: #FFF4E6 !important;
          color: #3A2C20 !important;
          border-left: 5px solid #CF6F4A !important;
          border-radius: 12px !important;
          padding: 14px 16px !important;
          margin-bottom: 14px !important;
        }

        .note * {
          color: #3A2C20 !important;
        }

        /* File uploader */
        [data-testid="stFileUploaderDropzone"] {
          background: #FFF4E6 !important;
          border-color: #D8C7AA !important;
        }

        [data-testid="stFileUploaderDropzone"] * {
          color: #3A2C20 !important;
        }

        /* Dataframes and alerts */
        [data-testid="stAlert"] {
          color: #3A2C20 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="✨", layout="wide")
    style()

    st.title("✨ Teddie & Lane Cloud AI Print Enhancer")
    st.caption("Real-ESRGAN clarity enhancement with exact 150–350 DPI print export, JPG/SVG support, and true-size ratio presets.")

    st.markdown(
        """
        <div class="note">
        This cloud-compatible build removes BasicSR, torchvision, GFPGAN and the
        old Real-ESRGAN Python wrapper. It uses the official RealESRGAN x2 Plus
        model directly through CPU PyTorch. It now also includes <b>JPG</b>,
        <b>SVG</b> and true-size ratio presets: <b>3:4</b>, <b>4:5</b>, <b>11:14</b> and <b>2:3</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload an image or PDF",
        type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "pdf"],
    )

    left, right = st.columns([0.95, 1.25], gap="large")


    with left:
        size_mode = st.radio("Canvas type", ["Standard size", "Ratio preset"], horizontal=True)

        if size_mode == "Standard size":
            preset = st.selectbox("Finished size", list(PAGE_PRESETS.keys()))
            preset_unit, preset_w, preset_h = PAGE_PRESETS[preset]

            if preset == "Custom size":
                unit = st.selectbox("Units", ["mm", "cm", "in"])
                width = st.number_input("Width", min_value=1.0, value=210.0)
                height = st.number_input("Height", min_value=1.0, value=297.0)
            else:
                unit, width, height = preset_unit, preset_w, preset_h

            orientation = st.radio("Orientation", ["Portrait", "Landscape"], horizontal=True)
            if orientation == "Landscape" and height > width:
                width, height = height, width
            elif orientation == "Portrait" and width > height:
                width, height = height, width

        else:
            ratio_label = st.selectbox("Ratio preset", list(RATIO_PRESETS.keys()), index=0)
            unit = st.selectbox("Units", ["in", "cm", "mm"], index=0)
            orientation = st.radio("Orientation", ["Portrait", "Landscape"], horizontal=True, key="ratio_orientation")
            ratio_w, ratio_h = RATIO_PRESETS[ratio_label]
            if orientation == "Landscape":
                ratio_w, ratio_h = ratio_h, ratio_w
            width = st.number_input("Width", min_value=1.0, value=float(ratio_w), step=1.0)
            height = float(width) * float(ratio_h) / float(ratio_w)
            st.caption(f"Calculated height: {height:.2f} {unit}  •  Ratio {ratio_label}")

        width_in, height_in = to_inches(unit, float(width), float(height))
        dpi = st.selectbox("Output DPI", [150, 200, 300, 350], index=2)
        fit_mode = st.radio(
            "Artwork fit",
            ["Keep whole design + add margins", "Fill page and crop"],
        )
        background_name = st.selectbox("Background", list(BACKGROUNDS.keys()))
        use_ai = st.checkbox("Use Real-ESRGAN AI clarity enhancement", value=True)
        tile_size = st.selectbox(
            "AI tile size",
            [96, 128, 160],
            index=1,
            help="Use 96 if the app runs out of memory. 128 is the recommended starting point.",
        )
        sharpen = st.slider(
            "Final edge sharpening",
            min_value=0.0,
            max_value=0.6,
            value=0.15,
            step=0.05,
        )
        formats = st.multiselect(
            "Export formats",
            ["PDF", "PNG", "JPG", "SVG"],
            default=["PDF", "PNG", "JPG"],
        )
        st.caption("SVG downloads are embedded-raster SVG files at the exact physical size.")

        target_w = round(width_in * dpi)
        target_h = round(height_in * dpi)
        c1, c2, c3 = st.columns(3)
        c1.metric("Pixels", f"{target_w:,} × {target_h:,}")
        c2.metric("DPI", dpi)
        c3.metric("Size", f"{target_w * target_h / 1_000_000:.1f} MP")
    st.caption("Use 150 DPI for very large formats like A1, and 200 DPI for large A2 prints when 300 DPI is too large for the cloud-safe build.")

    with right:
        st.subheader("Original preview")
        if uploaded:
            try:
                data = uploaded.getvalue()
                if uploaded.name.lower().endswith(".pdf"):
                    preview = render_pdf_first_page(data, dpi=100)
                else:
                    preview = open_image_bytes(data)
                st.image(preview, use_container_width=True)
                st.caption(f"Original: {preview.width:,} × {preview.height:,} pixels")
            except Exception as exc:
                st.error(f"Preview error: {exc}")
        else:
            st.info("Upload an image or PDF to begin.")

    if not uploaded:
        return
    if not formats:
        st.warning("Choose at least one export format.")
        return

    spec = OutputSpec(
        width_in=width_in,
        height_in=height_in,
        dpi=int(dpi),
        fit_mode=fit_mode,
        background=BACKGROUNDS[background_name],
        sharpen=float(sharpen),
        output_formats=list(formats),
        use_ai=bool(use_ai),
        tile_size=int(tile_size),
    )

    if target_w * target_h > MAX_OUTPUT_PIXELS:
        st.error("This output is too large for the cloud-safe build. Use 300 DPI or a smaller page.")
        return

    if st.button("Create AI-enhanced print-ready files", type="primary", use_container_width=True):
        try:
            files, report, output = process_file(uploaded.name, uploaded.getvalue(), spec)
            st.session_state["output_zip"] = make_zip(files)
            st.session_state["output_preview"] = output
            st.session_state["output_report"] = report
        except Exception as exc:
            st.exception(exc)

    if "output_zip" in st.session_state:
        st.success("AI-enhanced print-ready files are complete.")
        st.image(
            st.session_state["output_preview"],
            caption="Processed preview",
            use_container_width=True,
        )
        st.dataframe(
            [
                {"Check": key, "Result": value}
                for key, value in st.session_state["output_report"].items()
                if value
            ],
            hide_index=True,
            use_container_width=True,
        )
        st.download_button(
            "Download print-ready ZIP",
            data=st.session_state["output_zip"],
            file_name=f"Teddie_Lane_AI_Print_Ready_{dpi}dpi.zip",
            mime="application/zip",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
