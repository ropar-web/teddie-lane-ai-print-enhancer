
from __future__ import annotations

import io
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from PIL import Image, ImageFilter, ImageOps, ImageCms

# Lazy-heavy imports are kept inside functions:
# torch, cv2, basicsr, realesrgan

Image.MAX_IMAGE_PIXELS = 250_000_000

APP_TITLE = "Teddie & Lane Real-ESRGAN Print App"
MAX_OUTPUT_PIXELS = 150_000_000

PAGE_PRESETS = {
    "A4 — 210 × 297 mm": ("mm", 210.0, 297.0),
    "A3 — 297 × 420 mm": ("mm", 297.0, 420.0),
    "A5 — 148 × 210 mm": ("mm", 148.0, 210.0),
    "US Letter — 8.5 × 11 in": ("in", 8.5, 11.0),
    "Square book — 8.5 × 8.5 in": ("in", 8.5, 8.5),
    "Art print — 8 × 10 in": ("in", 8.0, 10.0),
    "Custom size": ("mm", 210.0, 297.0),
}

BACKGROUND_COLOURS = {
    "White": "#FFFFFF",
    "Warm cream": "#FDF6EA",
    "Soft beige": "#E8D9BF",
    "Transparent (PNG/TIFF only)": None,
}

AI_MODELS = {
    "RealESRGAN x2 Plus (recommended for templates/text)": "RealESRGAN_x2plus",
    "RealESRGAN x4 Plus (stronger enlargement)": "RealESRGAN_x4plus",
    "Off — no AI upscaling": "off",
}


@dataclass
class OutputSpec:
    width_in: float
    height_in: float
    dpi: int
    fit_mode: str
    background: str | None
    bleed_mm: float
    sharpen: float
    output_formats: list[str]
    ai_model: str
    ai_outscale: float
    keep_pdf_size: bool

    @property
    def final_width_in(self) -> float:
        return self.width_in + (2 * self.bleed_mm / 25.4)

    @property
    def final_height_in(self) -> float:
        return self.height_in + (2 * self.bleed_mm / 25.4)

    @property
    def target_px(self) -> tuple[int, int]:
        return (
            max(1, round(self.final_width_in * self.dpi)),
            max(1, round(self.final_height_in * self.dpi)),
        )


def safe_name(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return stem or "print_ready"


def mm_to_in(value: float) -> float:
    return value / 25.4


def to_inches(unit: str, width: float, height: float) -> tuple[float, float]:
    if unit == "mm":
        return mm_to_in(width), mm_to_in(height)
    if unit == "cm":
        return width / 2.54, height / 2.54
    return width, height


def srgb_profile_bytes() -> bytes | None:
    try:
        profile = ImageCms.createProfile("sRGB")
        return ImageCms.ImageCmsProfile(profile).tobytes()
    except Exception:
        return None


def normalize_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA"):
        return image.convert("RGBA")
    if image.mode == "P":
        return image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    if image.mode not in ("RGB", "RGBA", "L"):
        return image.convert("RGB")
    return image


def open_image_bytes(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as opened:
        opened.load()
        return normalize_image(opened.copy())


def pil_to_cv(image: Image.Image):
    import cv2

    if image.mode == "RGBA":
        arr = np.array(image)
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.array(image)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv_to_pil(array) -> Image.Image:
    import cv2

    if array.ndim == 3 and array.shape[2] == 4:
        return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGRA2RGBA))
    return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))


@st.cache_resource(show_spinner=False)
def load_realesrgan_upsampler(model_name: str):
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.download_util import load_file_from_url
    from realesrgan import RealESRGANer

    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)

    if model_name == "RealESRGAN_x4plus":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    elif model_name == "RealESRGAN_x2plus":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model_path = load_file_from_url(url=url, model_dir=str(weights_dir), progress=True)

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=200,
        tile_pad=16,
        pre_pad=0,
        half=torch.cuda.is_available(),
        gpu_id=0 if torch.cuda.is_available() else None,
    )
    return upsampler, netscale


def apply_realesrgan(image: Image.Image, model_name: str, outscale: float, progress_text: str = "") -> Image.Image:
    if model_name == "off":
        return image

    upsampler, netscale = load_realesrgan_upsampler(model_name)
    bgr = pil_to_cv(image)
    output, _ = upsampler.enhance(bgr, outscale=outscale)
    return cv_to_pil(output)


def make_background_canvas(size: tuple[int, int], background: str | None) -> Image.Image:
    if background is None:
        return Image.new("RGBA", size, (255, 255, 255, 0))
    return Image.new("RGB", size, background)


def alpha_composite_center(canvas: Image.Image, image: Image.Image, x: int, y: int) -> Image.Image:
    if canvas.mode == "RGBA":
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        canvas.alpha_composite(image, (x, y))
        return canvas

    if image.mode == "RGBA":
        canvas.paste(image.convert("RGB"), (x, y), image.getchannel("A"))
    else:
        canvas.paste(image.convert("RGB"), (x, y))
    return canvas


def fit_image(
    image: Image.Image,
    target_size: tuple[int, int],
    fit_mode: str,
    background: str | None,
) -> Image.Image:
    src_w, src_h = image.size
    dst_w, dst_h = target_size

    if fit_mode == "Fill page and crop":
        scale = max(dst_w / src_w, dst_h / src_h)
        new_size = (max(1, round(src_w * scale)), max(1, round(src_h * scale)))
        resized = image.resize(new_size, Image.Resampling.LANCZOS, reducing_gap=3.0)
        left = max(0, (resized.width - dst_w) // 2)
        top = max(0, (resized.height - dst_h) // 2)
        return resized.crop((left, top, left + dst_w, top + dst_h))

    scale = min(dst_w / src_w, dst_h / src_h)
    new_size = (max(1, round(src_w * scale)), max(1, round(src_h * scale)))
    resized = image.resize(new_size, Image.Resampling.LANCZOS, reducing_gap=3.0)
    canvas = make_background_canvas(target_size, background)
    x = (dst_w - resized.width) // 2
    y = (dst_h - resized.height) // 2
    return alpha_composite_center(canvas, resized, x, y)


def apply_gentle_sharpen(image: Image.Image, amount: float) -> Image.Image:
    if amount <= 0:
        return image
    percent = int(40 + amount * 100)
    radius = 0.55 + amount * 0.55
    threshold = 2
    return image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))


def flatten_image(image: Image.Image, background: str = "#FFFFFF") -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    base = Image.new("RGB", image.size, background)
    base.paste(image, mask=image.getchannel("A"))
    return base


def encode_image(image: Image.Image, fmt: str, dpi: int, background: str | None) -> bytes:
    out = io.BytesIO()
    icc = srgb_profile_bytes()
    fmt = fmt.upper()

    if fmt == "PNG":
        kwargs = {"format": "PNG", "dpi": (dpi, dpi), "compress_level": 6}
        if icc:
            kwargs["icc_profile"] = icc
        image.save(out, **kwargs)

    elif fmt in ("JPG", "JPEG"):
        save_image = flatten_image(image, background or "#FFFFFF")
        kwargs = {
            "format": "JPEG",
            "quality": 96,
            "subsampling": 0,
            "optimize": True,
            "dpi": (dpi, dpi),
        }
        if icc:
            kwargs["icc_profile"] = icc
        save_image.save(out, **kwargs)

    elif fmt in ("TIF", "TIFF"):
        kwargs = {"format": "TIFF", "compression": "tiff_lzw", "dpi": (dpi, dpi)}
        if icc:
            kwargs["icc_profile"] = icc
        image.save(out, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    return out.getvalue()


def image_to_exact_pdf(image: Image.Image, width_in: float, height_in: float, dpi: int) -> bytes:
    png_bytes = encode_image(flatten_image(image), "PNG", dpi, "#FFFFFF")
    doc = fitz.open()
    page = doc.new_page(width=width_in * 72.0, height=height_in * 72.0)
    page.insert_image(page.rect, stream=png_bytes, keep_proportion=False, overlay=True)
    result = doc.tobytes(garbage=4, deflate=True)
    doc.close()
    return result


def assess_quality(src_size: tuple[int, int], target_size: tuple[int, int], target_dpi: int) -> dict:
    src_w, src_h = src_size
    dst_w, dst_h = target_size
    scale_x = dst_w / src_w
    scale_y = dst_h / src_h
    largest_scale = max(scale_x, scale_y)
    megapixels = (dst_w * dst_h) / 1_000_000

    if largest_scale <= 1:
        status = "Excellent — no enlargement required"
    elif largest_scale <= 2:
        status = "Good — moderate enlargement"
    elif largest_scale <= 4:
        status = "Usable, but inspect small text and fine lines"
    else:
        status = "Source is very small — AI can help, but detail still has limits"

    return {
        "Original pixels": f"{src_w:,} × {src_h:,}",
        "Output pixels": f"{dst_w:,} × {dst_h:,}",
        "Largest enlargement": f"{largest_scale:.2f}×",
        "Output megapixels": f"{megapixels:.1f} MP",
        "Target DPI": f"{target_dpi} DPI",
        "Assessment": status,
    }


def render_pdf_page(page: fitz.Page, dpi: int) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    mode = "RGB" if pix.n < 4 else "RGBA"
    image = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    return normalize_image(image)


def process_single_image(filename: str, data: bytes, spec: OutputSpec):
    image = open_image_bytes(data)
    original_size = image.size

    if spec.ai_model != "off":
        with st.spinner("Running Real-ESRGAN AI upscaling…"):
            image = apply_realesrgan(image, spec.ai_model, spec.ai_outscale)

    target_size = spec.target_px
    if target_size[0] * target_size[1] > MAX_OUTPUT_PIXELS:
        raise ValueError("Requested output is too large. Reduce the page size or DPI.")

    output_image = fit_image(image, target_size, spec.fit_mode, spec.background)
    output_image = apply_gentle_sharpen(output_image, spec.sharpen)

    stem = safe_name(filename)
    outputs = []
    for fmt in spec.output_formats:
        if fmt == "PDF":
            outputs.append((f"{stem}_PRINT_READY_{spec.dpi}dpi.pdf",
                            image_to_exact_pdf(output_image, spec.final_width_in, spec.final_height_in, spec.dpi)))
        else:
            ext = {"JPEG": "jpg", "TIFF": "tif"}.get(fmt, fmt.lower())
            outputs.append((f"{stem}_PRINT_READY_{spec.dpi}dpi.{ext}",
                            encode_image(output_image, fmt, spec.dpi, spec.background)))

    report = assess_quality(original_size, target_size, spec.dpi)
    report["AI model"] = spec.ai_model if spec.ai_model != "off" else "Off"
    return outputs, report, output_image


def process_pdf(filename: str, data: bytes, spec: OutputSpec):
    source = fitz.open(stream=data, filetype="pdf")
    output_doc = fitz.open()
    outputs = []
    reports = []
    preview = None
    stem = safe_name(filename)
    page_pngs = []

    for index, page in enumerate(source):
        original_width_in = page.rect.width / 72.0
        original_height_in = page.rect.height / 72.0

        if spec.keep_pdf_size:
            page_width_in = original_width_in
            page_height_in = original_height_in
            target_px = (max(1, round(page_width_in * spec.dpi)), max(1, round(page_height_in * spec.dpi)))
        else:
            page_width_in = spec.final_width_in
            page_height_in = spec.final_height_in
            target_px = spec.target_px

        rendered = render_pdf_page(page, spec.dpi)

        if spec.ai_model != "off":
            with st.spinner(f"Running Real-ESRGAN on PDF page {index + 1}…"):
                rendered = apply_realesrgan(rendered, spec.ai_model, spec.ai_outscale)

        if spec.keep_pdf_size:
            page_image = rendered.resize(target_px, Image.Resampling.LANCZOS, reducing_gap=3.0) \
                if rendered.size != target_px else rendered
        else:
            page_image = fit_image(rendered, target_px, spec.fit_mode, spec.background)

        page_image = apply_gentle_sharpen(page_image, spec.sharpen)
        page_image = flatten_image(page_image, spec.background or "#FFFFFF")

        if preview is None:
            preview = page_image.copy()

        png_bytes = encode_image(page_image, "PNG", spec.dpi, "#FFFFFF")
        page_pngs.append((f"{stem}_page_{index + 1:02d}_{spec.dpi}dpi.png", png_bytes))

        out_page = output_doc.new_page(width=page_width_in * 72.0, height=page_height_in * 72.0)
        out_page.insert_image(out_page.rect, stream=png_bytes, keep_proportion=False, overlay=True)

        report = assess_quality(rendered.size, target_px, spec.dpi)
        report["Page"] = index + 1
        report["Final print size"] = f"{page_width_in:.3f} × {page_height_in:.3f} in"
        report["AI model"] = spec.ai_model if spec.ai_model != "off" else "Off"
        reports.append(report)

    outputs.append((f"{stem}_PRINT_READY_{spec.dpi}dpi.pdf", output_doc.tobytes(garbage=4, deflate=True)))

    if "PNG" in spec.output_formats:
        outputs.extend(page_pngs)
    if "TIFF" in spec.output_formats:
        for png_name, png_data in page_pngs:
            page_image = open_image_bytes(png_data)
            outputs.append((png_name.rsplit(".", 1)[0] + ".tif", encode_image(page_image, "TIFF", spec.dpi, "#FFFFFF")))
    if "JPEG" in spec.output_formats:
        for png_name, png_data in page_pngs:
            page_image = open_image_bytes(png_data)
            outputs.append((png_name.rsplit(".", 1)[0] + ".jpg", encode_image(page_image, "JPEG", spec.dpi, "#FFFFFF")))

    source.close()
    output_doc.close()
    return outputs, reports, preview


def build_zip(files: Iterable[tuple[str, bytes]], report_text: str) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, data in files:
            zf.writestr(filename, data)
        zf.writestr("PRINT_READINESS_REPORT.txt", report_text)
    return out.getvalue()


def format_report(all_reports: list[tuple[str, dict]]) -> str:
    lines = [
        APP_TITLE,
        "=" * len(APP_TITLE),
        "",
        "This report confirms the requested pixel dimensions and DPI metadata.",
        "Real-ESRGAN AI upscaling was used when selected.",
        "",
    ]
    for filename, report in all_reports:
        lines.append(filename)
        lines.append("-" * len(filename))
        for key, value in report.items():
            lines.append(f"{key}: {value}")
        lines.append("")
    return "\n".join(lines)


def inject_style() -> None:
    st.markdown(
        """
        <style>
          .stApp { background: #FDF6EA; }
          h1, h2, h3 { color: #3A2C20; }
          div[data-testid="stMetric"] {
              background: #FFF4E6;
              border: 1px solid #E8D9BF;
              border-radius: 16px;
              padding: 10px;
          }
          div.stButton > button, div.stDownloadButton > button {
              border-radius: 12px;
              border: 0;
              background: #7F946F;
              color: white;
              font-weight: 700;
          }
          .note {
              background: #FFF4E6;
              border-left: 5px solid #CF6F4A;
              border-radius: 12px;
              padding: 14px 16px;
              color: #3A2C20;
              margin-bottom: 14px;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🖨️", layout="wide")
    inject_style()

    st.title("🖨️ Real-ESRGAN + 300 DPI Print App")
    st.caption("A stronger local app for AI clarity improvement plus exact print-ready export.")

    st.markdown(
        """
        <div class="note">
        <b>This is the stronger version:</b> it uses <b>Real-ESRGAN</b> locally on your computer,
        which is much more capable than the browser-only HTML attempt. It still exports exact print sizes,
        real 300 DPI dimensions and exact-size PDFs.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload one or more images or PDFs",
        type=["png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp", "pdf"],
        accept_multiple_files=True,
    )

    settings_col, preview_col = st.columns([0.95, 1.25], gap="large")

    with settings_col:
        st.subheader("Print settings")

        preset = st.selectbox("Finished size", list(PAGE_PRESETS.keys()), index=0)
        preset_unit, preset_w, preset_h = PAGE_PRESETS[preset]

        if preset == "Custom size":
            unit = st.selectbox("Units", ["mm", "cm", "in"], index=0)
            width = st.number_input("Width", min_value=1.0, value=210.0, step=1.0)
            height = st.number_input("Height", min_value=1.0, value=297.0, step=1.0)
        else:
            unit, width, height = preset_unit, preset_w, preset_h

        orientation = st.radio("Orientation", ["Portrait", "Landscape"], horizontal=True)
        if orientation == "Landscape" and height > width:
            width, height = height, width
        elif orientation == "Portrait" and width > height:
            width, height = height, width

        width_in, height_in = to_inches(unit, float(width), float(height))
        dpi = st.slider("Output DPI", min_value=300, max_value=600, value=300, step=50)
        bleed_mm = st.selectbox("Bleed", [0.0, 3.0, 5.0], format_func=lambda x: f"{x:g} mm")
        fit_mode = st.radio("Fit mode", ["Keep whole design + add margins", "Fill page and crop"])
        background_name = st.selectbox("Margin/background", list(BACKGROUND_COLOURS.keys()), index=0)
        background = BACKGROUND_COLOURS[background_name]

        st.subheader("AI clarity")
        model_label = st.selectbox("AI model", list(AI_MODELS.keys()), index=0)
        ai_model = AI_MODELS[model_label]
        ai_outscale = st.selectbox("AI upscale amount", [1.5, 2.0, 3.0, 4.0], index=1)
        sharpen = st.slider("Gentle final sharpening", min_value=0.0, max_value=1.0, value=0.20, step=0.05)

        output_formats = st.multiselect("Export formats", ["PDF", "PNG", "TIFF", "JPEG"], default=["PDF", "PNG"])
        keep_pdf_size = st.checkbox("For uploaded PDFs, keep each PDF page's original physical size", value=True)

        final_w_in = width_in + (2 * bleed_mm / 25.4)
        final_h_in = height_in + (2 * bleed_mm / 25.4)
        target_w = round(final_w_in * dpi)
        target_h = round(final_h_in * dpi)

        c1, c2, c3 = st.columns(3)
        c1.metric("Pixels", f"{target_w:,} × {target_h:,}")
        c2.metric("DPI", f"{dpi}")
        c3.metric("Output", f"{target_w * target_h / 1_000_000:.1f} MP")

    with preview_col:
        st.subheader("Preview and quality check")
        if uploaded_files:
            first = uploaded_files[0]
            first_data = first.getvalue()
            try:
                if first.name.lower().endswith(".pdf"):
                    doc = fitz.open(stream=first_data, filetype="pdf")
                    if len(doc):
                        preview_img = render_pdf_page(doc[0], min(dpi, 180))
                        st.image(preview_img, caption=f"{first.name} — first page preview", use_container_width=True)
                    doc.close()
                else:
                    img = open_image_bytes(first_data)
                    st.image(img, caption=f"{first.name} — original preview", use_container_width=True)
                    st.dataframe(
                        [{"Check": k, "Result": v} for k, v in assess_quality(img.size, (target_w, target_h), dpi).items()],
                        hide_index=True,
                        use_container_width=True,
                    )
            except Exception as exc:
                st.error(f"Preview could not be created: {exc}")
        else:
            st.info("Upload an image or PDF to see its print-readiness assessment.")

    if not uploaded_files:
        return
    if not output_formats:
        st.warning("Choose at least one export format.")
        return

    spec = OutputSpec(
        width_in=width_in,
        height_in=height_in,
        dpi=dpi,
        fit_mode=fit_mode,
        background=background,
        bleed_mm=float(bleed_mm),
        sharpen=float(sharpen),
        output_formats=list(output_formats),
        ai_model=ai_model,
        ai_outscale=float(ai_outscale),
        keep_pdf_size=keep_pdf_size,
    )

    if st.button("Create AI-enhanced print-ready files", type="primary", use_container_width=True):
        produced_files = []
        all_reports = []
        errors = []
        final_preview = None

        progress = st.progress(0, text="Preparing files…")
        for index, uploaded in enumerate(uploaded_files):
            try:
                data = uploaded.getvalue()
                if uploaded.name.lower().endswith(".pdf"):
                    outputs, reports, preview = process_pdf(uploaded.name, data, spec)
                    produced_files.extend(outputs)
                    for page_report in reports:
                        all_reports.append((f"{uploaded.name} — page {page_report.get('Page')}", page_report))
                    if final_preview is None:
                        final_preview = preview
                else:
                    outputs, report, preview = process_single_image(uploaded.name, data, spec)
                    produced_files.extend(outputs)
                    all_reports.append((uploaded.name, report))
                    if final_preview is None:
                        final_preview = preview
            except Exception as exc:
                errors.append(f"{uploaded.name}: {exc}")

            progress.progress((index + 1) / len(uploaded_files),
                              text=f"Processed {index + 1} of {len(uploaded_files)} file(s)")

        report_text = format_report(all_reports)
        if produced_files:
            st.session_state["zip_bytes"] = build_zip(produced_files, report_text)
            st.session_state["reports"] = all_reports
            st.session_state["preview"] = final_preview
            st.session_state["errors"] = errors
        else:
            st.error("No output files were created.")
            for error in errors:
                st.error(error)

    if "zip_bytes" in st.session_state:
        st.success("Your AI-enhanced print-ready export is complete.")
        if st.session_state.get("preview") is not None:
            st.image(st.session_state["preview"], caption="Processed preview", use_container_width=True)

        st.download_button(
            "Download all print-ready files (.zip)",
            data=st.session_state["zip_bytes"],
            file_name=f"Teddie_Lane_RealESRGAN_Print_Ready_{dpi}dpi.zip",
            mime="application/zip",
            use_container_width=True,
        )

        rows = []
        for filename, report in st.session_state.get("reports", []):
            row = {"File": filename}
            row.update(report)
            rows.append(row)
        if rows:
            st.dataframe(rows, hide_index=True, use_container_width=True)

        for error in st.session_state.get("errors", []):
            st.warning(error)


if __name__ == "__main__":
    main()
