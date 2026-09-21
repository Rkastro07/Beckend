"""Bound PDF rasterization before allocating a bitmap, not after it."""
from __future__ import annotations

import math
from pathlib import Path

MAX_SIDE = 4096
MAX_PIXELS = 12_000_000
MAX_SCALE = 3.0


def bounded_scale(width: float, height: float) -> float:
    if not all(math.isfinite(v) and v > 0 for v in (width, height)):
        raise ValueError("Dimensões inválidas na página do PDF.")
    scale = min(
        MAX_SCALE,
        MAX_SIDE / max(width, height),
        math.sqrt(MAX_PIXELS) / math.sqrt(width) / math.sqrt(height),
    )
    # PDFium rounds each side up. Account for that before allocating pixels.
    while (max(math.ceil(width * scale), math.ceil(height * scale)) > MAX_SIDE
           or math.ceil(width * scale) * math.ceil(height * scale) > MAX_PIXELS):
        scale *= 0.999
    return scale


def render_first_page(source: Path, destination: Path) -> dict:
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(source))
    try:
        if len(document) == 0:
            raise ValueError("PDF sem páginas.")
        page = document[0]
        try:
            width, height = page.get_size()
            scale = bounded_scale(width, height)
            bitmap = page.render(scale=scale)
            try:
                image = bitmap.to_pil().convert("RGB")
                try:
                    image.save(destination, format="PNG", optimize=True)
                    return {
                        "page": 1,
                        "page_count": len(document),
                        "page_size_pt": [width, height],
                        "size_px": list(image.size),
                        "render_scale": scale,
                        "adaptive": scale < MAX_SCALE,
                        "max_side": MAX_SIDE,
                        "max_pixels": MAX_PIXELS,
                    }
                finally:
                    image.close()
            finally:
                bitmap.close()
        finally:
            page.close()
    finally:
        document.close()
