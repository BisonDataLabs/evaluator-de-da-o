from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageOps


def comparison_image(before_png: bytes, after_png: bytes, position_pct: int) -> Image.Image:
    """Build a deterministic side-by-side image for Streamlit's native image renderer."""
    if not 0 <= position_pct <= 100:
        raise ValueError("La posición del divisor debe estar entre 0 y 100.")
    before = Image.open(io.BytesIO(before_png)).convert("RGB")
    after = Image.open(io.BytesIO(after_png)).convert("RGB")
    width = min(before.width, after.width)
    height = min(before.height, after.height)
    before = ImageOps.fit(before, (width, height))
    after = ImageOps.fit(after, (width, height))
    split = round(width * position_pct / 100)
    output = before.copy()
    if split < width:
        output.paste(after.crop((split, 0, width, height)), (split, 0))
    draw = ImageDraw.Draw(output)
    draw.line((split, 0, split, height), fill="white", width=5)
    draw.line((max(0, split - 2), 0, max(0, split - 2), height), fill="#17313B", width=1)
    return output
