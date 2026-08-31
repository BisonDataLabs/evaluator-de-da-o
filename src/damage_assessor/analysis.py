from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .geometry import feature_area_ha_approx, iter_polygons


@dataclass(frozen=True)
class RealChangeResult:
    change_png: bytes
    summary: dict[str, Any]
    lot_summaries: list[dict[str, Any]]


def _rgb(payload: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(payload)).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def _mask(
    feature: dict[str, Any],
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> np.ndarray:
    min_x, min_y, max_x, max_y = bbox
    span_x = max(max_x - min_x, 1e-12)
    span_y = max(max_y - min_y, 1e-12)

    def pixel(point: list[float]) -> tuple[int, int]:
        x = int((float(point[0]) - min_x) / span_x * (width - 1))
        y = int((max_y - float(point[1])) / span_y * (height - 1))
        return x, y

    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for polygon in iter_polygons(feature["geometry"]):
        draw.polygon([pixel(point) for point in polygon[0]], fill=255)
        for hole in polygon[1:]:
            draw.polygon([pixel(point) for point in hole], fill=0)
    return np.asarray(canvas, dtype=np.uint8) > 0


def calculate_rendered_radar_change(
    before_png: bytes,
    after_png: bytes,
    geojson: dict[str, Any],
    render_bbox: tuple[float, float, float, float],
) -> RealChangeResult:
    """Legacy visual-screening utility; the Streamlit app never uses it for metrics."""
    before = _rgb(before_png)
    after = _rgb(after_png)
    if before.shape != after.shape:
        raise ValueError("Las imágenes anterior y posterior no tienen la misma grilla.")
    height, width = before.shape[:2]
    feature_masks = [_mask(feature, render_bbox, width, height) for feature in geojson["features"]]
    aoi_mask = np.logical_or.reduce(feature_masks)
    if not aoi_mask.any():
        raise ValueError("El recorte no contiene píxeles de los lotes seleccionados.")

    absolute_change = np.mean(np.abs(after - before), axis=2)
    values = absolute_change[aoi_mask]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = max(1.4826 * mad, 0.01)
    z_score = np.maximum((absolute_change - median) / robust_sigma, 0)
    changed = np.logical_and(z_score >= 3.0, aoi_mask)
    uncertain = np.logical_and((z_score >= 2.0) & (z_score < 3.0), aoi_mask)

    total_area = sum(feature_area_ha_approx(feature) for feature in geojson["features"])
    pixel_area = total_area / max(int(aoi_mask.sum()), 1)
    summary = {
        "reference_area_ha": round(total_area, 2),
        "radar_change_area_ha_preliminary": round(float(changed.sum() * pixel_area), 2),
        "uncertain_change_area_ha_preliminary": round(float(uncertain.sum() * pixel_area), 2),
        "radar_change_share_pct_preliminary": round(float(changed.sum() / aoi_mask.sum() * 100), 1),
        "method": "Cambio robusto sobre renders SAR reales; umbral z-MAD >= 3",
        "severity_status": "NO_CALIBRADA",
    }

    lot_summaries: list[dict[str, Any]] = []
    for feature, mask in zip(geojson["features"], feature_masks, strict=True):
        properties = feature.get("properties") or {}
        area = feature_area_ha_approx(feature)
        denominator = max(int(mask.sum()), 1)
        lot_summaries.append(
            {
                "pedido": properties.get("pedido", properties.get("lote", "")),
                "establecimiento": properties.get("establecimiento", ""),
                "area_referencia_ha": round(area, 2),
                "cambio_radar_ha_preliminar": round(area * float(changed[mask].sum()) / denominator, 2),
                "cambio_incierto_ha_preliminar": round(
                    area * float(uncertain[mask].sum()) / denominator, 2
                ),
                "severidad_satelital": "No calibrada",
            }
        )

    strength = np.clip(z_score / 6.0, 0, 1)
    heat = np.zeros((height, width, 4), dtype=np.uint8)
    heat[..., 0] = np.asarray(255 * strength, dtype=np.uint8)
    heat[..., 1] = np.asarray(210 * (1 - strength), dtype=np.uint8)
    heat[..., 2] = 40
    heat[..., 3] = np.where(aoi_mask, 230, 0).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(heat, mode="RGBA").save(output, format="PNG")
    return RealChangeResult(output.getvalue(), summary, lot_summaries)
