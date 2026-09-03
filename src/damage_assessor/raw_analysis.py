from __future__ import annotations

import io
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .catalog import SceneAcquisition, SceneCandidate
from .geometry import feature_area_ha_approx, geojson_bbox
from .imagery import fetch_raw_array, native_grid_size


@dataclass(frozen=True)
class SensorMetricResult:
    sensor: str
    summary: dict[str, Any]
    lot_summaries: list[dict[str, Any]]
    overlays: tuple[dict[str, Any], ...] = ()


RADAR_EVENT_CHANGE_THRESHOLD_DB = 2.5626
RADAR_EVENT_POSTERIOR_THRESHOLD_DB = -18.3445
RADAR_DECISION_LOW_DB = 2.1
RADAR_DECISION_HIGH_DB = 3.1
RADAR_PIXEL_THRESHOLDS_DB = (2.202, 2.916, 3.812)

RADAR_CLASS_COLORS = {
    0: (205, 216, 224, 105),
    1: (249, 207, 92, 190),
    2: (239, 126, 64, 210),
    3: (166, 38, 58, 230),
}

_PATTERN_FEATURE_CENTER = np.asarray(
    [0.501522, 3.211228, 0.793259, 3.523374, 0.234254, -16.822468, -12.757853],
    dtype=np.float64,
)
_PATTERN_FEATURE_SCALE = np.asarray(
    [1.108129, 1.146009, 0.802213, 0.685360, 0.942199, 1.479230, 2.643879],
    dtype=np.float64,
)
_PATTERN_CENTERS = {
    "Cambio amplio e intenso": np.asarray(
        [2.083282, 2.266789, 1.619253, 2.061361, -1.062773, 0.235620, -0.056473]
    ),
    "Aumento con señal posterior alta": np.asarray(
        [0.581326, 0.547297, -0.302746, -0.418039, -0.844212, 1.371475, 1.341632]
    ),
    "Cambio localizado o de menor amplitud": np.asarray(
        [-0.234708, -0.167606, -0.014954, 0.112387, 0.330238, -0.556418, 0.055640]
    ),
}


def _feature_collection(feature: dict[str, Any]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": [feature]}


def _bbox_overlap(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def _scene_for_feature(
    acquisition: SceneAcquisition, feature: dict[str, Any]
) -> SceneCandidate | None:
    feature_bbox = geojson_bbox(_feature_collection(feature))
    ranked = sorted(
        (
            (_bbox_overlap(feature_bbox, scene.bbox), scene)
            for scene in acquisition.scenes
            if scene.bbox is not None
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    if ranked and ranked[0][0] > 0:
        return ranked[0][1]
    return acquisition.scenes[0] if len(acquisition.scenes) == 1 else None


def _mask_for_feature(
    feature: dict[str, Any],
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> np.ndarray:
    from .analysis import _mask

    return _mask(feature, bbox, width, height)


def _median_filter(array: np.ndarray, size: int = 3) -> np.ndarray:
    padding = size // 2
    padded = np.pad(array, padding, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        return np.nanmedian(windows, axis=(-2, -1)).astype(np.float32)


def _erode_mask(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    return np.all(windows, axis=(-2, -1))


def _remove_small_regions(
    classes: np.ndarray,
    valid: np.ndarray,
    minimum_pixels: int = 4,
) -> None:
    active = valid & (classes > 0)
    seen = np.zeros(active.shape, dtype=bool)
    height, width = active.shape
    for row, column in np.argwhere(active):
        if seen[row, column]:
            continue
        stack = [(int(row), int(column))]
        seen[row, column] = True
        component: list[tuple[int, int]] = []
        while stack:
            current_row, current_column = stack.pop()
            component.append((current_row, current_column))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    next_row = current_row + dy
                    next_column = current_column + dx
                    if not (0 <= next_row < height and 0 <= next_column < width):
                        continue
                    if active[next_row, next_column] and not seen[next_row, next_column]:
                        seen[next_row, next_column] = True
                        stack.append((next_row, next_column))
        if len(component) < minimum_pixels:
            for current_row, current_column in component:
                classes[current_row, current_column] = 0


def _class_png(classes: np.ndarray, valid: np.ndarray) -> bytes:
    rgba = np.zeros((*classes.shape, 4), dtype=np.uint8)
    for level, color in RADAR_CLASS_COLORS.items():
        rgba[valid & (classes == level)] = color
    output = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(output, format="PNG", optimize=True)
    return output.getvalue()


def _radar_decision(delta_vh_p95_db: float) -> str:
    if delta_vh_p95_db >= RADAR_DECISION_HIGH_DB:
        return "Cambio claro"
    if delta_vh_p95_db <= RADAR_DECISION_LOW_DB:
        return "Sin cambio claro"
    return "Indeterminado"


def _radar_pattern(features: np.ndarray, event_detected: bool) -> str:
    if not event_detected:
        return "Sin patrón asignado"
    standardized = (features - _PATTERN_FEATURE_CENTER) / _PATTERN_FEATURE_SCALE
    return min(
        _PATTERN_CENTERS,
        key=lambda name: float(np.sum((standardized - _PATTERN_CENTERS[name]) ** 2)),
    )


def _identity(feature: dict[str, Any]) -> dict[str, Any]:
    properties = feature.get("properties") or {}
    return {
        "pedido": properties.get("pedido", properties.get("lote", "")),
        "establecimiento": properties.get("establecimiento", ""),
        "area_referencia_ha": round(feature_area_ha_approx(feature), 2),
    }


def _severity_class(score: float) -> str:
    if score < 1:
        return "Sin señal"
    if score < 3:
        return "Leve"
    if score < 5:
        return "Moderada"
    if score < 7.5:
        return "Severa"
    return "Muy severa"


def _run_by_feature(
    geojson: dict[str, Any],
    worker: Callable[[dict[str, Any]], dict[str, Any]],
    max_workers: int = 4,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any] | None] = [None] * len(geojson["features"])
    with ThreadPoolExecutor(max_workers=min(max_workers, len(output) or 1)) as executor:
        futures = {
            executor.submit(worker, feature): index
            for index, feature in enumerate(geojson["features"])
        }
        for future in as_completed(futures):
            index = futures[future]
            feature = geojson["features"][index]
            try:
                output[index] = future.result()
            except Exception as exc:  # noqa: BLE001 - one lot must not invalidate the group
                output[index] = {
                    **_identity(feature),
                    "estado": "No calculado",
                    "motivo": str(exc),
                }
    return [row for row in output if row is not None]


def calculate_optical_metrics(
    before: SceneAcquisition,
    after: SceneAcquisition,
    geojson: dict[str, Any],
    cloud_limit_pct: float = 10.0,
) -> SensorMetricResult:
    """Calculate S2 index change from numerical bands with an SCL mask per lot."""

    def worker(feature: dict[str, Any]) -> dict[str, Any]:
        identity = _identity(feature)
        pre_scene = _scene_for_feature(before, feature)
        post_scene = _scene_for_feature(after, feature)
        if pre_scene is None or post_scene is None:
            return {
                **identity,
                "estado": "No calculado",
                "motivo": "La adquisición no cubre completamente este lote.",
            }
        bbox = geojson_bbox(_feature_collection(feature))
        width, height = native_grid_size(bbox, target_meters=20, max_dimension=768)
        mask = _mask_for_feature(feature, bbox, width, height)

        def read(scene: SceneCandidate, assets: tuple[str, ...], expression: str) -> np.ndarray:
            return fetch_raw_array(
                scene,
                bbox,
                assets=assets,
                expression=expression,
                width=width,
                height=height,
                unscale=True,
            )

        pre_scl = fetch_raw_array(
            pre_scene,
            bbox,
            assets=("SCL",),
            expression="SCL",
            width=width,
            height=height,
            categorical=True,
        )
        post_scl = fetch_raw_array(
            post_scene,
            bbox,
            assets=("SCL",),
            expression="SCL",
            width=width,
            height=height,
            categorical=True,
        )
        cloud_classes = (3, 8, 9, 10, 11)
        observed_pre = mask & np.isfinite(pre_scl) & (pre_scl > 0)
        observed_post = mask & np.isfinite(post_scl) & (post_scl > 0)
        pre_classes = np.nan_to_num(np.rint(pre_scl), nan=0).astype(np.int16)
        post_classes = np.nan_to_num(np.rint(post_scl), nan=0).astype(np.int16)
        cloud_pre = observed_pre & np.isin(pre_classes, cloud_classes)
        cloud_post = observed_post & np.isin(post_classes, cloud_classes)
        pre_cloud_pct = 100 * cloud_pre.sum() / max(observed_pre.sum(), 1)
        post_cloud_pct = 100 * cloud_post.sum() / max(observed_post.sum(), 1)
        cloud_values = {
            "nubes_lote_anterior_pct": round(float(pre_cloud_pct), 1),
            "nubes_lote_posterior_pct": round(float(post_cloud_pct), 1),
        }
        if pre_cloud_pct > cloud_limit_pct or post_cloud_pct > cloud_limit_pct:
            return {
                **identity,
                **cloud_values,
                "estado": "Bloqueado por nubes",
                "motivo": f"Más de {cloud_limit_pct:.0f}% de nube/sombra dentro del lote.",
            }

        indices = {
            "ndvi": (("B08", "B04"), "(B08-B04)/(B08+B04)"),
            "ndre": (("B8A", "B05"), "(B8A-B05)/(B8A+B05)"),
            "ndmi": (("B08", "B11"), "(B08-B11)/(B08+B11)"),
        }
        deltas: dict[str, np.ndarray] = {}
        clear = mask & observed_pre & observed_post & ~cloud_pre & ~cloud_post
        clear &= ~np.isin(pre_classes, (1, 2)) & ~np.isin(post_classes, (1, 2))
        for name, (assets, expression) in indices.items():
            pre = read(pre_scene, assets, expression)
            post = read(post_scene, assets, expression)
            deltas[name] = post - pre
            clear &= np.isfinite(pre) & np.isfinite(post) & (pre >= -1.2) & (pre <= 1.2)
            clear &= (post >= -1.2) & (post <= 1.2)
        clear_pct = 100 * clear.sum() / max(mask.sum(), 1)
        if clear_pct < 90:
            return {
                **identity,
                **cloud_values,
                "pixeles_validos_pct": round(float(clear_pct), 1),
                "estado": "No calculado",
                "motivo": "La cobertura válida conjunta es menor a 90% del lote.",
            }

        votes = (
            (deltas["ndvi"] <= -0.12).astype(np.uint8)
            + (deltas["ndre"] <= -0.08).astype(np.uint8)
            + (deltas["ndmi"] <= -0.12).astype(np.uint8)
        )
        affected = clear & (votes >= 2)
        affected_share = affected.sum() / max(clear.sum(), 1)
        area = float(identity["area_referencia_ha"])
        drops = []
        for name, scale in (("ndvi", 0.30), ("ndre", 0.22), ("ndmi", 0.30)):
            values = np.maximum(-deltas[name][clear], 0) / scale
            drops.append(float(np.nanmedian(np.clip(values, 0, 1))))
        intensity = float(np.mean(drops))
        score = round(10 * np.clip(0.55 * intensity + 0.45 * affected_share, 0, 1), 1)
        return {
            **identity,
            **cloud_values,
            "pixeles_validos_pct": round(float(clear_pct), 1),
            "delta_ndvi_mediana": round(float(np.nanmedian(deltas["ndvi"][clear])), 3),
            "delta_ndre_mediana": round(float(np.nanmedian(deltas["ndre"][clear])), 3),
            "delta_ndmi_mediana": round(float(np.nanmedian(deltas["ndmi"][clear])), 3),
            "area_perturbada_ha_preliminar": round(area * float(affected_share), 2),
            "superficie_perturbada_pct": round(100 * float(affected_share), 1),
            "severidad_satelital_0_10_preliminar": score,
            "clase_preliminar": _severity_class(score),
            "estado": "Calculado",
            "motivo": "",
        }

    rows = _run_by_feature(geojson, worker)
    calculated = [row for row in rows if row.get("estado") == "Calculado"]
    blocked = [row for row in rows if row.get("estado") == "Bloqueado por nubes"]
    total_area = sum(float(row.get("area_referencia_ha") or 0) for row in rows)
    changed_area = sum(float(row.get("area_perturbada_ha_preliminar") or 0) for row in calculated)
    return SensorMetricResult(
        sensor="Sentinel-2 L2A",
        summary={
            "area_referencia_ha": round(total_area, 2),
            "area_perturbada_ha_preliminar": round(changed_area, 2),
            "lotes_calculados": len(calculated),
            "lotes_bloqueados_por_nubes": len(blocked),
            "metodo": "Bandas numéricas L2A + SCL por lote; acuerdo 2 de 3 índices",
        },
        lot_summaries=rows,
    )


def calculate_radar_metrics(
    before: SceneAcquisition,
    after: SceneAcquisition,
    geojson: dict[str, Any],
) -> SensorMetricResult:
    """Calculate the calibrated S1 intralot change product on a common per-lot grid."""
    comparable = (
        before.relative_orbit is not None
        and before.relative_orbit == after.relative_orbit
        and before.orbit_state == after.orbit_state
    )
    if not comparable:
        rows = [
            {
                **_identity(feature),
                "estado": "No calculado",
                "motivo": "Las adquisiciones no comparten órbita relativa y dirección.",
            }
            for feature in geojson["features"]
        ]
        return SensorMetricResult(
            "Sentinel-1 RTC",
            {
                "area_referencia_ha": round(
                    sum(float(row["area_referencia_ha"]) for row in rows), 2
                ),
                "cambio_total_ha": 0.0,
                "cambio_moderado_fuerte_ha": 0.0,
                "lotes_calculados": 0,
                "metodo": "Bloqueado por geometría de adquisición no comparable",
            },
            rows,
        )

    def worker(feature: dict[str, Any]) -> dict[str, Any]:
        identity = _identity(feature)
        pre_scene = _scene_for_feature(before, feature)
        post_scene = _scene_for_feature(after, feature)
        if pre_scene is None or post_scene is None:
            return {
                **identity,
                "estado": "No calculado",
                "motivo": "La pasada no cubre completamente este lote.",
            }
        if not {"vv", "vh"}.issubset(set(pre_scene.asset_keys)) or not {
            "vv",
            "vh",
        }.issubset(set(post_scene.asset_keys)):
            return {
                **identity,
                "estado": "No calculado",
                "motivo": "El producto no contiene simultáneamente VV y VH.",
            }
        bbox = geojson_bbox(_feature_collection(feature))
        width, height = native_grid_size(bbox, target_meters=10, max_dimension=768)
        mask = _mask_for_feature(feature, bbox, width, height)
        interior = _erode_mask(mask)

        def read(scene: SceneCandidate, asset: str) -> np.ndarray:
            array = fetch_raw_array(
                scene,
                bbox,
                assets=(asset,),
                expression=asset,
                width=width,
                height=height,
            )
            array = np.where(array > 0, array, np.nan)
            return _median_filter(array, 3)

        pre_vv = read(pre_scene, "vv")
        pre_vh = read(pre_scene, "vh")
        post_vv = read(post_scene, "vv")
        post_vh = read(post_scene, "vh")
        valid = interior & np.isfinite(pre_vv) & np.isfinite(pre_vh)
        valid &= np.isfinite(post_vv) & np.isfinite(post_vh)
        valid_pct = 100 * valid.sum() / max(interior.sum(), 1)
        if valid.sum() < 25 or valid_pct < 90:
            return {
                **identity,
                "pixeles_validos_pct": round(float(valid_pct), 1),
                "estado": "No calculado",
                "motivo": "La cobertura radar válida conjunta es menor a 90% del interior del lote.",
            }

        pre_vv_db = 10 * np.log10(np.maximum(pre_vv, 1e-8))
        pre_vh_db = 10 * np.log10(np.maximum(pre_vh, 1e-8))
        post_vv_db = 10 * np.log10(np.maximum(post_vv, 1e-8))
        post_vh_db = 10 * np.log10(np.maximum(post_vh, 1e-8))
        delta_vv = post_vv_db - pre_vv_db
        delta_vh = post_vh_db - pre_vh_db
        delta_ratio = (post_vv_db - post_vh_db) - (pre_vv_db - pre_vh_db)

        delta_vh_p95 = float(np.nanpercentile(delta_vh[valid], 95))
        post_vh_median = float(np.nanmedian(post_vh_db[valid]))
        event_detected = (
            delta_vh_p95 > RADAR_EVENT_CHANGE_THRESHOLD_DB
            and post_vh_median > RADAR_EVENT_POSTERIOR_THRESHOLD_DB
        )

        smooth_delta_vh = _median_filter(
            np.where(valid, delta_vh, np.nan).astype(np.float32), 3
        )
        classes = np.zeros(valid.shape, dtype=np.uint8)
        if event_detected:
            classes[valid & (smooth_delta_vh > RADAR_PIXEL_THRESHOLDS_DB[0])] = 1
            classes[valid & (smooth_delta_vh > RADAR_PIXEL_THRESHOLDS_DB[1])] = 2
            classes[valid & (smooth_delta_vh > RADAR_PIXEL_THRESHOLDS_DB[2])] = 3
            _remove_small_regions(classes, valid, minimum_pixels=4)

        valid_count = max(int(valid.sum()), 1)
        counts = {
            level: int(np.sum(valid & (classes == level)))
            for level in range(4)
        }
        shares = {level: counts[level] / valid_count for level in range(4)}
        area = float(identity["area_referencia_ha"])
        features = np.asarray(
            [
                float(np.nanmedian(delta_vh[valid])),
                delta_vh_p95,
                float(np.nanmedian(delta_vv[valid])),
                float(np.nanpercentile(delta_vv[valid], 95)),
                float(np.nanmedian(delta_ratio[valid])),
                post_vh_median,
                float(np.nanmedian(post_vv_db[valid])),
            ]
        )
        row: dict[str, Any] = {
            **identity,
            "pixeles_validos_pct": round(float(valid_pct), 1),
            "resultado_satelital": _radar_decision(delta_vh_p95),
            "patron_respuesta_radar": _radar_pattern(features, event_detected),
            "cambio_total_ha": round(area * sum(shares[level] for level in (1, 2, 3)), 2),
            "cambio_total_pct": round(100 * sum(shares[level] for level in (1, 2, 3)), 1),
            "cambio_moderado_fuerte_ha": round(area * (shares[2] + shares[3]), 2),
            "cambio_moderado_fuerte_pct": round(100 * (shares[2] + shares[3]), 1),
            "estado": "Calculado",
            "motivo": "",
            "_overlay_png": _class_png(classes, valid),
            "_overlay_bbox": bbox,
            "_class_data": np.where(valid, classes, 255).astype(np.uint8),
        }
        for level in range(4):
            row[f"clase_{level}_ha"] = round(area * shares[level], 2)
            row[f"clase_{level}_pct"] = round(100 * shares[level], 1)
        return row

    worker_rows = _run_by_feature(geojson, worker)
    overlays = tuple(
        {
            "pedido": row.get("pedido", ""),
            "bbox": row["_overlay_bbox"],
            "image_png": row["_overlay_png"],
            "class_data": row["_class_data"],
        }
        for row in worker_rows
        if row.get("_overlay_png") is not None
    )
    rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in worker_rows
    ]
    calculated = [row for row in rows if row.get("estado") == "Calculado"]
    total_area = sum(float(row.get("area_referencia_ha") or 0) for row in rows)
    changed_area = sum(float(row.get("cambio_total_ha") or 0) for row in calculated)
    moderate_strong_area = sum(
        float(row.get("cambio_moderado_fuerte_ha") or 0) for row in calculated
    )
    return SensorMetricResult(
        sensor="Sentinel-1 RTC",
        summary={
            "area_referencia_ha": round(total_area, 2),
            "cambio_total_ha": round(changed_area, 2),
            "cambio_moderado_fuerte_ha": round(moderate_strong_area, 2),
            "lotes_calculados": len(calculated),
            "lotes_cambio_claro": sum(
                row.get("resultado_satelital") == "Cambio claro" for row in calculated
            ),
            "lotes_indeterminados": sum(
                row.get("resultado_satelital") == "Indeterminado" for row in calculated
            ),
            "lotes_sin_cambio_claro": sum(
                row.get("resultado_satelital") == "Sin cambio claro" for row in calculated
            ),
            "lotes_no_evaluables": len(rows) - len(calculated),
            "metodo": (
                "Sentinel-1 RTC · cambio VH positivo · filtro mediana 3×3 · "
                "calibración específica del evento"
            ),
        },
        lot_summaries=rows,
        overlays=overlays,
    )
