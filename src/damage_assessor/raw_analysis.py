from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np

from .catalog import SceneAcquisition, SceneCandidate
from .geometry import feature_area_ha_approx, geojson_bbox
from .imagery import fetch_raw_array, native_grid_size


@dataclass(frozen=True)
class SensorMetricResult:
    sensor: str
    summary: dict[str, Any]
    lot_summaries: list[dict[str, Any]]


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
    return np.nanmedian(windows, axis=(-2, -1)).astype(np.float32)


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
    """Calculate S1 change from linear-power RTC VV/VH on a common per-lot grid."""
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
                "area_perturbada_ha_preliminar": 0.0,
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
        valid = mask & np.isfinite(pre_vv) & np.isfinite(pre_vh)
        valid &= np.isfinite(post_vv) & np.isfinite(post_vh)
        valid_pct = 100 * valid.sum() / max(mask.sum(), 1)
        if valid_pct < 90:
            return {
                **identity,
                "pixeles_validos_pct": round(float(valid_pct), 1),
                "estado": "No calculado",
                "motivo": "La cobertura radar válida conjunta es menor a 90% del lote.",
            }

        pre_vv_db = 10 * np.log10(np.maximum(pre_vv, 1e-8))
        pre_vh_db = 10 * np.log10(np.maximum(pre_vh, 1e-8))
        post_vv_db = 10 * np.log10(np.maximum(post_vv, 1e-8))
        post_vh_db = 10 * np.log10(np.maximum(post_vh, 1e-8))
        delta_vv = post_vv_db - pre_vv_db
        delta_vh = post_vh_db - pre_vh_db
        pre_rvi = 4 * pre_vh / np.maximum(pre_vv + pre_vh, 1e-8)
        post_rvi = 4 * post_vh / np.maximum(post_vv + post_vh, 1e-8)
        delta_rvi = post_rvi - pre_rvi
        affected = valid & (np.abs(delta_vh) >= 2.0)
        affected &= (np.abs(delta_vv) >= 1.5) | (np.abs(delta_rvi) >= 0.12)
        affected_share = affected.sum() / max(valid.sum(), 1)
        magnitude = (
            np.nanmedian(
                np.clip(np.abs(delta_vh[valid]) / 4.0, 0, 1)
                + np.clip(np.abs(delta_vv[valid]) / 4.0, 0, 1)
            )
            / 2
        )
        score = round(10 * np.clip(0.55 * magnitude + 0.45 * affected_share, 0, 1), 1)
        area = float(identity["area_referencia_ha"])
        return {
            **identity,
            "pixeles_validos_pct": round(float(valid_pct), 1),
            "delta_vv_db_mediana": round(float(np.nanmedian(delta_vv[valid])), 2),
            "delta_vh_db_mediana": round(float(np.nanmedian(delta_vh[valid])), 2),
            "delta_rvi_mediana": round(float(np.nanmedian(delta_rvi[valid])), 3),
            "area_perturbada_ha_preliminar": round(area * float(affected_share), 2),
            "superficie_perturbada_pct": round(100 * float(affected_share), 1),
            "severidad_satelital_0_10_preliminar": score,
            "clase_preliminar": _severity_class(score),
            "estado": "Calculado",
            "motivo": "",
        }

    rows = _run_by_feature(geojson, worker)
    calculated = [row for row in rows if row.get("estado") == "Calculado"]
    total_area = sum(float(row.get("area_referencia_ha") or 0) for row in rows)
    changed_area = sum(float(row.get("area_perturbada_ha_preliminar") or 0) for row in calculated)
    return SensorMetricResult(
        sensor="Sentinel-1 RTC",
        summary={
            "area_referencia_ha": round(total_area, 2),
            "area_perturbada_ha_preliminar": round(changed_area, 2),
            "lotes_calculados": len(calculated),
            "metodo": "Potencia lineal RTC; filtro mediana 3×3; cambios VV/VH/RVI",
        },
        lot_summaries=rows,
    )
