from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

EARTH_RADIUS_M = 6_371_008.8


class GeometryValidationError(ValueError):
    """Raised when an uploaded GeoJSON is not safe to process."""


@dataclass(frozen=True)
class GeometryProfile:
    feature_count: int
    geometry_types: dict[str, int]
    bbox: tuple[float, float, float, float]
    total_area_ha_approx: float
    properties: tuple[str, ...]
    warnings: tuple[str, ...]


def load_geojson_bytes(payload: bytes, max_bytes: int = 50 * 1024 * 1024) -> dict[str, Any]:
    if not payload:
        raise GeometryValidationError("El archivo está vacío.")
    if len(payload) > max_bytes:
        raise GeometryValidationError("El archivo supera el límite de 50 MB.")
    try:
        data = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GeometryValidationError("El archivo no es un GeoJSON UTF-8 válido.") from exc
    validate_geojson(data)
    return data


def validate_geojson(data: dict[str, Any]) -> None:
    if data.get("type") != "FeatureCollection":
        raise GeometryValidationError("Se requiere un GeoJSON FeatureCollection.")
    features = data.get("features")
    if not isinstance(features, list) or not features:
        raise GeometryValidationError("El GeoJSON debe contener al menos una geometría.")
    if len(features) > 10_000:
        raise GeometryValidationError("El prototipo admite hasta 10.000 lotes por carga.")

    coordinate_count = 0
    for index, feature in enumerate(features, start=1):
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise GeometryValidationError(f"El elemento {index} no es una Feature válida.")
        geometry = feature.get("geometry") or {}
        if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            raise GeometryValidationError(
                f"La geometría {index} es {geometry.get('type')!r}; solo se admiten Polygon/MultiPolygon."
            )
        for ring in iter_rings(geometry):
            if len(ring) < 4:
                raise GeometryValidationError(
                    f"La geometría {index} contiene un anillo incompleto."
                )
            if ring[0][:2] != ring[-1][:2]:
                raise GeometryValidationError(
                    f"La geometría {index} contiene un anillo sin cerrar."
                )
            for point in ring:
                if not isinstance(point, list) or len(point) < 2:
                    raise GeometryValidationError(
                        f"La geometría {index} contiene una coordenada inválida."
                    )
                x, y = point[:2]
                if not all(
                    isinstance(value, (int, float)) and math.isfinite(value) for value in (x, y)
                ):
                    raise GeometryValidationError(
                        f"La geometría {index} contiene coordenadas no numéricas."
                    )
                coordinate_count += 1
    if coordinate_count > 2_000_000:
        raise GeometryValidationError(
            "La geometría contiene demasiados vértices para este prototipo."
        )


def iter_polygons(geometry: dict[str, Any]) -> Iterator[list[list[list[float]]]]:
    if geometry["type"] == "Polygon":
        yield geometry["coordinates"]
    else:
        yield from geometry["coordinates"]


def iter_rings(geometry: dict[str, Any]) -> Iterator[list[list[float]]]:
    for polygon in iter_polygons(geometry):
        yield from polygon


def iter_points(data: dict[str, Any]) -> Iterator[tuple[float, float]]:
    for feature in data["features"]:
        for ring in iter_rings(feature["geometry"]):
            for point in ring:
                yield float(point[0]), float(point[1])


def geojson_bbox(data: dict[str, Any]) -> tuple[float, float, float, float]:
    points = list(iter_points(data))
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _ring_area_m2(ring: list[list[float]]) -> float:
    latitude = sum(float(point[1]) for point in ring) / len(ring)
    cos_latitude = math.cos(math.radians(latitude))
    xy = [
        (
            math.radians(float(point[0])) * EARTH_RADIUS_M * cos_latitude,
            math.radians(float(point[1])) * EARTH_RADIUS_M,
        )
        for point in ring
    ]
    doubled_area = sum(
        x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(xy, xy[1:] + xy[:1], strict=True)
    )
    return abs(doubled_area) / 2


def feature_area_ha_approx(feature: dict[str, Any]) -> float:
    area_m2 = 0.0
    for polygon in iter_polygons(feature["geometry"]):
        for ring_index, ring in enumerate(polygon):
            ring_area = _ring_area_m2(ring)
            area_m2 += ring_area if ring_index == 0 else -ring_area
    return max(area_m2 / 10_000, 0.0)


def field_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(data["features"], start=1):
        properties = dict(feature.get("properties") or {})
        properties["feature_index"] = index
        properties["geometry_type"] = feature["geometry"]["type"]
        properties["area_ha_approx"] = round(feature_area_ha_approx(feature), 2)
        rows.append(properties)
    return rows


def categorical_property_names(data: dict[str, Any]) -> list[str]:
    """Return properties that can be used to group or select features."""
    names = sorted(
        {key for feature in data["features"] for key in (feature.get("properties") or {})}
    )
    return [
        name
        for name in names
        if any(
            (feature.get("properties") or {}).get(name) not in (None, "")
            for feature in data["features"]
        )
    ]


def property_values(data: dict[str, Any], property_name: str) -> list[str]:
    return sorted(
        {
            str((feature.get("properties") or {}).get(property_name))
            for feature in data["features"]
            if (feature.get("properties") or {}).get(property_name) not in (None, "")
        }
    )


def filter_features(
    data: dict[str, Any], property_name: str, selected_values: Iterable[str]
) -> dict[str, Any]:
    selected = {str(value) for value in selected_values}
    features = [
        feature
        for feature in data["features"]
        if str((feature.get("properties") or {}).get(property_name)) in selected
    ]
    if not features:
        raise GeometryValidationError("La selección no contiene lotes.")
    output = {"type": "FeatureCollection", "features": features}
    if "crs" in data:
        output["crs"] = data["crs"]
    return output


def profile_geojson(data: dict[str, Any]) -> GeometryProfile:
    validate_geojson(data)
    bbox = geojson_bbox(data)
    warnings: list[str] = []
    if not all(-180 <= x <= 180 and -90 <= y <= 90 for x, y in iter_points(data)):
        warnings.append("Las coordenadas no parecen longitud/latitud; se requiere declarar CRS.")

    geometry_types = Counter(feature["geometry"]["type"] for feature in data["features"])
    properties = sorted(
        {key for feature in data["features"] for key in (feature.get("properties") or {})}
    )
    total_area = sum(feature_area_ha_approx(feature) for feature in data["features"])
    return GeometryProfile(
        feature_count=len(data["features"]),
        geometry_types=dict(geometry_types),
        bbox=bbox,
        total_area_ha_approx=total_area,
        properties=tuple(properties),
        warnings=tuple(warnings),
    )


def properties_complete(data: dict[str, Any], keys: Iterable[str]) -> dict[str, float]:
    features = data["features"]
    return {
        key: sum(
            (feature.get("properties") or {}).get(key) not in (None, "") for feature in features
        )
        / len(features)
        for key in keys
    }
