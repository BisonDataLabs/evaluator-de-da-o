from __future__ import annotations

import copy
import csv
import io
import json
import math
import re
import unicodedata
import zipfile
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, TiffImagePlugin

from .catalog import PLANETARY_PROVIDER_NAME, SceneAcquisition
from .geometry import geojson_bbox, iter_rings
from .raw_analysis import RADAR_CLASS_COLORS, SensorMetricResult
from .weather import PowerDailySeries

RADAR_CLASS_NAMES = {
    0: "Sin cambio destacado",
    1: "Cambio leve",
    2: "Cambio moderado",
    3: "Cambio fuerte",
}

RESULT_COLUMNS = [
    ("pedido", "Lote", "text", 25),
    ("establecimiento", "Campo", "text", 23),
    ("area_referencia_ha", "Superficie analizada (ha)", "number", 22),
    ("dano_observado_campo", "Daño observado en campo (0–10)", "score", 28),
    ("resultado_satelital", "Resultado satelital", "text", 20),
    ("patron_respuesta_radar", "Patrón de respuesta radar", "text", 35),
    ("cambio_total_ha", "Cambio total (ha)", "number", 18),
    ("cambio_total_pct", "Cambio total (%)", "percent", 18),
    ("cambio_moderado_fuerte_ha", "Moderado + fuerte (ha)", "number", 22),
    ("cambio_moderado_fuerte_pct", "Moderado + fuerte (%)", "percent", 22),
    ("clase_0_ha", "Sin cambio destacado (ha)", "number", 25),
    ("clase_0_pct", "Sin cambio destacado (%)", "percent", 25),
    ("clase_1_ha", "Cambio leve (ha)", "number", 18),
    ("clase_1_pct", "Cambio leve (%)", "percent", 18),
    ("clase_2_ha", "Cambio moderado (ha)", "number", 22),
    ("clase_2_pct", "Cambio moderado (%)", "percent", 22),
    ("clase_3_ha", "Cambio fuerte (ha)", "number", 19),
    ("clase_3_pct", "Cambio fuerte (%)", "percent", 19),
    ("estado", "Estado del cálculo", "text", 19),
    ("motivo", "Observación del cálculo", "text", 38),
    ("fecha_anterior_utc", "Fecha anterior (UTC)", "datetime", 21),
    ("fecha_posterior_utc", "Fecha posterior (UTC)", "datetime", 21),
    ("orbita_relativa", "Órbita relativa", "integer", 15),
    ("direccion", "Dirección", "text", 14),
]


def _field_scores(field_observations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        str(row.get("pedido", "")): row.get(
            "escala_dano_0_10", row.get("dano_observado_campo")
        )
        for row in field_observations
    }


def _is_blank_cell(value: Any) -> bool:
    """Treat empty editable cells as blank spreadsheet cells."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return True
    return isinstance(value, (float, np.floating)) and math.isnan(float(value))


def result_table_rows(
    result: SensorMetricResult,
    field_observations: list[dict[str, Any]],
    before: SceneAcquisition,
    after: SceneAcquisition,
) -> list[dict[str, Any]]:
    scores = _field_scores(field_observations)
    rows: list[dict[str, Any]] = []
    for source in result.lot_summaries:
        order = str(source.get("pedido", ""))
        rows.append(
            {
                **source,
                "dano_observado_campo": scores.get(order),
                "fecha_anterior_utc": before.acquired_at,
                "fecha_posterior_utc": after.acquired_at,
                "orbita_relativa": before.relative_orbit,
                "direccion": before.orbit_state,
            }
        )
    return rows


def build_results_workbook(
    result: SensorMetricResult,
    field_observations: list[dict[str, Any]],
    before: SceneAcquisition,
    after: SceneAcquisition,
) -> bytes:
    """Create the standalone Excel table used by both download options."""
    try:
        import xlsxwriter
    except ImportError as exc:  # pragma: no cover - dependency is declared for deployment
        raise RuntimeError("La dependencia XlsxWriter no está instalada.") from exc

    rows = result_table_rows(result, field_observations, before, after)
    output = io.BytesIO()
    workbook = xlsxwriter.Workbook(output, {"in_memory": True})
    workbook.set_properties(
        {
            "title": "Resultados de evaluación de daño agrícola",
            "subject": "Resultados Sentinel-1 por lote",
            "company": "Bison",
        }
    )
    sheet = workbook.add_worksheet("Resultados por lote")
    sheet.hide_gridlines(2)
    sheet.freeze_panes(1, 2)
    header_format = workbook.add_format(
        {
            "bold": True,
            "font_color": "#FFFFFF",
            "bg_color": "#9B3A32",
            "text_wrap": True,
            "valign": "vcenter",
        }
    )
    formats = {
        "text": workbook.add_format({"valign": "top"}),
        "number": workbook.add_format({"num_format": "#,##0.00", "valign": "top"}),
        "score": workbook.add_format({"num_format": "0.0", "valign": "top"}),
        "percent": workbook.add_format({"num_format": "0.0%", "valign": "top"}),
        "integer": workbook.add_format({"num_format": "0", "valign": "top"}),
        "datetime": workbook.add_format(
            {"num_format": "yyyy-mm-dd hh:mm", "valign": "top"}
        ),
    }
    for column_index, (_, label, _, width) in enumerate(RESULT_COLUMNS):
        sheet.write(0, column_index, label, header_format)
        sheet.set_column(column_index, column_index, width)
    sheet.set_row(0, 34)

    for row_index, row in enumerate(rows, start=1):
        for column_index, (key, _, kind, _) in enumerate(RESULT_COLUMNS):
            value = row.get(key)
            if _is_blank_cell(value):
                sheet.write_blank(row_index, column_index, None, formats[kind])
            elif kind == "percent":
                sheet.write_number(row_index, column_index, float(value) / 100, formats[kind])
            elif kind == "datetime":
                moment = value.replace(tzinfo=None) if isinstance(value, datetime) else value
                sheet.write_datetime(row_index, column_index, moment, formats[kind])
            elif kind in {"number", "score", "integer"}:
                sheet.write_number(row_index, column_index, float(value), formats[kind])
            else:
                sheet.write(row_index, column_index, str(value), formats[kind])

    last_row = max(len(rows), 1)
    last_column = len(RESULT_COLUMNS) - 1
    sheet.autofilter(0, 0, last_row, last_column)
    sheet.conditional_format(
        1,
        4,
        last_row,
        4,
        {
            "type": "text",
            "criteria": "containing",
            "value": "Cambio claro",
            "format": workbook.add_format({"bg_color": "#F4CCCC", "font_color": "#7A1F2A"}),
        },
    )
    sheet.conditional_format(
        1,
        4,
        last_row,
        4,
        {
            "type": "text",
            "criteria": "containing",
            "value": "Indeterminado",
            "format": workbook.add_format({"bg_color": "#FFF2CC", "font_color": "#7F6000"}),
        },
    )
    workbook.close()
    return output.getvalue()


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"No se puede serializar {type(value).__name__}.")


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def _safe_filename(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized).strip("._")
    return cleaned or "lote"


def _result_geojson(
    geojson: dict[str, Any],
    result: SensorMetricResult,
    field_observations: list[dict[str, Any]],
) -> dict[str, Any]:
    output = copy.deepcopy(geojson)
    scores = _field_scores(field_observations)
    for index, feature in enumerate(output["features"]):
        row = result.lot_summaries[index] if index < len(result.lot_summaries) else {}
        properties = feature.setdefault("properties", {})
        order = str(row.get("pedido", properties.get("pedido", "")))
        properties.update(
            {
                key: value
                for key, value in row.items()
                if not key.startswith("_") and value is not None
            }
        )
        properties["dano_observado_campo_0_10"] = scores.get(order)
    return output


def _overlay_class_data(overlay: dict[str, Any]) -> np.ndarray:
    supplied = overlay.get("class_data")
    if supplied is not None:
        return np.asarray(supplied, dtype=np.uint8)
    rgba = np.asarray(Image.open(io.BytesIO(overlay["image_png"])).convert("RGBA"))
    output = np.full(rgba.shape[:2], 255, dtype=np.uint8)
    for level, color in RADAR_CLASS_COLORS.items():
        matches = np.all(rgba[..., :3] == np.asarray(color[:3], dtype=np.uint8), axis=2)
        output[matches & (rgba[..., 3] > 0)] = level
    return output


def _geotiff_bytes(values: np.ndarray, bbox: tuple[float, float, float, float]) -> bytes:
    height, width = values.shape
    min_x, min_y, max_x, max_y = bbox
    pixel_width = (max_x - min_x) / width
    pixel_height = (max_y - min_y) / height
    tags = TiffImagePlugin.ImageFileDirectory_v2()
    tags[33550] = (pixel_width, pixel_height, 0.0)
    tags[33922] = (0.0, 0.0, 0.0, min_x, max_y, 0.0)
    tags[34735] = (
        1,
        1,
        0,
        3,
        1024,
        0,
        1,
        2,
        1025,
        0,
        1,
        1,
        2048,
        0,
        1,
        4326,
    )
    tags[42113] = "255"
    output = io.BytesIO()
    Image.fromarray(values, mode="L").save(
        output,
        format="TIFF",
        compression="tiff_deflate",
        tiffinfo=tags,
    )
    return output.getvalue()


def _change_zones(
    overlays: tuple[dict[str, Any], ...],
    lot_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows_by_order = {str(row.get("pedido", "")): row for row in lot_rows}
    features: list[dict[str, Any]] = []
    zone_id = 0
    for overlay in overlays:
        values = _overlay_class_data(overlay)
        height, width = values.shape
        min_x, min_y, max_x, max_y = tuple(overlay["bbox"])
        x_step = (max_x - min_x) / width
        y_step = (max_y - min_y) / height
        order = str(overlay.get("pedido", ""))
        lot_row = rows_by_order.get(order, {})
        valid_count = int(np.sum(values != 255))
        area_per_pixel = float(lot_row.get("area_referencia_ha") or 0) / max(valid_count, 1)
        for row_index in range(height):
            column = 0
            while column < width:
                level = int(values[row_index, column])
                if level not in (1, 2, 3):
                    column += 1
                    continue
                end = column + 1
                while end < width and int(values[row_index, end]) == level:
                    end += 1
                zone_id += 1
                left = min_x + column * x_step
                right = min_x + end * x_step
                top = max_y - row_index * y_step
                bottom = max_y - (row_index + 1) * y_step
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [left, bottom],
                                    [right, bottom],
                                    [right, top],
                                    [left, top],
                                    [left, bottom],
                                ]
                            ],
                        },
                        "properties": {
                            "zona_id": zone_id,
                            "pedido": order,
                            "establecimiento": lot_row.get("establecimiento", ""),
                            "clase": level,
                            "clase_nombre": RADAR_CLASS_NAMES[level],
                            "superficie_ha": round(area_per_pixel * (end - column), 4),
                        },
                    }
                )
                column = end
    return {"type": "FeatureCollection", "features": features}


def _weather_csv(weather: PowerDailySeries | None) -> bytes:
    output = io.StringIO()
    fieldnames = [
        "fecha",
        "precipitacion_mm",
        "humedad_superficial_0_1",
        "latitud",
        "longitud",
        "estandar_horario",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    if weather is not None:
        for observation in weather.observations:
            writer.writerow(
                {
                    "fecha": observation.day.isoformat(),
                    "precipitacion_mm": observation.precipitation_mm,
                    "humedad_superficial_0_1": observation.surface_soil_wetness,
                    "latitud": weather.latitude,
                    "longitud": weather.longitude,
                    "estandar_horario": weather.time_standard,
                }
            )
    return output.getvalue().encode("utf-8-sig")


def _acquisition_metadata(acquisition: SceneAcquisition) -> dict[str, Any]:
    return {
        "fecha_utc": acquisition.acquired_at.isoformat(),
        "orbita_relativa": acquisition.relative_orbit,
        "direccion": acquisition.orbit_state,
        "cantidad_de_coberturas": acquisition.tile_count,
        "escenas": [scene.to_row() for scene in acquisition.scenes],
    }


def _image_with_boundaries(
    payload: bytes,
    bbox: tuple[float, float, float, float],
    geojson: dict[str, Any],
) -> bytes:
    image = Image.open(io.BytesIO(payload)).convert("RGBA")
    draw = ImageDraw.Draw(image)
    min_x, min_y, max_x, max_y = bbox

    def point_to_pixel(point: list[float]) -> tuple[float, float]:
        x = (float(point[0]) - min_x) / (max_x - min_x) * image.width
        y = (max_y - float(point[1])) / (max_y - min_y) * image.height
        return x, y

    for feature in geojson["features"]:
        for ring in iter_rings(feature["geometry"]):
            pixels = [point_to_pixel(point) for point in ring]
            draw.line(pixels, fill=(0, 0, 0, 220), width=5, joint="curve")
            draw.line(pixels, fill=(255, 255, 255, 255), width=2, joint="curve")
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _comparison_png(before: bytes, after: bytes) -> bytes:
    left = Image.open(io.BytesIO(before)).convert("RGB")
    right = Image.open(io.BytesIO(after)).convert("RGB")
    height = max(left.height, right.height)
    header = 34
    canvas = Image.new("RGB", (left.width + right.width, height + header), "white")
    canvas.paste(left, (0, header))
    canvas.paste(right, (left.width, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), "Anterior", fill="#17313B")
    draw.text((left.width + 12, 10), "Posterior", fill="#17313B")
    draw.line((left.width, 0, left.width, canvas.height), fill="#17313B", width=2)
    output = io.BytesIO()
    canvas.save(output, format="PNG", optimize=True)
    return output.getvalue()


def build_download_package(
    geojson: dict[str, Any],
    event_date: date,
    result: SensorMetricResult,
    field_observations: list[dict[str, Any]],
    before: SceneAcquisition,
    after: SceneAcquisition,
    weather: PowerDailySeries | None,
    export_images: dict[str, bytes],
    image_bbox: tuple[float, float, float, float] | None,
) -> bytes:
    """Build the non-HTML technical package requested for the app."""
    workbook = build_results_workbook(result, field_observations, before, after)
    result_geojson = _result_geojson(geojson, result, field_observations)
    zones = _change_zones(result.overlays, result.lot_summaries)
    metadata = {
        "generado_utc": datetime.now(UTC).isoformat(),
        "evento": "granizo",
        "fecha_evento": event_date.isoformat(),
        "proveedor_imagenes": PLANETARY_PROVIDER_NAME,
        "producto": "Sentinel-1 RTC",
        "adquisicion_anterior": _acquisition_metadata(before),
        "adquisicion_posterior": _acquisition_metadata(after),
        "seleccion": {
            "cantidad_lotes": len(geojson["features"]),
            "bbox_epsg_4326": geojson_bbox(geojson),
        },
        "imagenes": sorted(export_images),
        "clasificacion": {
            "crs": "EPSG:4326",
            "nodata": 255,
            "valores": RADAR_CLASS_NAMES,
        },
        "contexto_climatico": {
            "fuente": "NASA POWER",
            "disponible": weather is not None,
        },
    }

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("resultados_por_lote.xlsx", workbook)
        archive.writestr("lotes_resultados.geojson", _json_bytes(result_geojson))
        archive.writestr("zonas_de_cambio.geojson", _json_bytes(zones))
        archive.writestr("contexto_climatico.csv", _weather_csv(weather))
        archive.writestr("metadatos.json", _json_bytes(metadata))

        for index, overlay in enumerate(result.overlays, start=1):
            filename = f"{index:03d}_{_safe_filename(str(overlay.get('pedido', 'lote')))}"
            class_data = _overlay_class_data(overlay)
            archive.writestr(
                f"clasificacion_intralote/{filename}.tif",
                _geotiff_bytes(class_data, tuple(overlay["bbox"])),
            )
            archive.writestr(
                f"imagenes/clasificacion_intralote/{filename}.png",
                overlay["image_png"],
            )

        prepared_images: dict[str, bytes] = {}
        for filename, payload in export_images.items():
            prepared = (
                _image_with_boundaries(payload, image_bbox, geojson)
                if image_bbox is not None
                else payload
            )
            prepared_images[filename] = prepared
            archive.writestr(f"imagenes/{filename}", prepared)
        if {"sentinel1_anterior.png", "sentinel1_posterior.png"} <= prepared_images.keys():
            archive.writestr(
                "imagenes/comparacion_antes_despues.png",
                _comparison_png(
                    prepared_images["sentinel1_anterior.png"],
                    prepared_images["sentinel1_posterior.png"],
                ),
            )
    return output.getvalue()
