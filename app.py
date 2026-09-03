from __future__ import annotations

import hashlib
import io
import math
import os
import sys
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
from PIL import Image

from damage_assessor.acquisition_plan import next_sentinel1_pass
from damage_assessor.catalog import (
    DEFAULT_OPTICAL_COLLECTION,
    DEFAULT_RADAR_COLLECTION,
    PLANETARY_PROVIDER_NAME,
    SceneAcquisition,
    ScenePair,
    acquisition_coverage_pct,
    group_scene_acquisitions,
    search_scene_metadata,
    select_optical_pair,
    select_radar_pair,
)
from damage_assessor.geometry import (
    GeometryValidationError,
    categorical_property_names,
    field_rows,
    filter_features,
    geojson_bbox,
    load_geojson_bytes,
    profile_geojson,
    property_values,
)
from damage_assessor.ground_truth import load_ground_truth, rows_for_geojson
from damage_assessor.imagery import (
    OPTICAL_STANDARD_LAYER,
    RADAR_STANDARD_LAYER,
    RADAR_VISUAL_LAYERS,
    SENTINEL2_INDEX_LAYERS,
    fetch_expression_image,
    fetch_scene_image,
    padded_bbox,
    resolve_tile_layer_url,
)
from damage_assessor.mapping import comparison_tile_swipe_map, radar_results_map, satellite_tile_map
from damage_assessor.raw_analysis import (
    SensorMetricResult,
    calculate_radar_metrics,
)
from damage_assessor.reporting import build_download_package, build_results_workbook
from damage_assessor.severity import severity_class
from damage_assessor.weather import PowerDailySeries, fetch_power_daily

st.set_page_config(
    page_title="Evaluador de daño agrícola",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

LAYER_ORDER = [
    RADAR_STANDARD_LAYER,
    *RADAR_VISUAL_LAYERS,
    OPTICAL_STANDARD_LAYER,
    "Sentinel-2 · NDVI",
    "Sentinel-2 · NDRE",
    "Sentinel-2 · NDMI",
]

LAYER_GUIDANCE = {
    RADAR_STANDARD_LAYER: """
### Composición VV/VH

- **Azul/cian:** respuesta baja; agua o superficies lisas.
- **Verde:** respuesta intermedia, frecuente en vegetación y superficies rurales.
- **Naranja:** respuesta VV elevada; superficie rugosa o húmeda.
- **Amarillo:** respuesta muy alta en VV y VH; predominante en ciudades y construcciones.

Aporta contexto visual. Ningún color debe leerse como daño.
""",
    "Sentinel-1 · VV alto contraste": """
### VV alto contraste

- **Negro:** respuesta VV muy baja; agua o superficies lisas.
- **Gris:** respuesta frecuente en suelos y cultivos.
- **Blanco:** respuesta VV elevada; superficies rugosas o húmedas, construcciones y montes.

Un aumento de brillo posterior puede relacionarse con mayor humedad o rugosidad. Una reducción
indica menor respuesta superficial.
""",
    "Sentinel-1 · VH alto contraste": """
### VH alto contraste

- **Negro:** escasa vegetación o poca estructura.
- **Gris:** cobertura vegetal presente.
- **Blanco:** estructuras densas.

Una reducción de brillo posterior puede ser compatible con pérdida o aplanamiento de la estructura
vegetal. Un aumento también puede responder a mayor humedad.
""",
    "Sentinel-1 · Cociente VV/VH": """
### Cociente VV/VH

- **Violeta/azul:** VH tiene mayor peso relativo; respuesta asociada con vegetación o estructuras
  que dispersan la señal.
- **Cian:** combinación de respuesta vegetal y superficial.
- **Verde/amarillo:** VV predomina sobre VH; frecuente en suelo expuesto, superficies ordenadas y
  construcciones.

Un cambio de **azul/cian a verde/amarillo** indica una reducción relativa de VH y puede ser compatible
con pérdida o aplanamiento de la estructura vegetal. El cambio inverso puede estar asociado con mayor
humedad o mayor dispersión de la vegetación.

Los puntos aislados corresponden principalmente al moteado propio del radar.
""",
    "Sentinel-1 · RVI": """
### RVI

- **Rojo:** vegetación ausente, escasa o con poca estructura.
- **Amarillo:** cobertura y estructura vegetal intermedias.
- **Verde:** vegetación con mayor cobertura y complejidad.

Una disminución de verde hacia amarillo o rojo después del evento puede ser compatible con pérdida
o aplanamiento del cultivo.

La escena completa permanece visible como contexto. Esta interpretación corresponde únicamente al
interior de los lotes.
""",
    OPTICAL_STANDARD_LAYER: """
### RGB

Representa una vista similar a la percepción humana.
""",
    "Sentinel-2 · NDVI": """
### NDVI

- **Rojo:** agua, sombra o ausencia de vegetación.
- **Amarillo:** suelo, rastrojo o cobertura vegetal escasa.
- **Verde claro:** vegetación presente.
- **Verde oscuro:** vegetación densa y activa.

Representa cobertura y actividad vegetal. Una disminución entre fechas puede indicar pérdida de
cobertura o vigor.
""",
    "Sentinel-2 · NDRE": """
### NDRE

- **Rojo:** ausencia de vegetación o contenido muy bajo de clorofila.
- **Amarillo:** cobertura escasa o baja actividad de clorofila.
- **Verde claro:** cultivo con actividad intermedia.
- **Verde oscuro:** dosel denso y mayor contenido de clorofila.

Representa la actividad del dosel, especialmente en cultivos desarrollados. Una disminución puede
indicar pérdida de estructura o clorofila.
""",
    "Sentinel-2 · NDMI": """
### NDMI

- **Rojo/naranja:** suelo, vegetación seca o bajo contenido de agua.
- **Amarillo:** humedad vegetal reducida.
- **Celeste:** vegetación con contenido de agua.
- **Azul:** vegetación muy húmeda o agua superficial.

Representa el contenido relativo de agua. La lluvia reciente puede modificar notablemente esta
visualización.
""",
}

FALLBACK_SEARCH_DAYS = 30
MAX_CATALOG_ITEMS = 120
RADAR_METRICS_VERSION = "s1-intralote-20260903-v1"


@st.cache_data(show_spinner=False)
def parse_geometry(payload: bytes) -> dict[str, Any]:
    return load_geojson_bytes(payload)


@st.cache_data(show_spinner=False, ttl=21_600)
def planned_pass(geojson: dict[str, Any], event_date: date):
    return next_sentinel1_pass(geojson, event_date)


@st.cache_data(show_spinner=False, ttl=21_600)
def catalog_scenes(
    geojson: dict[str, Any],
    event_date: date,
    days_before: int,
    days_after: int,
    collection: str,
) -> list[Any]:
    return search_scene_metadata(
        geojson,
        event_date,
        days_before,
        days_after,
        collection,
        max_items=MAX_CATALOG_ITEMS,
    )


def _search_with_explicit_fallback(
    geojson: dict[str, Any],
    event_date: date,
    days_before: int,
    days_after: int,
    collection: str,
    pair_selector: Callable[[list[Any]], ScenePair | None],
) -> tuple[list[Any], int, int, bool]:
    """Search the requested window and widen once only when a complete pair is absent."""
    scenes = catalog_scenes(geojson, event_date, days_before, days_after, collection)
    pair = pair_selector(scenes)
    if pair is not None and pair.comparable:
        return scenes, days_before, days_after, False

    fallback_before = max(days_before, FALLBACK_SEARCH_DAYS)
    fallback_after = max(days_after, FALLBACK_SEARCH_DAYS)
    if (fallback_before, fallback_after) == (days_before, days_after):
        return scenes, days_before, days_after, False
    expanded = catalog_scenes(
        geojson,
        event_date,
        fallback_before,
        fallback_after,
        collection,
    )
    return expanded, fallback_before, fallback_after, True


@st.cache_data(show_spinner=False, ttl=21_600)
def power_weather(
    latitude: float,
    longitude: float,
    start: date,
    end: date,
) -> PowerDailySeries:
    return fetch_power_daily(latitude, longitude, start, end)


@st.cache_data(show_spinner=False, ttl=21_600)
def radar_metrics(
    before: SceneAcquisition,
    after: SceneAcquisition,
    geojson: dict[str, Any],
    algorithm_version: str,
) -> SensorMetricResult:
    # The explicit version is part of Streamlit's cache key. Without it, a
    # result produced by an earlier methodology can survive a code reload.
    _ = algorithm_version
    return calculate_radar_metrics(before, after, geojson)


@st.cache_data(show_spinner=False, ttl=21_600)
def rendered_scene(scene: Any, bbox: tuple[float, float, float, float]) -> bytes:
    return fetch_scene_image(scene, bbox)


@st.cache_data(show_spinner=False, ttl=21_600)
def rendered_acquisition_mosaic(
    acquisition: SceneAcquisition,
    bbox: tuple[float, float, float, float],
) -> bytes:
    """Render adjacent products on one transparent canvas for the download package."""
    layers = [Image.open(io.BytesIO(rendered_scene(scene, bbox))).convert("RGBA") for scene in acquisition.scenes]
    if not layers:
        raise ValueError("La adquisición no contiene imágenes para exportar.")
    canvas = Image.new("RGBA", layers[0].size, (0, 0, 0, 0))
    for layer in layers:
        if layer.size != canvas.size:
            layer = layer.resize(canvas.size)
        canvas.alpha_composite(layer)
    output = io.BytesIO()
    canvas.save(output, format="PNG", optimize=True)
    return output.getvalue()


@st.cache_data(show_spinner=False, ttl=21_600)
def resolved_scene_tile(
    scene: Any,
    assets: tuple[str, ...] | None,
    expression: str | None,
    rescale: str | None,
    colormap_name: str | None,
) -> str:
    return resolve_tile_layer_url(
        scene,
        assets=assets,
        expression=expression,
        rescale=rescale,
        colormap_name=colormap_name,
    )


@st.cache_data(show_spinner=False, ttl=21_600)
def rendered_index(
    scene: Any,
    bbox: tuple[float, float, float, float],
    assets: tuple[str, ...],
    expression: str,
    rescale: str,
    colormap_name: str,
) -> bytes:
    return fetch_expression_image(
        scene,
        bbox,
        assets=assets,
        expression=expression,
        rescale=rescale,
        colormap_name=colormap_name,
    )


def _render_standard_pair(
    pair: ScenePair, bbox: tuple[float, float, float, float]
) -> tuple[bytes, bytes]:
    return rendered_scene(pair.before, bbox), rendered_scene(pair.after, bbox)


def _render_index_pair(
    pair: ScenePair,
    bbox: tuple[float, float, float, float],
    config: dict[str, object],
) -> tuple[bytes, bytes]:
    arguments = {
        "assets": config["assets"],
        "expression": config["expression"],
        "rescale": config["rescale"],
        "colormap_name": config["colormap_name"],
    }
    return (
        rendered_index(pair.before, bbox, **arguments),
        rendered_index(pair.after, bbox, **arguments),
    )


def _render_selected_layer(
    layer_name: str,
    pair: ScenePair,
    bbox: tuple[float, float, float, float],
) -> tuple[bytes, bytes]:
    if layer_name in {RADAR_STANDARD_LAYER, OPTICAL_STANDARD_LAYER}:
        return _render_standard_pair(pair, bbox)
    config = RADAR_VISUAL_LAYERS.get(layer_name) or SENTINEL2_INDEX_LAYERS.get(layer_name)
    if config is None:
        raise ValueError(f"No existe una configuración para {layer_name}.")
    return _render_index_pair(pair, bbox, config)


def _preferred_property(properties: list[str], preferred: tuple[str, ...]) -> str:
    return next((name for name in preferred if name in properties), properties[0])


def _ordered_layer_names(images: dict[str, Any]) -> list[str]:
    return [name for name in LAYER_ORDER if name in images] + sorted(
        name for name in images if name not in LAYER_ORDER
    )


def _acquisition_label(
    acquisition: SceneAcquisition,
    geojson: dict[str, Any],
    event_date: date,
    preferred_days: int,
) -> str:
    delta = (acquisition.acquired_date - event_date).days
    distance = abs(delta)
    temporal = f"{distance} día(s) {'después' if delta >= 0 else 'antes'}"
    if distance > preferred_days:
        temporal += f" · fuera de {preferred_days} días"
    details = [
        f"{acquisition.acquired_at:%d/%m/%Y %H:%M UTC}",
        temporal,
        f"{acquisition.tile_count} cobertura(s)",
        f"{acquisition_coverage_pct(acquisition, geojson):.0f}% del área",
    ]
    if acquisition.relative_orbit is not None:
        details.append(f"órbita {acquisition.relative_orbit}")
    if acquisition.cloud_cover is not None:
        details.append(f"nubes de escena ≤{acquisition.cloud_cover:.1f}%")
    return " · ".join(details)


def _default_acquisition_index(
    acquisitions: list[SceneAcquisition], preferred_scene: Any | None
) -> int:
    if preferred_scene is None:
        return 0
    return next(
        (
            index
            for index, acquisition in enumerate(acquisitions)
            if any(scene.scene_id == preferred_scene.scene_id for scene in acquisition.scenes)
        ),
        0,
    )


def _default_available_pair(
    acquisitions: list[SceneAcquisition], is_radar: bool, geojson: dict[str, Any]
) -> tuple[SceneAcquisition, SceneAcquisition] | None:
    """Choose a recent pair, prioritizing full coverage and radar compatibility."""
    ordered = sorted(acquisitions, key=lambda acquisition: acquisition.acquired_at)
    if len(ordered) < 2:
        return None
    full_coverage = [
        acquisition
        for acquisition in ordered
        if acquisition_coverage_pct(acquisition, geojson) >= 95
    ]

    def newest_pair(
        candidates: list[SceneAcquisition], require_radar_compatibility: bool
    ) -> tuple[SceneAcquisition, SceneAcquisition] | None:
        for right_index in range(len(candidates) - 1, 0, -1):
            right = candidates[right_index]
            for left_index in range(right_index - 1, -1, -1):
                left = candidates[left_index]
                compatible = (
                    not is_radar
                    or not require_radar_compatibility
                    or (
                        left.relative_orbit is not None
                        and left.relative_orbit == right.relative_orbit
                        and left.orbit_state == right.orbit_state
                    )
                )
                if compatible:
                    return left, right
        return None

    for candidates, require_compatibility in (
        (full_coverage, True),
        (full_coverage, False),
        (ordered, True),
        (ordered, False),
    ):
        pair = newest_pair(candidates, require_compatibility)
        if pair is not None:
            return pair
    return ordered[-2], ordered[-1]


def _acquisition_for_scene(
    acquisitions: list[SceneAcquisition], scene: Any | None
) -> SceneAcquisition | None:
    if scene is None:
        return None
    return next(
        (
            acquisition
            for acquisition in acquisitions
            if any(candidate.scene_id == scene.scene_id for candidate in acquisition.scenes)
        ),
        None,
    )


def _tile_sources(acquisition: SceneAcquisition, layer_name: str) -> list[dict[str, Any]]:
    config = RADAR_VISUAL_LAYERS.get(layer_name) or SENTINEL2_INDEX_LAYERS.get(layer_name)
    sources: list[dict[str, Any]] = []
    for scene in acquisition.scenes:
        url = resolved_scene_tile(
            scene,
            None if config is None else config["assets"],
            None if config is None else config["expression"],
            None if config is None else config["rescale"],
            None if config is None else config["colormap_name"],
        )
        sources.append(
            {
                "url": url,
                "geometry": scene.geometry,
                "bbox": scene.bbox,
                "label": scene.grid_code or scene.scene_id,
            }
        )
    return sources


def _acquisition_rows(
    acquisitions: list[SceneAcquisition], geojson: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        {
            "fecha_utc": acquisition.acquired_at.strftime("%d/%m/%Y %H:%M"),
            "coberturas": acquisition.tile_count,
            "cobertura_seleccion_pct": acquisition_coverage_pct(acquisition, geojson),
            "orbita_relativa": acquisition.relative_orbit,
            "direccion": acquisition.orbit_state,
            "nubes_escena_max_pct": acquisition.cloud_cover,
            "teselas": ", ".join(scene.grid_code or scene.scene_id for scene in acquisition.scenes),
        }
        for acquisition in acquisitions
    ]


def _severity_editor_rows(
    observations: list[dict[str, Any]], geojson: dict[str, Any]
) -> list[dict[str, Any]]:
    by_order = {str(row.get("pedido")): row for row in observations}
    rows: list[dict[str, Any]] = []
    for feature in geojson["features"]:
        properties = feature.get("properties") or {}
        order = str(properties.get("pedido", properties.get("lote", "")))
        observed = by_order.get(order, {})
        rows.append(
            {
                "pedido": order,
                "establecimiento": properties.get("establecimiento", ""),
                "cultivo": observed.get("cultivo", ""),
                "superficie_sembrada_ha": observed.get("superficie_sembrada_ha"),
                "ha_dano": observed.get("ha_dano"),
                "escala_dano_0_10": observed.get("escala_dano_0_10"),
            }
        )
    return rows


def _normalized_severity_rows(edited: Any) -> list[dict[str, Any]]:
    rows = edited.to_dict("records") if hasattr(edited, "to_dict") else list(edited)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        output = dict(row)
        score = output.get("escala_dano_0_10")
        if score is None or (isinstance(score, float) and math.isnan(score)):
            output["escala_dano_0_10"] = ""
            output["clase_severidad"] = "Sin dato"
        else:
            numeric_score = float(score)
            output["escala_dano_0_10"] = numeric_score
            output["clase_severidad"] = severity_class(numeric_score)
        normalized.append(output)
    return normalized


def _result_editor_rows(
    lot_summaries: list[dict[str, Any]],
    field_scores: dict[str, float | None],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in lot_summaries:
        order = str(row.get("pedido", ""))
        rows.append(
            {
                "ver_en_mapa": False,
                "pedido": order,
                "establecimiento": row.get("establecimiento", ""),
                "area_referencia_ha": row.get("area_referencia_ha"),
                "dano_observado_campo": field_scores.get(order),
                "resultado_satelital": row.get("resultado_satelital", "No evaluable"),
                "patron_respuesta_radar": row.get(
                    "patron_respuesta_radar", "Sin patrón asignado"
                ),
                "cambio_total_ha": row.get("cambio_total_ha"),
                "cambio_total_pct": row.get("cambio_total_pct"),
                "cambio_moderado_fuerte_ha": row.get("cambio_moderado_fuerte_ha"),
                "cambio_moderado_fuerte_pct": row.get("cambio_moderado_fuerte_pct"),
                "clase_0_ha": row.get("clase_0_ha"),
                "clase_0_pct": row.get("clase_0_pct"),
                "clase_1_ha": row.get("clase_1_ha"),
                "clase_1_pct": row.get("clase_1_pct"),
                "clase_2_ha": row.get("clase_2_ha"),
                "clase_2_pct": row.get("clase_2_pct"),
                "clase_3_ha": row.get("clase_3_ha"),
                "clase_3_pct": row.get("clase_3_pct"),
                "motivo": row.get("motivo", ""),
            }
        )
    return rows


def _field_rows_from_scores(
    observations: list[dict[str, Any]],
    geojson: dict[str, Any],
    field_scores: dict[str, float | None],
) -> list[dict[str, Any]]:
    rows = _severity_editor_rows(observations, geojson)
    for row in rows:
        row["escala_dano_0_10"] = field_scores.get(str(row.get("pedido", "")))
    return _normalized_severity_rows(rows)


def _analysis_key(
    payload: bytes,
    property_name: str,
    selected_value: str,
    event_date: date,
    days_before: int,
    days_after: int,
    max_cloud: int,
) -> str:
    raw = "|".join(
        [
            hashlib.sha256(payload).hexdigest(),
            property_name,
            selected_value,
            event_date.isoformat(),
            str(days_before),
            str(days_after),
            str(max_cloud),
        ]
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def show_hybrid_map(geojson: dict[str, Any]) -> None:
    try:
        import folium
        from streamlit_folium import st_folium
    except ImportError:
        st.info("El mapa interactivo requiere las dependencias geográficas del proyecto.")
        return

    profile = profile_geojson(geojson)
    min_x, min_y, max_x, max_y = profile.bbox
    map_view = folium.Map(
        location=[(min_y + max_y) / 2, (min_x + max_x) / 2],
        tiles=None,
        control_scale=True,
    )
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Híbrido",
        overlay=False,
        control=True,
    ).add_to(map_view)
    folium.TileLayer(
        tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Boundaries and Places",
        name="Etiquetas",
        overlay=True,
        control=False,
    ).add_to(map_view)
    tooltip_fields = [
        key for key in ("pedido", "establecimiento", "lote") if key in profile.properties
    ]
    tooltip_aliases = {
        "pedido": "Pedido",
        "establecimiento": "Establecimiento",
        "lote": "Lote",
    }
    folium.GeoJson(
        geojson,
        name="Lotes seleccionados",
        style_function=lambda _: {
            "color": "#FFFFFF",
            "weight": 3,
            "fillColor": "#9B3A32",
            "fillOpacity": 0.18,
        },
        highlight_function=lambda _: {"color": "#F4C542", "weight": 4, "fillOpacity": 0.28},
        tooltip=(
            folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=[tooltip_aliases[key] for key in tooltip_fields],
            )
            if tooltip_fields
            else None
        ),
    ).add_to(map_view)
    map_view.fit_bounds([[min_y, min_x], [max_y, max_x]])
    st_folium(map_view, use_container_width=True, height=520, returned_objects=[])


def _pair_message(pair: ScenePair | None, sensor: str) -> None:
    if pair is None:
        st.info(f"No hay un par {sensor} anterior/posterior dentro de la ventana elegida.")
        return
    if pair.comparable:
        st.success(pair.reason)
    else:
        st.warning(pair.reason)


def _weather_markers(
    event_day: date,
    radar_pair: tuple[SceneAcquisition, SceneAcquisition] | None,
) -> list[dict[str, str]]:
    markers = [{"fecha": event_day.isoformat(), "momento": "Evento"}]
    if radar_pair:
        markers.extend(
            [
                {
                    "fecha": radar_pair[0].acquired_date.isoformat(),
                    "momento": "Anterior radar",
                },
                {
                    "fecha": radar_pair[1].acquired_date.isoformat(),
                    "momento": "Posterior radar",
                },
            ]
        )
    return markers


def _weather_chart(
    rows: list[dict[str, Any]],
    markers: list[dict[str, str]],
    field: str,
    title: str,
    color: str,
    mark_type: str,
) -> dict[str, Any]:
    value_layer: dict[str, Any] = {
        "data": {"values": rows},
        "mark": {"type": mark_type, "color": color, "tooltip": True},
        "encoding": {
            "x": {"field": "fecha", "type": "temporal", "title": None},
            "y": {"field": field, "type": "quantitative", "title": title},
            "tooltip": [
                {"field": "fecha", "type": "temporal", "title": "Fecha"},
                {"field": field, "type": "quantitative", "title": title, "format": ".2f"},
            ],
        },
    }
    if mark_type == "line":
        value_layer["mark"].update({"point": True, "strokeWidth": 3})
    return {
        "height": 270,
        "layer": [
            value_layer,
            {
                "data": {"values": markers},
                "mark": {"type": "rule", "color": "#9B3A32", "strokeDash": [5, 4]},
                "encoding": {
                    "x": {"field": "fecha", "type": "temporal"},
                    "tooltip": [
                        {"field": "momento", "type": "nominal", "title": "Referencia"},
                        {"field": "fecha", "type": "temporal", "title": "Fecha"},
                    ],
                },
            },
        ],
        "resolve": {"scale": {"y": "shared"}},
    }


st.title("Evaluador de daño agrícola")

with st.sidebar:
    st.header("Evento y lotes")
    upload = st.file_uploader("Subir lotes en GeoJSON", type=["geojson", "json"])
    event_date = st.date_input("Fecha del evento", value=datetime.now().astimezone().date())
    days_before = st.slider("Ventana anterior (días)", 1, 30, 10)
    days_after = st.slider("Ventana posterior (días)", 1, 30, 10)
    max_cloud = st.slider("Nubosidad máxima de toda la escena (%)", 0, 100, 20)

if upload is None:
    st.info("El análisis comienza con la carga de un GeoJSON de lotes. No hay datos precargados.")
    st.stop()

payload = upload.getvalue()
try:
    uploaded_geojson = parse_geometry(payload)
except GeometryValidationError as exc:
    st.error(str(exc))
    st.stop()

properties = categorical_property_names(uploaded_geojson)
if not properties:
    st.error("El GeoJSON no contiene atributos para identificar grupos o lotes.")
    st.stop()

with st.sidebar:
    st.divider()
    st.subheader("Selección")
    selection_mode = st.radio("Analizar por", ["Grupo de campo", "Lote"], horizontal=True)
    if selection_mode == "Grupo de campo":
        selection_property = st.selectbox(
            "Atributo de grupo",
            properties,
            index=properties.index(
                _preferred_property(properties, ("zona", "grupo", "establecimiento", "campo"))
            ),
        )
    else:
        selection_property = st.selectbox(
            "Atributo de lote",
            properties,
            index=properties.index(_preferred_property(properties, ("pedido", "lote", "lote_id"))),
        )
    selected_value = st.selectbox(
        "Grupo" if selection_mode == "Grupo de campo" else "Lote",
        property_values(uploaded_geojson, selection_property),
    )

active_geojson = filter_features(uploaded_geojson, selection_property, [selected_value])
profile = profile_geojson(active_geojson)
analysis_key = _analysis_key(
    payload,
    selection_property,
    selected_value,
    event_date,
    days_before,
    days_after,
    max_cloud,
)

truth_path = Path(
    os.getenv(
        "GROUND_TRUTH_PATH",
        str(ROOT / "data/private/ground_truth_transcribed_from_image002.csv"),
    )
)
ground_truth = load_ground_truth(truth_path)
active_truth = rows_for_geojson(ground_truth, active_geojson)
uploaded_truth = rows_for_geojson(ground_truth, uploaded_geojson)
upload_hash = hashlib.sha256(payload).hexdigest()[:16]
field_score_key = f"field_scores_{upload_hash}"
if field_score_key not in st.session_state:
    st.session_state[field_score_key] = {
        str(row.get("pedido", "")): float(row["escala_dano_0_10"])
        for row in uploaded_truth
        if row.get("escala_dano_0_10") not in (None, "")
    }

with st.sidebar:
    st.divider()
    run_search = st.button("Buscar y procesar imágenes", type="primary", width="stretch")

if run_search:
    state: dict[str, Any] = {
        "key": analysis_key,
        "radar_metrics_version": RADAR_METRICS_VERSION,
        "images": {},
        "errors": {},
        "metrics": {},
        "metric_pairs": {},
    }
    with st.spinner("Consultando escenas en Planetary Computer…"):
        try:
            radar, radar_search_before, radar_search_after, radar_expanded = (
                _search_with_explicit_fallback(
                    active_geojson,
                    event_date,
                    days_before,
                    days_after,
                    os.getenv("RADAR_COLLECTION", DEFAULT_RADAR_COLLECTION),
                    lambda scenes: select_radar_pair(scenes, event_date),
                )
            )
            optical, optical_search_before, optical_search_after, optical_expanded = (
                _search_with_explicit_fallback(
                    active_geojson,
                    event_date,
                    days_before,
                    days_after,
                    os.getenv("OPTICAL_COLLECTION", DEFAULT_OPTICAL_COLLECTION),
                    lambda scenes: select_optical_pair(scenes, event_date, max_cloud),
                )
            )
            radar_pair = select_radar_pair(radar, event_date)
            optical_pair = select_optical_pair(optical, event_date, max_cloud)
            radar_acquisitions = group_scene_acquisitions(radar, "radar")
            optical_acquisitions = group_scene_acquisitions(optical, "optical")
            state.update(
                {
                    "radar": radar,
                    "optical": optical,
                    "radar_pair": radar_pair,
                    "optical_pair": optical_pair,
                    "radar_acquisitions": radar_acquisitions,
                    "optical_acquisitions": optical_acquisitions,
                    "search_windows": {
                        "radar": (radar_search_before, radar_search_after),
                        "optical": (optical_search_before, optical_search_after),
                    },
                    "expanded_search": {
                        "radar": radar_expanded,
                        "optical": optical_expanded,
                    },
                }
            )
            if radar_pair:
                radar_before = _acquisition_for_scene(radar_acquisitions, radar_pair.before)
                radar_after = _acquisition_for_scene(radar_acquisitions, radar_pair.after)
                if radar_before and radar_after:
                    state["metric_pairs"]["radar"] = (radar_before, radar_after)
                    state["weather_radar_pair"] = (radar_before, radar_after)
            bbox = geojson_bbox(active_geojson)
            state["render_bbox"] = padded_bbox(bbox)
        except Exception as exc:  # noqa: BLE001 - mensaje controlado en límite STAC
            state["fatal_error"] = str(exc)

    radar_pair = state.get("radar_pair")
    if not state.get("fatal_error") and (not radar_pair or not radar_pair.comparable):
        with st.spinner("Consultando la próxima adquisición Sentinel-1…"):
            try:
                state["next_pass"] = planned_pass(active_geojson, event_date)
            except Exception as exc:  # noqa: BLE001 - el plan no debe invalidar escenas disponibles
                state["plan_error"] = str(exc)

    if not state.get("fatal_error"):
        radar_metric_pair = state.get("metric_pairs", {}).get("radar")
        if radar_metric_pair:
            with st.spinner("Procesando las métricas Sentinel-1 del par seleccionado…"):
                try:
                    state["metrics"]["radar"] = radar_metrics(
                        radar_metric_pair[0],
                        radar_metric_pair[1],
                        active_geojson,
                        RADAR_METRICS_VERSION,
                    )
                except Exception as exc:  # noqa: BLE001 - un sensor no invalida el otro
                    state["errors"]["Métricas Sentinel-1"] = str(exc)

            with st.spinner("Preparando las imágenes Sentinel-1 para descarga…"):
                try:
                    export_bbox = geojson_bbox(active_geojson)
                    state["export_image_bbox"] = padded_bbox(export_bbox)
                    state["export_images"] = {
                        "sentinel1_anterior.png": rendered_acquisition_mosaic(
                            radar_metric_pair[0], export_bbox
                        ),
                        "sentinel1_posterior.png": rendered_acquisition_mosaic(
                            radar_metric_pair[1], export_bbox
                        ),
                    }
                except Exception as exc:  # noqa: BLE001 - no invalida los datos calculados
                    state["errors"]["Imágenes de descarga"] = str(exc)

        with st.spinner("Consultando el contexto meteorológico de NASA POWER…"):
            try:
                radar_weather_pair = state.get("weather_radar_pair")
                reference_dates = [event_date]
                if radar_weather_pair:
                    reference_dates.extend(
                        [
                            radar_weather_pair[0].acquired_date,
                            radar_weather_pair[1].acquired_date,
                        ]
                    )
                weather_start = min(reference_dates) - timedelta(days=8)
                weather_end = max(reference_dates) + timedelta(days=1)
                min_x, min_y, max_x, max_y = geojson_bbox(active_geojson)
                state["weather"] = power_weather(
                    (min_y + max_y) / 2,
                    (min_x + max_x) / 2,
                    weather_start,
                    weather_end,
                )
            except Exception as exc:  # noqa: BLE001 - clima no invalida el análisis
                state["weather_error"] = str(exc)
    st.session_state["real_analysis"] = state

state = st.session_state.get("real_analysis")
if state and state.get("key") != analysis_key:
    state = None

compare_tab, metrics_tab, weather_tab, export_tab, overview_tab, scenes_tab = st.tabs(
    [
        "Comparador",
        "Métricas",
        "Contexto climático",
        "Descarga",
        "Resumen",
        "Escenas",
    ],
    default="Comparador",
    key="main_navigation",
    on_change="ignore",
)

with overview_tab:
    metric_columns = st.columns(4)
    metric_columns[0].metric("Lotes seleccionados", profile.feature_count)
    metric_columns[1].metric("Superficie geométrica", f"{profile.total_area_ha_approx:,.1f} ha")
    metric_columns[2].metric("Modo", selection_mode)
    metric_columns[3].metric("Selección", selected_value)
    show_hybrid_map(active_geojson)
    with st.expander("Lotes incluidos"):
        st.dataframe(field_rows(active_geojson), width="stretch", hide_index=True)

with scenes_tab:
    st.subheader("Escenas y disponibilidad")
    st.caption(f"Proveedor: {PLANETARY_PROVIDER_NAME} · catálogo STAC público")
    if not state:
        st.info("La consulta al catálogo se inicia con ‘Buscar y procesar imágenes’.")
    elif state.get("fatal_error"):
        st.error(f"No se pudo consultar Planetary Computer: {state['fatal_error']}")
    else:
        left, right = st.columns(2)
        with left:
            st.markdown("#### Radar · Sentinel-1 RTC")
            _pair_message(state.get("radar_pair"), "radar compatible")
            st.dataframe(
                _acquisition_rows(state.get("radar_acquisitions", []), active_geojson),
                width="stretch",
                hide_index=True,
            )
        with right:
            st.markdown("#### Óptico · Sentinel-2 L2A")
            _pair_message(
                state.get("optical_pair"),
                f"óptico con ≤{max_cloud}% de nubes en toda la escena",
            )
            st.dataframe(
                _acquisition_rows(state.get("optical_acquisitions", []), active_geojson),
                width="stretch",
                hide_index=True,
            )

        radar_pair = state.get("radar_pair")
        next_pass = state.get("next_pass")
        if not radar_pair or not radar_pair.comparable:
            if next_pass:
                pass_row = next_pass.to_row()
                days = pass_row["days_until"]
                st.warning(
                    "No hay imágenes Sentinel-1 posteriores compatibles disponibles. "
                    f"La próxima adquisición IW planificada sobre la selección es "
                    f"{next_pass.starts_at:%d/%m/%Y %H:%M} UTC ({next_pass.satellite}). "
                    f"La disponibilidad se estima en aproximadamente {days} día(s), más la posible "
                    "demora de publicación del producto."
                )
                st.dataframe([pass_row], width="stretch", hide_index=True)
            elif state.get("plan_error"):
                st.warning(
                    "No hay imágenes Sentinel-1 posteriores compatibles y no se pudo consultar el "
                    f"plan oficial: {state['plan_error']}"
                )
            else:
                st.warning(
                    "No hay imágenes Sentinel-1 posteriores compatibles ni una pasada futura publicada "
                    "que intersecte la selección."
                )
        elif next_pass:
            pass_row = next_pass.to_row()
            st.info(
                "Próxima adquisición Sentinel-1 IW planificada sobre la selección: "
                f"{next_pass.starts_at:%d/%m/%Y %H:%M} UTC ({next_pass.satellite}), "
                f"dentro de aproximadamente {pass_row['days_until']} día(s)."
            )
            st.dataframe([pass_row], width="stretch", hide_index=True)
        for sensor, message in state.get("errors", {}).items():
            st.warning(f"{sensor}: {message}")

    with st.expander("Fuentes abiertas y rutas de respaldo"):
        st.dataframe(
            [
                {
                    "fuente": "Microsoft Planetary Computer",
                    "producto": "Sentinel-1 RTC / Sentinel-2 L2A",
                    "rol": "Catálogo y visor activo",
                    "acceso": "Público para catálogo y teselas; RTC crudo sujeto a cuenta",
                },
                {
                    "fuente": "Copernicus Data Space",
                    "producto": "Sentinel-1 GRD / Sentinel-2 L2A",
                    "rol": "Catálogo oficial de respaldo",
                    "acceso": "Cuenta gratuita para descarga y procesamiento",
                },
                {
                    "fuente": "AWS Earth Search",
                    "producto": "Sentinel-2 L2A COG",
                    "rol": "Respaldo óptico directo",
                    "acceso": "Público, sin cuenta",
                },
                {
                    "fuente": "ASF / OPERA",
                    "producto": "OPERA RTC-S1",
                    "rol": "Radar listo para análisis",
                    "acceso": "Earthdata Login gratuito",
                },
                {
                    "fuente": "NASA HLS + Landsat 8/9",
                    "producto": "HLS v2 / Collection 2 L2",
                    "rol": "Respaldo temporal a 30 m",
                    "acceso": "Abierto; Earthdata según ruta",
                },
                {
                    "fuente": "NASA POWER",
                    "producto": "Precipitación y humedad superficial",
                    "rol": "Contexto meteorológico para radar",
                    "acceso": "Público, sin credenciales",
                },
            ],
            width="stretch",
            hide_index=True,
        )

@st.fragment
def render_comparator(
    state: dict[str, Any] | None,
    active_geojson: dict[str, Any],
    event_date: date,
    days_before: int,
    days_after: int,
    max_cloud: int,
    analysis_key: str,
) -> None:
    st.subheader("Comparación antes / después")
    if not state:
        st.info("El comparador aparecerá después de la consulta de escenas.")
    else:
        sensor = st.segmented_control(
            "Sensor",
            ["Sentinel-1 radar", "Sentinel-2 óptico"],
            default="Sentinel-1 radar",
            key="comparison_sensor",
            width="stretch",
        )
        is_radar = sensor == "Sentinel-1 radar"
        source_acquisitions = list(
            state.get("radar_acquisitions" if is_radar else "optical_acquisitions", [])
        )
        all_before_acquisitions = sorted(
            (
                acquisition
                for acquisition in source_acquisitions
                if acquisition.acquired_date < event_date
            ),
            key=lambda acquisition: acquisition.acquired_at,
            reverse=True,
        )
        all_after_acquisitions = sorted(
            (
                acquisition
                for acquisition in source_acquisitions
                if acquisition.acquired_date >= event_date
            ),
            key=lambda acquisition: acquisition.acquired_at,
        )
        preferred_before_acquisitions = [
            acquisition
            for acquisition in all_before_acquisitions
            if (event_date - acquisition.acquired_date).days <= days_before
        ]
        preferred_after_acquisitions = [
            acquisition
            for acquisition in all_after_acquisitions
            if (acquisition.acquired_date - event_date).days <= days_after
        ]
        preferred_pair = state.get("radar_pair" if is_radar else "optical_pair")
        has_pair_in_window = bool(preferred_before_acquisitions and preferred_after_acquisitions)
        has_event_pair = bool(all_before_acquisitions and all_after_acquisitions)
        available_pair = _default_available_pair(source_acquisitions, is_radar, active_geojson)
        comparison_modes = ["Par alrededor del evento"]
        if available_pair is not None:
            comparison_modes.append("Par entre fechas disponibles")
        comparison_mode = st.segmented_control(
            "Tipo de comparación",
            comparison_modes,
            default=(
                "Par alrededor del evento"
                if has_event_pair or available_pair is None
                else "Par entre fechas disponibles"
            ),
            key=f"comparison_pair_mode_{'radar' if is_radar else 'optical'}_{analysis_key}",
            width="stretch",
        )
        before_acquisition: SceneAcquisition | None = None
        after_acquisition: SceneAcquisition | None = None
        left_date_label = "Anterior"
        right_date_label = "Posterior"

        if comparison_mode == "Par entre fechas disponibles" and available_pair is not None:
            if not has_event_pair:
                st.info(
                    "No hay un par completo alrededor del evento. El comparador permite usar dos "
                    "fechas disponibles, aunque ambas pertenezcan al mismo lado del evento."
                )
            ordered_acquisitions = sorted(
                source_acquisitions, key=lambda acquisition: acquisition.acquired_at
            )
            default_before, default_after = available_pair
            before_options = list(reversed(ordered_acquisitions[:-1]))
            scene_columns = st.columns(2)
            with scene_columns[0]:
                before_acquisition = st.selectbox(
                    f"Fecha anterior ({len(before_options)} opciones)",
                    before_options,
                    index=before_options.index(default_before),
                    format_func=lambda acquisition: _acquisition_label(
                        acquisition, active_geojson, event_date, 10_000
                    ),
                    key=f"comparison_free_before_v3_{'radar' if is_radar else 'optical'}_{analysis_key}",
                )
            after_options = [
                acquisition
                for acquisition in reversed(ordered_acquisitions)
                if acquisition.acquired_at > before_acquisition.acquired_at
            ]
            with scene_columns[1]:
                after_acquisition = st.selectbox(
                    f"Fecha posterior ({len(after_options)} opciones)",
                    after_options,
                    index=(
                        after_options.index(default_after) if default_after in after_options else 0
                    ),
                    format_func=lambda acquisition: _acquisition_label(
                        acquisition, active_geojson, event_date, 10_000
                    ),
                    key=f"comparison_free_after_v3_{'radar' if is_radar else 'optical'}_{analysis_key}",
                )
            left_date_label = "Anterior"
            right_date_label = "Posterior"
        else:
            before_acquisitions = all_before_acquisitions
            after_acquisitions = all_after_acquisitions
            sensor_key = "radar" if is_radar else "optical"
            search_before, search_after = state.get("search_windows", {}).get(
                sensor_key,
                (days_before, days_after),
            )
            if state.get("expanded_search", {}).get(sensor_key):
                st.caption(
                    f"Período consultado: {search_before} días anteriores y "
                    f"{search_after} posteriores. La ventana se amplió porque faltaba una fecha "
                    "para completar el par."
                )
            if not has_pair_in_window:
                st.info(
                    "No hay un par completo dentro de la ventana elegida. Las fechas cercanas "
                    "encontradas permanecen disponibles."
                )
            scene_columns = st.columns(2)
            with scene_columns[0]:
                if before_acquisitions:
                    before_acquisition = st.selectbox(
                        f"Adquisición anterior ({len(before_acquisitions)} disponibles)",
                        before_acquisitions,
                        index=_default_acquisition_index(
                            before_acquisitions, preferred_pair.before if preferred_pair else None
                        ),
                        format_func=lambda acquisition: _acquisition_label(
                            acquisition, active_geojson, event_date, days_before
                        ),
                        key=f"comparison_before_{'radar' if is_radar else 'optical'}_{analysis_key}",
                    )
                else:
                    st.caption("Sin adquisiciones anteriores en este rango")
            with scene_columns[1]:
                if after_acquisitions:
                    after_acquisition = st.selectbox(
                        f"Adquisición posterior ({len(after_acquisitions)} disponibles)",
                        after_acquisitions,
                        index=_default_acquisition_index(
                            after_acquisitions, preferred_pair.after if preferred_pair else None
                        ),
                        format_func=lambda acquisition: _acquisition_label(
                            acquisition, active_geojson, event_date, days_after
                        ),
                        key=f"comparison_after_{'radar' if is_radar else 'optical'}_{analysis_key}",
                    )
                else:
                    st.caption("Sin adquisiciones posteriores en este rango")

        if before_acquisition is None and after_acquisition is None:
            st.info("No se encontraron imágenes del sensor para el rango disponible.")
        else:
            if is_radar:
                layer_options = [RADAR_STANDARD_LAYER, *RADAR_VISUAL_LAYERS]
            else:
                layer_options = [OPTICAL_STANDARD_LAYER, *SENTINEL2_INDEX_LAYERS]
            layer = st.selectbox(
                "Visualización",
                layer_options,
                key=f"comparison_layer_{'radar' if is_radar else 'optical'}_{analysis_key}",
            )
            chosen_acquisitions = [
                acquisition
                for acquisition in (before_acquisition, after_acquisition)
                if acquisition is not None
            ]
            incomplete_coverage = [
                (
                    acquisition,
                    acquisition_coverage_pct(acquisition, active_geojson),
                )
                for acquisition in chosen_acquisitions
                if acquisition_coverage_pct(acquisition, active_geojson) < 95
            ]
            if incomplete_coverage:
                coverage_text = ", ".join(
                    f"{acquisition.acquired_at:%d/%m}: {coverage:.0f}%"
                    for acquisition, coverage in incomplete_coverage
                )
                st.warning(
                    "Cobertura incompleta sobre la selección ("
                    f"{coverage_text}). Fuera de la huella satelital permanece visible el mapa base."
                )
            if not is_radar and any(
                acquisition.cloud_cover is not None and acquisition.cloud_cover > max_cloud
                for acquisition in chosen_acquisitions
            ):
                st.warning(
                    "Al menos una fecha supera el límite de nubosidad de escena. "
                    "La imagen continúa disponible en el visor; las métricas ópticas verifican "
                    "además las nubes dentro de cada lote."
                )
            try:
                if before_acquisition is not None and after_acquisition is not None:
                    comparable = not is_radar or (
                        before_acquisition.relative_orbit is not None
                        and before_acquisition.relative_orbit == after_acquisition.relative_orbit
                        and before_acquisition.orbit_state == after_acquisition.orbit_state
                    )
                    if not comparable:
                        st.warning(
                            "Estas pasadas radar no comparten órbita relativa y dirección; la vista "
                            "queda disponible, pero el cálculo cuantitativo permanece bloqueado."
                        )
                    before_sources = _tile_sources(before_acquisition, layer)
                    after_sources = _tile_sources(after_acquisition, layer)
                    left, right = st.columns(2)
                    left.markdown(
                        f"**{left_date_label} · "
                        f"{before_acquisition.acquired_at:%d/%m/%Y %H:%M UTC}**"
                    )
                    right.markdown(
                        f"**{right_date_label} · "
                        f"{after_acquisition.acquired_at:%d/%m/%Y %H:%M UTC}**"
                    )
                    map_view = comparison_tile_swipe_map(
                        before_sources,
                        after_sources,
                        active_geojson,
                    )
                else:
                    available_acquisition = before_acquisition or after_acquisition
                    assert available_acquisition is not None
                    side_label = (
                        left_date_label if before_acquisition is not None else right_date_label
                    )
                    st.markdown(
                        f"**{side_label} · {available_acquisition.acquired_at:%d/%m/%Y %H:%M UTC}**"
                    )
                    map_view = satellite_tile_map(
                        _tile_sources(available_acquisition, layer),
                        active_geojson,
                        side_label,
                    )
                st.iframe(map_view.get_root().render(), width="stretch", height=620)
                guidance = LAYER_GUIDANCE.get(layer)
                if guidance:
                    st.markdown(guidance)
            except Exception as exc:  # noqa: BLE001 - render elegido aislado
                st.error(f"No se pudo preparar la visualización elegida: {exc}")


with compare_tab:
    render_comparator(
        state,
        active_geojson,
        event_date,
        days_before,
        days_after,
        max_cloud,
        analysis_key,
    )


with weather_tab:
    st.subheader("Contexto climático")
    if not state:
        st.info("El contexto climático queda disponible después de la búsqueda de imágenes.")
    elif state.get("fatal_error"):
        st.info("El contexto climático queda asociado a una selección satelital válida.")
    else:
        radar_weather_pair = state.get("weather_radar_pair")
        try:
            weather = state["weather"]

            chart_rows = [
                {
                    "fecha": observation.day.isoformat(),
                    "precipitacion_mm": observation.precipitation_mm,
                    "humedad_superficial": observation.surface_soil_wetness,
                }
                for observation in weather.observations
            ]
            markers = _weather_markers(event_date, radar_weather_pair)
            chart_columns = st.columns(2)
            with chart_columns[0]:
                st.markdown("#### Precipitación diaria")
                st.vega_lite_chart(
                    spec=_weather_chart(
                        chart_rows,
                        markers,
                        "precipitacion_mm",
                        "Precipitación (mm/día)",
                        "#3D78A5",
                        "bar",
                    ),
                    width="stretch",
                )
            with chart_columns[1]:
                st.markdown("#### Humedad superficial del suelo")
                st.vega_lite_chart(
                    spec=_weather_chart(
                        chart_rows,
                        markers,
                        "humedad_superficial",
                        "Humedad superficial (0–1)",
                        "#8A6742",
                        "line",
                    ),
                    width="stretch",
                )

            references: list[tuple[str, date]] = []
            if radar_weather_pair:
                references.append(("Anterior radar", radar_weather_pair[0].acquired_date))
            references.append(("Evento", event_date))
            if radar_weather_pair:
                references.append(("Posterior radar", radar_weather_pair[1].acquired_date))

            context_rows: list[dict[str, Any]] = []
            for label, reference_day in references:
                wetness = weather.surface_wetness_on_or_before(reference_day)
                context_rows.append(
                    {
                        "momento": label,
                        "fecha": reference_day.strftime("%d/%m/%Y"),
                        "lluvia_previa_24h_mm": weather.precipitation_before(reference_day, 1),
                        "lluvia_previa_72h_mm": weather.precipitation_before(reference_day, 3),
                        "lluvia_previa_7d_mm": weather.precipitation_before(reference_day, 7),
                        "humedad_superficial_0_1": wetness[0] if wetness else None,
                        "fecha_dato_humedad": wetness[1].strftime("%d/%m/%Y") if wetness else None,
                    }
                )
            st.dataframe(
                context_rows,
                width="stretch",
                hide_index=True,
                column_config={
                    "momento": "Referencia",
                    "fecha": "Fecha",
                    "lluvia_previa_24h_mm": st.column_config.NumberColumn(
                        "Lluvia previa 24 h (mm)", format="%.2f"
                    ),
                    "lluvia_previa_72h_mm": st.column_config.NumberColumn(
                        "Lluvia previa 72 h (mm)", format="%.2f"
                    ),
                    "lluvia_previa_7d_mm": st.column_config.NumberColumn(
                        "Lluvia previa 7 días (mm)", format="%.2f"
                    ),
                    "humedad_superficial_0_1": st.column_config.NumberColumn(
                        "Humedad superficial (0–1)", format="%.2f"
                    ),
                    "fecha_dato_humedad": "Fecha del dato de humedad",
                },
            )

            if radar_weather_pair:
                before_wetness = weather.surface_wetness_on_or_before(
                    radar_weather_pair[0].acquired_date
                )
                after_wetness = weather.surface_wetness_on_or_before(
                    radar_weather_pair[1].acquired_date
                )
                if before_wetness and after_wetness:
                    wetness_difference = after_wetness[0] - before_wetness[0]
                    if abs(wetness_difference) < 0.03:
                        st.info(
                            "Las adquisiciones radar presentan condiciones de humedad superficial "
                            "similares según NASA POWER."
                        )
                    elif wetness_difference > 0:
                        st.info(
                            "La adquisición posterior presenta mayor humedad superficial. Esta "
                            "diferencia puede contribuir al cambio observado en Sentinel-1."
                        )
                    else:
                        st.info(
                            "La adquisición anterior presenta mayor humedad superficial. Esta "
                            "diferencia puede contribuir al cambio observado en Sentinel-1."
                        )
                else:
                    st.info(
                        "NASA POWER todavía no dispone de humedad superficial para una o ambas "
                        "adquisiciones radar."
                    )

            st.caption(
                "NASA POWER · punto representativo de la selección · horario solar local. "
                "La precipitación y la humedad superficial aportan contexto regional y no describen "
                "variaciones dentro de cada lote. La humedad se expresa entre 0 (seco) y 1 (saturado)."
            )
        except Exception as exc:  # noqa: BLE001 - servicio externo aislado del análisis
            message = state.get("weather_error", str(exc))
            st.warning(f"El contexto de NASA POWER no está disponible en este momento: {message}")


@st.fragment
def render_metrics(
    state: dict[str, Any] | None,
    active_geojson: dict[str, Any],
    analysis_key: str,
    field_score_key: str,
) -> None:
    st.subheader("Resultados Sentinel-1")
    if not state:
        st.info("Los resultados quedan disponibles después de la búsqueda de adquisiciones.")
        return

    result: SensorMetricResult | None = state.get("metrics", {}).get("radar")
    if result is None:
        st.info(
            "No hubo un par Sentinel-1 compatible para realizar el cálculo. Las adquisiciones "
            "deben compartir órbita relativa y dirección."
        )
        message = state.get("errors", {}).get("Métricas Sentinel-1")
        if message:
            st.warning(message)
        return

    if state.get("radar_metrics_version") != RADAR_METRICS_VERSION:
        st.info(
            "Las escenas corresponden a una búsqueda anterior a la clasificación intralote "
            "actual. Una nueva búsqueda procesa esas adquisiciones con la metodología vigente."
        )
        return

    calculated_rows = [
        row for row in result.lot_summaries if row.get("estado") == "Calculado"
    ]
    if not calculated_rows:
        reasons = sorted(
            {
                str(row.get("motivo", "")).strip()
                for row in result.lot_summaries
                if str(row.get("motivo", "")).strip()
            }
        )
        detail = reasons[0] if len(reasons) == 1 else " · ".join(reasons[:3])
        st.warning(
            "Las escenas fueron encontradas, pero no fue posible calcular los lotes."
            + (f" Motivo: {detail}" if detail else "")
        )

    view_mode = st.segmented_control(
        "Vista del mapa",
        ["Cambio intralote", "Resultado por lote"],
        default="Cambio intralote",
        key=f"metric_view_{analysis_key}",
    )
    summary = result.summary
    total_area = float(summary.get("area_referencia_ha") or 0)
    changed_area = float(summary.get("cambio_total_ha") or 0)
    moderate_strong_area = float(summary.get("cambio_moderado_fuerte_ha") or 0)
    changed_pct = 100 * changed_area / total_area if total_area else 0
    summary_columns = st.columns(6)
    summary_columns[0].metric("Superficie analizada", f"{total_area:,.1f} ha")
    summary_columns[1].metric("Superficie con cambio", f"{changed_area:,.1f} ha")
    summary_columns[2].metric("Cambio sobre el total", f"{changed_pct:,.1f}%")
    summary_columns[3].metric("Moderado + fuerte", f"{moderate_strong_area:,.1f} ha")
    summary_columns[4].metric(
        "Cambio claro", int(summary.get("lotes_cambio_claro") or 0)
    )
    summary_columns[5].metric(
        "Indeterminados", int(summary.get("lotes_indeterminados") or 0)
    )

    editor_key = f"metric_editor_{analysis_key}"
    editor_rows = _result_editor_rows(
        result.lot_summaries,
        st.session_state[field_score_key],
    )
    column_order = [
        "ver_en_mapa",
        "pedido",
        "establecimiento",
        "area_referencia_ha",
        "dano_observado_campo",
        "resultado_satelital",
        "patron_respuesta_radar",
        "cambio_total_ha",
        "cambio_total_pct",
        "cambio_moderado_fuerte_ha",
        "cambio_moderado_fuerte_pct",
        "clase_0_ha",
        "clase_0_pct",
        "clase_1_ha",
        "clase_1_pct",
        "clase_2_ha",
        "clase_2_pct",
        "clase_3_ha",
        "clase_3_pct",
        "motivo",
    ]
    edited_table = st.data_editor(
        editor_rows,
        width="stretch",
        height=430,
        hide_index=True,
        num_rows="fixed",
        column_order=column_order,
        disabled=[
            column
            for column in column_order
            if column not in {"ver_en_mapa", "dano_observado_campo"}
        ],
        column_config={
            "ver_en_mapa": st.column_config.CheckboxColumn("Ver", width="small"),
            "pedido": "Lote",
            "establecimiento": "Campo",
            "area_referencia_ha": st.column_config.NumberColumn(
                "Superficie analizada (ha)", format="%.2f"
            ),
            "dano_observado_campo": st.column_config.NumberColumn(
                "Daño observado en campo (0–10)",
                min_value=0.0,
                max_value=10.0,
                step=0.5,
                format="%.1f",
            ),
            "resultado_satelital": "Resultado satelital",
            "patron_respuesta_radar": "Patrón de respuesta radar",
            "cambio_total_ha": st.column_config.NumberColumn(
                "Cambio total (ha)", format="%.2f"
            ),
            "cambio_total_pct": st.column_config.NumberColumn(
                "Cambio total (%)", format="%.1f"
            ),
            "cambio_moderado_fuerte_ha": st.column_config.NumberColumn(
                "Moderado + fuerte (ha)", format="%.2f"
            ),
            "cambio_moderado_fuerte_pct": st.column_config.NumberColumn(
                "Moderado + fuerte (%)", format="%.1f"
            ),
            "clase_0_ha": st.column_config.NumberColumn(
                "Sin cambio destacado (ha)", format="%.2f"
            ),
            "clase_0_pct": st.column_config.NumberColumn(
                "Sin cambio destacado (%)", format="%.1f"
            ),
            "clase_1_ha": st.column_config.NumberColumn("Cambio leve (ha)", format="%.2f"),
            "clase_1_pct": st.column_config.NumberColumn("Cambio leve (%)", format="%.1f"),
            "clase_2_ha": st.column_config.NumberColumn(
                "Cambio moderado (ha)", format="%.2f"
            ),
            "clase_2_pct": st.column_config.NumberColumn(
                "Cambio moderado (%)", format="%.1f"
            ),
            "clase_3_ha": st.column_config.NumberColumn("Cambio fuerte (ha)", format="%.2f"),
            "clase_3_pct": st.column_config.NumberColumn("Cambio fuerte (%)", format="%.1f"),
            "motivo": "Estado del cálculo",
        },
        key=editor_key,
    )

    edited_rows = (
        edited_table.to_dict("records")
        if hasattr(edited_table, "to_dict")
        else list(edited_table)
    )
    field_scores = dict(st.session_state[field_score_key])
    for row in edited_rows:
        order = str(row.get("pedido", ""))
        value = row.get("dano_observado_campo")
        if value is None or (isinstance(value, float) and math.isnan(value)):
            field_scores.pop(order, None)
        else:
            field_scores[order] = float(value)
    score_state = st.session_state[field_score_key]
    score_state.clear()
    score_state.update(field_scores)

    checked_rows = [
        index for index, row in enumerate(edited_rows) if bool(row.get("ver_en_mapa"))
    ]
    selected_row = checked_rows[-1] if checked_rows else None
    metric_map = radar_results_map(
        active_geojson,
        result.lot_summaries,
        getattr(result, "overlays", ()),
        view_mode or "Cambio intralote",
        selected_row,
    )
    st.iframe(metric_map.get_root().render(), width="stretch", height=640)
    st.caption(
        "La clasificación representa cambios de la señal Sentinel-1 dentro de los lotes. "
        "La evaluación de campo permanece editable y no modifica automáticamente el cálculo "
        "satelital de la sesión."
    )

with metrics_tab:
    render_metrics(state, active_geojson, analysis_key, field_score_key)

@st.fragment
def render_downloads(
    state: dict[str, Any] | None,
    active_geojson: dict[str, Any],
    active_truth: list[dict[str, Any]],
    field_score_key: str,
    event_date: date,
    selected_value: str,
) -> None:
    st.subheader("Descargas")
    if not state:
        st.info("Los archivos quedan disponibles después de la búsqueda de adquisiciones.")
        return
    result: SensorMetricResult | None = state.get("metrics", {}).get("radar")
    radar_pair = state.get("metric_pairs", {}).get("radar")
    if result is None or radar_pair is None:
        st.info("No hay resultados Sentinel-1 disponibles para preparar los archivos.")
        return

    before, after = radar_pair
    score_state = st.session_state[field_score_key]

    def observations() -> list[dict[str, Any]]:
        return _field_rows_from_scores(active_truth, active_geojson, score_state)

    safe_selection = selected_value.replace(" ", "_")
    st.download_button(
        "Descargar tabla de resultados (.xlsx)",
        data=lambda: build_results_workbook(result, observations(), before, after),
        file_name=f"resultados_{safe_selection}_{event_date.isoformat()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        on_click="ignore",
        width="stretch",
    )

    export_images = state.get("export_images", {})
    st.download_button(
        "Descargar paquete completo (.zip)",
        data=lambda: build_download_package(
            active_geojson,
            event_date,
            result,
            observations(),
            before,
            after,
            state.get("weather"),
            export_images,
            state.get("export_image_bbox"),
        ),
        file_name=f"analisis_{safe_selection}_{event_date.isoformat()}.zip",
        mime="application/zip",
        on_click="ignore",
        width="stretch",
    )
    st.caption(
        "El paquete contiene la tabla, los lotes con resultados, la clasificación intralote "
        "georreferenciada, las zonas de cambio, el contexto climático, los metadatos y las "
        "imágenes Sentinel-1 anterior y posterior."
    )
    if not export_images:
        st.warning(
            "Esta sesión se procesó antes de habilitar la exportación de imágenes. Los demás "
            "archivos permanecen disponibles; una nueva búsqueda incorpora las imágenes al ZIP."
        )


with export_tab:
    render_downloads(
        state,
        active_geojson,
        active_truth,
        field_score_key,
        event_date,
        selected_value,
    )
