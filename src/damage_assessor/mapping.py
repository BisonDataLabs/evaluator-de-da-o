from __future__ import annotations

import base64
import copy
from typing import Any

from branca.element import MacroElement, Template

from .geometry import geojson_bbox, profile_geojson
from .imagery import padded_bbox

HYBRID_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
LABEL_TILES = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/Reference/"
    "World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
)


class _ImageSwipeControl(MacroElement):
    """Leaflet swipe control compatible with ImageOverlay layers."""

    _template = Template(
        """
        {% macro header(this, kwargs) %}
        <style>
          #{{ this.get_name() }} {
            position: absolute; inset: 0; width: 100%; height: 100%; margin: 0;
            z-index: 1000; appearance: none; -webkit-appearance: none;
            background: transparent; pointer-events: none;
          }
          #{{ this.get_name() }}::-webkit-slider-runnable-track {
            height: 100%; background: transparent;
          }
          #{{ this.get_name() }}::-webkit-slider-thumb {
            appearance: none; -webkit-appearance: none; pointer-events: auto;
            width: 6px; height: 620px; cursor: ew-resize; background: #ffffff;
            border: 1px solid #17313b; box-shadow: 0 0 0 2px rgba(255,255,255,.7);
          }
          #{{ this.get_name() }}::-moz-range-track { background: transparent; }
          #{{ this.get_name() }}::-moz-range-thumb {
            pointer-events: auto; width: 6px; height: 620px; cursor: ew-resize;
            background: #ffffff; border: 1px solid #17313b;
          }
        </style>
        {% endmacro %}
        {% macro script(this, kwargs) %}
        const swipeMap = {{ this._parent.get_name() }};
        const swipeLeft = {{ this.left_layer.get_name() }};
        const swipeRight = {{ this.right_layer.get_name() }};
        const swipeRange = document.createElement("input");
        swipeRange.id = "{{ this.get_name() }}";
        swipeRange.type = "range";
        swipeRange.min = "0";
        swipeRange.max = "100";
        swipeRange.value = "50";
        swipeRange.setAttribute("aria-label", "Comparación anterior y posterior");
        swipeMap.getContainer().appendChild(swipeRange);
        L.DomEvent.disableClickPropagation(swipeRange);
        L.DomEvent.disableScrollPropagation(swipeRange);

        function updateSwipeClip() {
          const leftElement = swipeLeft.getElement();
          const rightElement = swipeRight.getElement();
          if (!leftElement || !rightElement) return;
          const mapRect = swipeMap.getContainer().getBoundingClientRect();
          const splitX = mapRect.left + mapRect.width * Number(swipeRange.value) / 100;
          const leftRect = leftElement.getBoundingClientRect();
          const rightRect = rightElement.getBoundingClientRect();
          const leftRightInset = Math.max(0, Math.min(leftRect.width, leftRect.right - splitX));
          const rightLeftInset = Math.max(0, Math.min(rightRect.width, splitX - rightRect.left));
          const leftClip = `inset(0px ${leftRightInset}px 0px 0px)`;
          const rightClip = `inset(0px 0px 0px ${rightLeftInset}px)`;
          leftElement.style.clipPath = leftClip;
          leftElement.style.webkitClipPath = leftClip;
          rightElement.style.clipPath = rightClip;
          rightElement.style.webkitClipPath = rightClip;
        }

        swipeRange.addEventListener("input", updateSwipeClip);
        swipeLeft.on("load", updateSwipeClip);
        swipeRight.on("load", updateSwipeClip);
        swipeMap.on("move zoom resize", updateSwipeClip);
        setTimeout(updateSwipeClip, 0);
        {% endmacro %}
        """
    )

    def __init__(self, left_layer: Any, right_layer: Any) -> None:
        super().__init__()
        self._name = "ImageSwipeControl"
        self.left_layer = left_layer
        self.right_layer = right_layer


class _TileLayerSwipeControl(MacroElement):
    """Leaflet swipe control for one or many raster tile layers on each date."""

    _template = Template(
        """
        {% macro header(this, kwargs) %}
        <style>
          #{{ this.get_name() }} {
            position: absolute; inset: 0; width: 100%; height: 100%; margin: 0;
            z-index: 1000; appearance: none; -webkit-appearance: none;
            background: transparent; pointer-events: none;
          }
          #{{ this.get_name() }}::-webkit-slider-runnable-track {
            height: 100%; background: transparent;
          }
          #{{ this.get_name() }}::-webkit-slider-thumb {
            appearance: none; -webkit-appearance: none; pointer-events: auto;
            width: 6px; height: 620px; cursor: ew-resize; background: #ffffff;
            border: 1px solid #17313b; box-shadow: 0 0 0 2px rgba(255,255,255,.7);
          }
          #{{ this.get_name() }}::-moz-range-track { background: transparent; }
          #{{ this.get_name() }}::-moz-range-thumb {
            pointer-events: auto; width: 6px; height: 620px; cursor: ew-resize;
            background: #ffffff; border: 1px solid #17313b;
          }
        </style>
        {% endmacro %}
        {% macro script(this, kwargs) %}
        const tileSwipeMap = {{ this._parent.get_name() }};
        const tileSwipeLeftLayers = [
          {% for layer in this.left_layers %}{{ layer.get_name() }},{% endfor %}
        ];
        const tileSwipeRightLayers = [
          {% for layer in this.right_layers %}{{ layer.get_name() }},{% endfor %}
        ];

        const tileSwipeRange = document.createElement("input");
        tileSwipeRange.id = "{{ this.get_name() }}";
        tileSwipeRange.type = "range";
        tileSwipeRange.min = "0";
        tileSwipeRange.max = "100";
        tileSwipeRange.value = "50";
        tileSwipeRange.setAttribute("aria-label", "Comparación anterior y posterior");
        tileSwipeMap.getContainer().appendChild(tileSwipeRange);
        L.DomEvent.disableClickPropagation(tileSwipeRange);
        L.DomEvent.disableScrollPropagation(tileSwipeRange);

        function updateTileSwipeClip() {
          const mapSize = tileSwipeMap.getSize();
          const northWest = tileSwipeMap.containerPointToLayerPoint([0, 0]);
          const southEast = tileSwipeMap.containerPointToLayerPoint(mapSize);
          const splitX = northWest.x + mapSize.x * Number(tileSwipeRange.value) / 100;
          const leftClip = `rect(${northWest.y}px, ${splitX}px, ${southEast.y}px, ${northWest.x}px)`;
          const rightClip = `rect(${northWest.y}px, ${southEast.x}px, ${southEast.y}px, ${splitX}px)`;
          for (const layer of tileSwipeLeftLayers) {
            const container = layer.getContainer();
            if (container) container.style.clip = leftClip;
          }
          for (const layer of tileSwipeRightLayers) {
            const container = layer.getContainer();
            if (container) container.style.clip = rightClip;
          }
        }
        tileSwipeRange.addEventListener("input", updateTileSwipeClip);
        for (const layer of [...tileSwipeLeftLayers, ...tileSwipeRightLayers]) {
          layer.on("load tileload", updateTileSwipeClip);
        }
        tileSwipeMap.on("move zoom resize layeradd", updateTileSwipeClip);
        setTimeout(updateTileSwipeClip, 0);
        setTimeout(updateTileSwipeClip, 300);
        {% endmacro %}
        """
    )

    def __init__(self, left_layers: list[Any], right_layers: list[Any]) -> None:
        super().__init__()
        self._name = "TileLayerSwipeControl"
        self.left_layers = left_layers
        self.right_layers = right_layers


class _ResponsiveFitBounds(MacroElement):
    """Refit a Leaflet map when a hidden Streamlit tab becomes visible."""

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        const responsiveMap = {{ this._parent.get_name() }};
        const responsiveBounds = L.latLngBounds({{ this.bounds | tojson }});
        let responsiveWasVisible = false;

        function fitVisibleMap() {
          const container = responsiveMap.getContainer();
          const visible = container.clientWidth > 40 && container.clientHeight > 40;
          if (!visible) {
            responsiveWasVisible = false;
            return;
          }
          responsiveMap.invalidateSize({pan: false, animate: false});
          if (!responsiveWasVisible) {
            responsiveMap.fitBounds(responsiveBounds, {padding: [12, 12], animate: false});
            responsiveWasVisible = true;
          }
          responsiveMap.fire("resize");
        }

        const responsiveObserver = new ResizeObserver(fitVisibleMap);
        responsiveObserver.observe(responsiveMap.getContainer());
        window.addEventListener("resize", fitVisibleMap);
        setTimeout(fitVisibleMap, 0);
        setTimeout(fitVisibleMap, 300);
        setTimeout(fitVisibleMap, 1000);
        {% endmacro %}
        """
    )

    def __init__(self, bounds: list[list[float]]) -> None:
        super().__init__()
        self._name = "ResponsiveFitBounds"
        self.bounds = bounds


def _data_uri(payload: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(payload).decode('ascii')}"


def _leaflet_bounds(
    bbox: tuple[float, float, float, float],
) -> list[list[float]]:
    min_x, min_y, max_x, max_y = bbox
    return [[min_y, min_x], [max_y, max_x]]


def _add_hybrid_base(map_view: Any) -> None:
    import folium

    folium.TileLayer(
        tiles=HYBRID_TILES,
        attr="Esri World Imagery",
        name="Híbrido",
        overlay=False,
        control=True,
    ).add_to(map_view)


def _add_labels(map_view: Any) -> None:
    import folium

    folium.TileLayer(
        tiles=LABEL_TILES,
        attr="Esri Boundaries and Places",
        name="Etiquetas",
        overlay=True,
        control=False,
    ).add_to(map_view)


def _add_boundaries(map_view: Any, geojson: dict[str, Any]) -> None:
    import folium

    properties = profile_geojson(geojson).properties
    tooltip_fields = [key for key in ("pedido", "establecimiento", "lote") if key in properties]
    aliases = {"pedido": "Pedido", "establecimiento": "Establecimiento", "lote": "Lote"}
    folium.GeoJson(
        geojson,
        name="Lotes seleccionados",
        style_function=lambda _: {
            "color": "#FFFFFF",
            "weight": 3,
            "fillColor": "#9B3A32",
            "fillOpacity": 0.08,
        },
        highlight_function=lambda _: {
            "color": "#F4C542",
            "weight": 5,
            "fillOpacity": 0.18,
        },
        tooltip=(
            folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=[aliases[key] for key in tooltip_fields],
            )
            if tooltip_fields
            else None
        ),
    ).add_to(map_view)


def comparison_dual_map(
    before_png: bytes,
    after_png: bytes,
    geojson: dict[str, Any],
    render_bbox: tuple[float, float, float, float],
) -> Any:
    """Create two synchronized, zoomable maps with the selected boundaries on both sides."""
    import folium
    from folium.plugins import DualMap

    min_x, min_y, max_x, max_y = geojson_bbox(geojson)
    location = [(min_y + max_y) / 2, (min_x + max_x) / 2]
    dual = DualMap(location=location, tiles=None, control_scale=True)
    image_bounds = _leaflet_bounds(render_bbox)
    selection_bounds = _leaflet_bounds(padded_bbox((min_x, min_y, max_x, max_y), 0.08))

    for map_view, payload, name in (
        (dual.m1, before_png, "Anterior"),
        (dual.m2, after_png, "Posterior"),
    ):
        _add_hybrid_base(map_view)
        folium.raster_layers.ImageOverlay(
            image=_data_uri(payload),
            bounds=image_bounds,
            name=name,
            opacity=1.0,
            interactive=True,
            cross_origin=False,
            zindex=2,
        ).add_to(map_view)
        _add_labels(map_view)
        _add_boundaries(map_view, geojson)
        map_view.fit_bounds(selection_bounds)
    return dual


def comparison_swipe_map(
    before_png: bytes,
    after_png: bytes,
    geojson: dict[str, Any],
    render_bbox: tuple[float, float, float, float],
) -> Any:
    """Create a zoomable map with a draggable before/after divider."""
    import folium

    min_x, min_y, max_x, max_y = geojson_bbox(geojson)
    map_view = folium.Map(
        location=[(min_y + max_y) / 2, (min_x + max_x) / 2],
        tiles=None,
        control_scale=True,
        zoom_control=True,
    )
    _add_hybrid_base(map_view)
    image_bounds = _leaflet_bounds(render_bbox)
    before_layer = folium.raster_layers.ImageOverlay(
        image=_data_uri(before_png),
        bounds=image_bounds,
        name="Anterior",
        opacity=1.0,
        interactive=True,
        cross_origin=False,
        zindex=2,
    ).add_to(map_view)
    after_layer = folium.raster_layers.ImageOverlay(
        image=_data_uri(after_png),
        bounds=image_bounds,
        name="Posterior",
        opacity=1.0,
        interactive=True,
        cross_origin=False,
        zindex=2,
    ).add_to(map_view)
    _ImageSwipeControl(before_layer, after_layer).add_to(map_view)
    _add_labels(map_view)
    _add_boundaries(map_view, geojson)
    selection_bounds = _leaflet_bounds(padded_bbox((min_x, min_y, max_x, max_y), 0.08))
    _ResponsiveFitBounds(selection_bounds).add_to(map_view)
    return map_view


def _coverage_features(sources: list[dict[str, Any]], date_label: str) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    for source in sources:
        geometry = source.get("geometry")
        if not geometry:
            bbox = source.get("bbox")
            if not bbox:
                continue
            min_x, min_y, max_x, max_y = bbox
            geometry = {
                "type": "Polygon",
                "coordinates": [
                    [
                        [min_x, min_y],
                        [max_x, min_y],
                        [max_x, max_y],
                        [min_x, max_y],
                        [min_x, min_y],
                    ]
                ],
            }
        features.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "fecha": date_label,
                    "producto": source.get("label", ""),
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def comparison_tile_swipe_map(
    before_sources: list[dict[str, Any]],
    after_sources: list[dict[str, Any]],
    geojson: dict[str, Any],
) -> Any:
    """Create a native-resolution tile mosaic with a draggable before/after divider."""
    import folium

    min_x, min_y, max_x, max_y = geojson_bbox(geojson)
    map_view = folium.Map(
        location=[(min_y + max_y) / 2, (min_x + max_x) / 2],
        tiles=None,
        control_scale=True,
        zoom_control=True,
        prefer_canvas=True,
    )
    _add_hybrid_base(map_view)
    before_layers: list[Any] = []
    after_layers: list[Any] = []
    for index, source in enumerate(before_sources):
        layer = folium.TileLayer(
            tiles=source["url"],
            attr="Microsoft Planetary Computer",
            name=f"Anterior {index + 1}",
            overlay=True,
            control=False,
            opacity=1.0,
        )
        layer.add_to(map_view)
        before_layers.append(layer)
    for index, source in enumerate(after_sources):
        layer = folium.TileLayer(
            tiles=source["url"],
            attr="Microsoft Planetary Computer",
            name=f"Posterior {index + 1}",
            overlay=True,
            control=False,
            opacity=1.0,
        )
        layer.add_to(map_view)
        after_layers.append(layer)
    _TileLayerSwipeControl(before_layers, after_layers).add_to(map_view)

    for sources, label, color in (
        (before_sources, "Anterior", "#00D5FF"),
        (after_sources, "Posterior", "#FF4FA3"),
    ):
        coverage = _coverage_features(sources, label)
        if coverage["features"]:
            folium.GeoJson(
                coverage,
                name=f"Cobertura {label.lower()}",
                style_function=lambda _, line_color=color: {
                    "color": line_color,
                    "weight": 2,
                    "dashArray": "7 5",
                    "fillOpacity": 0,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=["fecha", "producto"], aliases=["Fecha", "Cobertura"]
                ),
            ).add_to(map_view)
    _add_labels(map_view)
    _add_boundaries(map_view, geojson)
    folium.LayerControl(collapsed=True).add_to(map_view)
    selection_bounds = _leaflet_bounds(padded_bbox((min_x, min_y, max_x, max_y), 0.08))
    _ResponsiveFitBounds(selection_bounds).add_to(map_view)
    return map_view


def satellite_tile_map(
    sources: list[dict[str, Any]],
    geojson: dict[str, Any],
    date_label: str,
) -> Any:
    """Create a zoomable satellite mosaic when only one side of the event is available."""
    import folium

    min_x, min_y, max_x, max_y = geojson_bbox(geojson)
    map_view = folium.Map(
        location=[(min_y + max_y) / 2, (min_x + max_x) / 2],
        tiles=None,
        control_scale=True,
        zoom_control=True,
        prefer_canvas=True,
    )
    _add_hybrid_base(map_view)
    for index, source in enumerate(sources):
        folium.TileLayer(
            tiles=source["url"],
            attr="Microsoft Planetary Computer",
            name=f"Imagen {index + 1}",
            overlay=True,
            control=False,
            opacity=1.0,
        ).add_to(map_view)
    coverage = _coverage_features(sources, date_label)
    if coverage["features"]:
        folium.GeoJson(
            coverage,
            name="Cobertura de la imagen",
            style_function=lambda _: {
                "color": "#00D5FF",
                "weight": 2,
                "dashArray": "7 5",
                "fillOpacity": 0,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["fecha", "producto"], aliases=["Fecha", "Cobertura"]
            ),
        ).add_to(map_view)
    _add_labels(map_view)
    _add_boundaries(map_view, geojson)
    folium.LayerControl(collapsed=True).add_to(map_view)
    selection_bounds = _leaflet_bounds(padded_bbox((min_x, min_y, max_x, max_y), 0.08))
    _ResponsiveFitBounds(selection_bounds).add_to(map_view)
    return map_view


def metric_change_map(
    change_png: bytes | None,
    geojson: dict[str, Any],
    render_bbox: tuple[float, float, float, float] | None,
    lot_summaries: list[dict[str, Any]],
    selected_row: int | None = None,
    sensor_label: str = "Sentinel-1 RTC",
) -> Any:
    """Create a zoomable change map and focus it on the selected table row."""
    import folium

    enriched = copy.deepcopy(geojson)
    for index, feature in enumerate(enriched["features"]):
        row = lot_summaries[index] if index < len(lot_summaries) else {}
        properties = feature.setdefault("properties", {})
        properties["_row_index"] = index
        changed = float(
            row.get("area_perturbada_ha_preliminar") or row.get("cambio_radar_ha_preliminar") or 0
        )
        properties["area_perturbada_ha"] = changed
        properties["estado_calculo"] = row.get("estado", "Calculado")
        properties["severidad_preliminar"] = row.get("severidad_satelital_0_10_preliminar", "")
        area = float(row.get("area_referencia_ha") or 0)
        properties["superficie_perturbada_pct"] = round(changed / area * 100, 1) if area else 0
        properties["cambio_radar_pct"] = properties["superficie_perturbada_pct"]

    all_bbox = geojson_bbox(enriched)
    min_x, min_y, max_x, max_y = all_bbox
    map_view = folium.Map(
        location=[(min_y + max_y) / 2, (min_x + max_x) / 2],
        tiles=None,
        control_scale=True,
    )
    _add_hybrid_base(map_view)
    if change_png is not None and render_bbox is not None:
        folium.raster_layers.ImageOverlay(
            image=_data_uri(change_png),
            bounds=_leaflet_bounds(render_bbox),
            name="Cambio raster",
            opacity=0.72,
            interactive=True,
            cross_origin=False,
            zindex=2,
        ).add_to(map_view)
    _add_labels(map_view)

    def style(feature: dict[str, Any]) -> dict[str, Any]:
        properties = feature.get("properties") or {}
        value = float(properties.get("superficie_perturbada_pct") or 0)
        selected = properties.get("_row_index") == selected_row
        blocked = properties.get("estado_calculo") != "Calculado"
        if blocked:
            fill = "#8B9297"
        elif value >= 20:
            fill = "#A50026"
        elif value >= 10:
            fill = "#F46D43"
        elif value >= 3:
            fill = "#FDAE61"
        else:
            fill = "#FFFFBF"
        return {
            "color": "#00E5FF" if selected else "#FFFFFF",
            "weight": 6 if selected else 3,
            "fillColor": fill,
            "fillOpacity": 0.18,
        }

    tooltip_candidates = [
        ("pedido", "Lote"),
        ("establecimiento", "Establecimiento"),
        ("area_perturbada_ha", "Perturbación (ha)"),
        ("superficie_perturbada_pct", "Perturbación (%)"),
        ("severidad_preliminar", "Severidad preliminar (0–10)"),
        ("estado_calculo", "Estado"),
    ]
    available = {
        key for feature in enriched["features"] for key in (feature.get("properties") or {})
    }
    fields = [key for key, _ in tooltip_candidates if key in available]
    aliases = [label for key, label in tooltip_candidates if key in available]
    folium.GeoJson(
        enriched,
        name=f"Métrica por lote · {sensor_label}",
        style_function=style,
        highlight_function=lambda _: {"color": "#F4C542", "weight": 6},
        tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases),
    ).add_to(map_view)
    folium.LayerControl(collapsed=True).add_to(map_view)

    focus_bbox = all_bbox
    if selected_row is not None and 0 <= selected_row < len(enriched["features"]):
        focus_geojson = {
            "type": "FeatureCollection",
            "features": [enriched["features"][selected_row]],
        }
        focus_bbox = geojson_bbox(focus_geojson)
    _ResponsiveFitBounds(_leaflet_bounds(padded_bbox(focus_bbox, 0.12))).add_to(map_view)
    return map_view
