from __future__ import annotations

import io
import math
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
import requests
from PIL import Image

from .catalog import SceneCandidate

RADAR_STANDARD_LAYER = "Sentinel-1 · Composición VV/VH"
OPTICAL_STANDARD_LAYER = "Sentinel-2 · RGB"

RADAR_VISUAL_LAYERS: dict[str, dict[str, object]] = {
    "Sentinel-1 · VV alto contraste": {
        "assets": ("vv",),
        "expression": "vv",
        "rescale": "0,0.30",
        "colormap_name": "gray",
    },
    "Sentinel-1 · VH alto contraste": {
        "assets": ("vh",),
        "expression": "vh",
        "rescale": "0,0.12",
        "colormap_name": "gray",
    },
    "Sentinel-1 · Cociente VV/VH": {
        "assets": ("vv", "vh"),
        "expression": "vv/(vh+0.0001)",
        "rescale": "0,15",
        "colormap_name": "viridis",
    },
    "Sentinel-1 · RVI": {
        "assets": ("vv", "vh"),
        "expression": "4*vh/(vv+vh+0.0001)",
        "rescale": "0,1",
        "colormap_name": "turbo",
    },
}

SENTINEL2_INDEX_LAYERS: dict[str, dict[str, object]] = {
    "Sentinel-2 · NDVI": {
        "assets": ("B08", "B04"),
        "expression": "(B08-B04)/(B08+B04)",
        "rescale": "-1,1",
        "colormap_name": "rdylgn",
    },
    "Sentinel-2 · NDRE": {
        "assets": ("B8A", "B05"),
        "expression": "(B8A-B05)/(B8A+B05)",
        "rescale": "-1,1",
        "colormap_name": "rdylgn",
    },
    "Sentinel-2 · NDMI": {
        "assets": ("B08", "B11"),
        "expression": "(B08-B11)/(B08+B11)",
        "rescale": "-1,1",
        "colormap_name": "rdylbu",
    },
}


class ImageryError(RuntimeError):
    """Raised when a real scene cannot be rendered by the provider."""


def tile_layer_url(
    scene: SceneCandidate,
    *,
    assets: tuple[str, ...] | None = None,
    expression: str | None = None,
    rescale: str | None = None,
    colormap_name: str | None = None,
) -> str:
    """Build a zoom-aware WebMercator tile URL from a Planetary Computer item."""
    source = scene.tilejson_url or scene.rendered_preview_url
    if not source:
        raise ImageryError(f"La escena {scene.scene_id} no ofrece teselas compatibles.")
    parsed = urlsplit(source)
    prefix = parsed.path.split("/item/")[0]
    path = f"{prefix}/item/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}@1x"
    if assets is None and expression is None:
        query = parse_qsl(parsed.query, keep_blank_values=True)
    else:
        query = [
            ("collection", scene.collection),
            ("item", scene.scene_id),
            *(("assets", asset) for asset in (assets or ())),
        ]
        if expression:
            query.extend([("expression", expression), ("asset_as_band", "true")])
        if rescale:
            query.append(("rescale", rescale))
        if colormap_name:
            query.append(("colormap_name", colormap_name))
        query.extend([("return_mask", "true"), ("format", "png")])
    return urlunsplit((parsed.scheme, parsed.netloc, path, urlencode(query), ""))


def tilejson_request_url(
    scene: SceneCandidate,
    *,
    assets: tuple[str, ...] | None = None,
    expression: str | None = None,
    rescale: str | None = None,
    colormap_name: str | None = None,
) -> str:
    """Build the provider TileJSON request used to discover its canonical tile URL."""
    source = scene.tilejson_url or scene.rendered_preview_url
    if not source:
        raise ImageryError(f"La escena {scene.scene_id} no ofrece teselas compatibles.")
    parsed = urlsplit(source)
    prefix = parsed.path.split("/item/")[0]
    path = f"{prefix}/item/tilejson.json"
    if assets is None and expression is None and scene.tilejson_url:
        query = parse_qsl(parsed.query, keep_blank_values=True)
    else:
        query = [
            ("collection", scene.collection),
            ("item", scene.scene_id),
            *(("assets", asset) for asset in (assets or ())),
        ]
        if expression:
            query.extend([("expression", expression), ("asset_as_band", "true")])
        if rescale:
            query.append(("rescale", rescale))
        if colormap_name:
            query.append(("colormap_name", colormap_name))
        query.extend([("return_mask", "true"), ("format", "png")])
    return urlunsplit((parsed.scheme, parsed.netloc, path, urlencode(query), ""))


def resolve_tile_layer_url(
    scene: SceneCandidate,
    *,
    assets: tuple[str, ...] | None = None,
    expression: str | None = None,
    rescale: str | None = None,
    colormap_name: str | None = None,
    timeout_seconds: int = 30,
) -> str:
    """Resolve and validate the exact tile template published by the provider."""
    request_url = tilejson_request_url(
        scene,
        assets=assets,
        expression=expression,
        rescale=rescale,
        colormap_name=colormap_name,
    )
    response = requests.get(
        request_url,
        timeout=timeout_seconds,
        headers={"User-Agent": "ceibos-damage-assessor/0.5"},
    )
    if response.status_code != 200:
        detail = response.text[:240] if response.text else response.reason
        raise ImageryError(
            f"La imagen {scene.scene_id} no pudo abrirse: "
            f"HTTP {response.status_code} · {detail}"
        )
    try:
        tiles = response.json()["tiles"]
        tile_url = str(tiles[0])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ImageryError(
            f"La imagen {scene.scene_id} no publicó una dirección de teselas válida."
        ) from exc
    if "{z}" not in tile_url or "{x}" not in tile_url or "{y}" not in tile_url:
        raise ImageryError(
            f"La imagen {scene.scene_id} devolvió una dirección de teselas incompleta."
        )
    return tile_url


def native_grid_size(
    bbox: tuple[float, float, float, float],
    target_meters: float = 10.0,
    max_dimension: int = 1024,
) -> tuple[int, int]:
    """Approximate a sensor-scale grid while bounding memory and response size."""
    min_x, min_y, max_x, max_y = bbox
    latitude = (min_y + max_y) / 2
    width_m = abs(max_x - min_x) * 111_320 * max(math.cos(math.radians(latitude)), 0.1)
    height_m = abs(max_y - min_y) * 110_574
    width = max(32, math.ceil(width_m / target_meters))
    height = max(32, math.ceil(height_m / target_meters))
    scale = min(1.0, max_dimension / max(width, height))
    return max(32, int(width * scale)), max(32, int(height * scale))


def raw_bbox_url(
    scene: SceneCandidate,
    bbox: tuple[float, float, float, float],
    *,
    assets: tuple[str, ...],
    expression: str,
    width: int,
    height: int,
    categorical: bool = False,
    unscale: bool = False,
) -> str:
    """Request numerical GeoTIFF pixels, never a colour-rendered preview."""
    source = scene.tilejson_url or scene.rendered_preview_url
    if not source:
        raise ImageryError(f"La escena {scene.scene_id} no ofrece acceso de datos compatible.")
    min_x, min_y, max_x, max_y = bbox
    parsed = urlsplit(source)
    prefix = parsed.path.split("/item/")[0]
    path = (
        f"{prefix}/item/bbox/{min_x:.7f},{min_y:.7f},{max_x:.7f},{max_y:.7f}/{width}x{height}.tif"
    )
    resampling = "nearest" if categorical else "bilinear"
    query: list[tuple[str, str]] = [
        ("collection", scene.collection),
        ("item", scene.scene_id),
        *(("assets", asset) for asset in assets),
        ("expression", expression),
        ("asset_as_band", "true"),
        ("coord_crs", "epsg:4326"),
        ("dst_crs", "epsg:4326"),
        ("resampling", resampling),
        ("reproject", resampling),
        ("unscale", "true" if unscale else "false"),
        ("return_mask", "false"),
    ]
    return urlunsplit((parsed.scheme, parsed.netloc, path, urlencode(query), ""))


def fetch_raw_array(
    scene: SceneCandidate,
    bbox: tuple[float, float, float, float],
    *,
    assets: tuple[str, ...],
    expression: str,
    width: int,
    height: int,
    categorical: bool = False,
    unscale: bool = False,
    timeout_seconds: int = 120,
) -> np.ndarray:
    url = raw_bbox_url(
        scene,
        bbox,
        assets=assets,
        expression=expression,
        width=width,
        height=height,
        categorical=categorical,
        unscale=unscale,
    )
    response = requests.get(
        url,
        timeout=timeout_seconds,
        headers={"User-Agent": "ceibos-damage-assessor/0.4"},
    )
    if response.status_code != 200:
        detail = response.text[:240] if response.text else response.reason
        raise ImageryError(
            f"No se pudieron leer píxeles de {scene.scene_id}: "
            f"HTTP {response.status_code} · {detail}"
        )
    try:
        image = Image.open(io.BytesIO(response.content))
        image.load()
        array = np.asarray(image, dtype=np.float32)
    except Exception as exc:
        raise ImageryError("El proveedor no devolvió un GeoTIFF numérico válido.") from exc
    if array.ndim == 3:
        array = array[..., 0]
    if array.shape != (height, width):
        raise ImageryError(
            f"La grilla recibida tiene forma {array.shape}; se esperaba {(height, width)}."
        )
    return array


def padded_bbox(
    bbox: tuple[float, float, float, float], fraction: float = 0.04
) -> tuple[float, float, float, float]:
    min_x, min_y, max_x, max_y = bbox
    pad_x = max((max_x - min_x) * fraction, 0.001)
    pad_y = max((max_y - min_y) * fraction, 0.001)
    return min_x - pad_x, min_y - pad_y, max_x + pad_x, max_y + pad_y


def bbox_image_url(
    scene: SceneCandidate,
    bbox: tuple[float, float, float, float],
    width: int = 960,
    height: int = 640,
) -> str:
    if not scene.rendered_preview_url:
        raise ImageryError(f"La escena {scene.scene_id} no ofrece un render compatible.")
    min_x, min_y, max_x, max_y = padded_bbox(bbox)
    parsed = urlsplit(scene.rendered_preview_url)
    prefix = parsed.path.split("/item/")[0]
    path = (
        f"{prefix}/item/bbox/{min_x:.7f},{min_y:.7f},{max_x:.7f},{max_y:.7f}/{width}x{height}.png"
    )
    # Keep repeated parameters such as ``assets=vv&assets=vh``. Converting the
    # query to a dict silently discarded VV and broke the provider expression.
    overrides = {
        "collection": scene.collection,
        "item": scene.scene_id,
        "coord_crs": "epsg:4326",
        "dst_crs": "epsg:4326",
        "resampling": "bilinear",
        "return_mask": "true",
    }
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key not in overrides
    ]
    query.extend(overrides.items())
    return urlunsplit((parsed.scheme, parsed.netloc, path, urlencode(query), ""))


def bbox_expression_url(
    scene: SceneCandidate,
    bbox: tuple[float, float, float, float],
    assets: tuple[str, ...],
    expression: str,
    rescale: str = "-1,1",
    colormap_name: str = "rdylgn",
    width: int = 960,
    height: int = 640,
) -> str:
    if not scene.rendered_preview_url:
        raise ImageryError(f"La escena {scene.scene_id} no ofrece un render compatible.")
    min_x, min_y, max_x, max_y = padded_bbox(bbox)
    parsed = urlsplit(scene.rendered_preview_url)
    prefix = parsed.path.split("/item/")[0]
    path = (
        f"{prefix}/item/bbox/{min_x:.7f},{min_y:.7f},{max_x:.7f},{max_y:.7f}/{width}x{height}.png"
    )
    query: list[tuple[str, str]] = [
        ("collection", scene.collection),
        ("item", scene.scene_id),
        *(("assets", asset) for asset in assets),
        ("expression", expression),
        ("asset_as_band", "true"),
        ("rescale", rescale),
        ("colormap_name", colormap_name),
        ("coord_crs", "epsg:4326"),
        ("dst_crs", "epsg:4326"),
        ("resampling", "bilinear"),
        ("return_mask", "true"),
    ]
    return urlunsplit((parsed.scheme, parsed.netloc, path, urlencode(query), ""))


def fetch_scene_image(
    scene: SceneCandidate,
    bbox: tuple[float, float, float, float],
    timeout_seconds: int = 90,
) -> bytes:
    response = requests.get(
        bbox_image_url(scene, bbox),
        timeout=timeout_seconds,
        headers={"User-Agent": "ceibos-damage-assessor/0.2"},
    )
    if response.status_code != 200:
        detail = response.text[:300] if response.text else response.reason
        raise ImageryError(
            f"Planetary Computer no pudo renderizar {scene.scene_id}: "
            f"HTTP {response.status_code} · {detail}"
        )
    try:
        image = Image.open(io.BytesIO(response.content))
        image.load()
    except Exception as exc:
        raise ImageryError("El proveedor respondió, pero no devolvió una imagen válida.") from exc
    if image.width < 100 or image.height < 100:
        raise ImageryError("La imagen recibida es demasiado pequeña para comparar.")
    return response.content


def fetch_expression_image(
    scene: SceneCandidate,
    bbox: tuple[float, float, float, float],
    *,
    assets: tuple[str, ...],
    expression: str,
    rescale: str = "-1,1",
    colormap_name: str = "rdylgn",
    timeout_seconds: int = 90,
) -> bytes:
    url = bbox_expression_url(
        scene,
        bbox,
        assets,
        expression,
        rescale=rescale,
        colormap_name=colormap_name,
    )
    response = requests.get(
        url,
        timeout=timeout_seconds,
        headers={"User-Agent": "ceibos-damage-assessor/0.3"},
    )
    if response.status_code != 200:
        detail = response.text[:300] if response.text else response.reason
        raise ImageryError(
            f"Planetary Computer no pudo calcular el índice para {scene.scene_id}: "
            f"HTTP {response.status_code} · {detail}"
        )
    try:
        image = Image.open(io.BytesIO(response.content))
        image.load()
    except Exception as exc:
        raise ImageryError("El proveedor no devolvió una imagen de índice válida.") from exc
    if image.width < 100 or image.height < 100:
        raise ImageryError("La imagen de índice recibida es demasiado pequeña para comparar.")
    return response.content
