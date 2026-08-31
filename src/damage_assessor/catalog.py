from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

from .geometry import geojson_bbox

PLANETARY_PROVIDER_NAME = "Microsoft Planetary Computer"
PLANETARY_STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
DEFAULT_RADAR_COLLECTION = "sentinel-1-rtc"
DEFAULT_OPTICAL_COLLECTION = "sentinel-2-l2a"


@dataclass(frozen=True)
class SceneCandidate:
    scene_id: str
    collection: str
    acquired_at: datetime
    cloud_cover: float | None = None
    relative_orbit: int | None = None
    orbit_state: str | None = None
    polarizations: tuple[str, ...] = ()
    asset_keys: tuple[str, ...] = ()
    platform: str | None = None
    rendered_preview_url: str | None = None
    tilejson_url: str | None = None
    geometry: dict[str, Any] | None = None
    bbox: tuple[float, float, float, float] | None = None
    grid_code: str | None = None
    instrument_mode: str | None = None

    @property
    def acquired_date(self) -> date:
        return self.acquired_at.date()

    def to_row(self) -> dict[str, Any]:
        return {
            "escena": self.scene_id,
            "fecha_utc": self.acquired_at.isoformat(),
            "coleccion": self.collection,
            "plataforma": self.platform,
            "nubes_escena_pct": self.cloud_cover,
            "orbita_relativa": self.relative_orbit,
            "direccion": self.orbit_state,
            "polarizaciones": ", ".join(self.polarizations),
            "tesela": self.grid_code,
        }


@dataclass(frozen=True)
class SceneAcquisition:
    """All STAC items belonging to one optical acquisition or radar pass."""

    acquisition_id: str
    scenes: tuple[SceneCandidate, ...]

    @property
    def acquired_at(self) -> datetime:
        return min(scene.acquired_at for scene in self.scenes)

    @property
    def acquired_date(self) -> date:
        return self.acquired_at.date()

    @property
    def relative_orbit(self) -> int | None:
        values = {scene.relative_orbit for scene in self.scenes}
        return values.pop() if len(values) == 1 else None

    @property
    def orbit_state(self) -> str | None:
        values = {scene.orbit_state for scene in self.scenes}
        return values.pop() if len(values) == 1 else None

    @property
    def cloud_cover(self) -> float | None:
        values = [scene.cloud_cover for scene in self.scenes if scene.cloud_cover is not None]
        return max(values) if values else None

    @property
    def scene_id(self) -> str:
        return ",".join(scene.scene_id for scene in self.scenes)

    @property
    def tile_count(self) -> int:
        return len(self.scenes)


@dataclass(frozen=True)
class ScenePair:
    before: SceneCandidate
    after: SceneCandidate
    comparable: bool
    reason: str


def _utc_midnight(day: date) -> datetime:
    return datetime.combine(day, time.min, tzinfo=UTC)


def select_optical_pair(
    scenes: Iterable[SceneCandidate], event_date: date, max_scene_cloud: float = 60.0
) -> ScenePair | None:
    event = _utc_midnight(event_date)
    eligible = [
        scene
        for scene in scenes
        if scene.cloud_cover is None or scene.cloud_cover <= max_scene_cloud
    ]
    before = sorted(
        (scene for scene in eligible if scene.acquired_at < event),
        key=lambda scene: (event - scene.acquired_at, scene.cloud_cover or 0),
    )
    after = sorted(
        (scene for scene in eligible if scene.acquired_at >= event),
        key=lambda scene: (scene.acquired_at - event, scene.cloud_cover or 0),
    )
    if not before or not after:
        return None
    return ScenePair(
        before=before[0],
        after=after[0],
        comparable=True,
        reason="Par preliminar; falta verificar nubosidad y sombras dentro de cada lote.",
    )


def select_radar_pair(scenes: Iterable[SceneCandidate], event_date: date) -> ScenePair | None:
    event = _utc_midnight(event_date)
    before = [scene for scene in scenes if scene.acquired_at < event]
    after = [scene for scene in scenes if scene.acquired_at >= event]
    if not before or not after:
        return None

    comparable_pairs: list[tuple[timedelta, SceneCandidate, SceneCandidate]] = []
    for pre in before:
        for post in after:
            same_orbit = (
                pre.relative_orbit is not None
                and pre.relative_orbit == post.relative_orbit
                and pre.orbit_state == post.orbit_state
            )
            same_pol = (
                not pre.polarizations
                or not post.polarizations
                or pre.polarizations == post.polarizations
            )
            if same_orbit and same_pol:
                distance = (event - pre.acquired_at) + (post.acquired_at - event)
                comparable_pairs.append((distance, pre, post))

    if comparable_pairs:
        _, pre, post = min(comparable_pairs, key=lambda item: item[0])
        return ScenePair(
            before=pre,
            after=post,
            comparable=True,
            reason="Misma órbita relativa, dirección y polarización disponible.",
        )

    pre = min(before, key=lambda scene: event - scene.acquired_at)
    post = min(after, key=lambda scene: scene.acquired_at - event)
    return ScenePair(
        before=pre,
        after=post,
        comparable=False,
        reason="No se encontró un par de la misma órbita; no debe usarse para cuantificar cambio.",
    )


def _scene_from_item(item: Any) -> SceneCandidate:
    properties = item.properties
    acquired = item.datetime or properties.get("datetime")
    if isinstance(acquired, str):
        acquired = datetime.fromisoformat(acquired)
    if acquired.tzinfo is None:
        acquired = acquired.replace(tzinfo=UTC)
    polarizations = tuple(
        str(value).upper() for value in properties.get("sar:polarizations", []) or []
    )
    rendered_preview = item.assets.get("rendered_preview")
    tilejson = item.assets.get("tilejson")
    item_bbox = tuple(float(value) for value in item.bbox) if item.bbox else None
    grid_value = properties.get("s2:mgrs_tile") or properties.get("mgrs:utm_zone")
    return SceneCandidate(
        scene_id=item.id,
        collection=item.collection_id or "",
        acquired_at=acquired,
        cloud_cover=properties.get("eo:cloud_cover"),
        relative_orbit=properties.get("sat:relative_orbit"),
        orbit_state=properties.get("sat:orbit_state"),
        polarizations=polarizations,
        asset_keys=tuple(sorted(item.assets)),
        platform=properties.get("platform"),
        rendered_preview_url=rendered_preview.href if rendered_preview else None,
        tilejson_url=tilejson.href if tilejson else None,
        geometry=item.geometry,
        bbox=item_bbox,
        grid_code=str(grid_value) if grid_value is not None else None,
        instrument_mode=properties.get("sar:instrument_mode"),
    )


def group_scene_acquisitions(
    scenes: Iterable[SceneCandidate], sensor: str
) -> list[SceneAcquisition]:
    """Group adjacent tiles/slices from the same acquisition into one selectable mosaic."""
    grouped: dict[str, list[SceneCandidate]] = {}
    for scene in scenes:
        if sensor == "radar":
            key = "|".join(
                [
                    scene.acquired_at.strftime("%Y-%m-%d"),
                    str(scene.relative_orbit or ""),
                    str(scene.orbit_state or ""),
                    ",".join(scene.polarizations),
                ]
            )
        else:
            # Adjacent Sentinel-2 MGRS tiles from the same overpass can differ by a
            # few seconds in STAC. Minute + relative orbit identifies that overpass.
            key = "|".join(
                [
                    scene.acquired_at.strftime("%Y-%m-%dT%H:%M"),
                    str(scene.relative_orbit or ""),
                ]
            )
        grouped.setdefault(key, []).append(scene)
    return sorted(
        (
            SceneAcquisition(key, tuple(sorted(items, key=lambda item: item.scene_id)))
            for key, items in grouped.items()
        ),
        key=lambda group: group.acquired_at,
    )


def _point_in_ring(x: float, y: float, ring: list[list[float]]) -> bool:
    inside = False
    previous = ring[-1]
    for current in ring:
        x1, y1 = float(previous[0]), float(previous[1])
        x2, y2 = float(current[0]), float(current[1])
        if (y1 > y) != (y2 > y):
            crossing = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing:
                inside = not inside
        previous = current
    return inside


def _point_in_geometry(x: float, y: float, geometry: dict[str, Any]) -> bool:
    polygons = (
        [geometry.get("coordinates", [])]
        if geometry.get("type") == "Polygon"
        else geometry.get("coordinates", [])
    )
    for polygon in polygons:
        if (
            polygon
            and _point_in_ring(x, y, polygon[0])
            and not any(_point_in_ring(x, y, hole) for hole in polygon[1:])
        ):
            return True
    return False


def acquisition_coverage_pct(
    acquisition: SceneAcquisition, geojson: dict[str, Any], samples: int = 48
) -> float:
    """Estimate AOI coverage, including mosaics that span multiple STAC items."""
    min_x, min_y, max_x, max_y = geojson_bbox(geojson)
    inside = 0
    covered = 0
    for row in range(samples):
        y = min_y + (row + 0.5) / samples * (max_y - min_y)
        for column in range(samples):
            x = min_x + (column + 0.5) / samples * (max_x - min_x)
            if not any(
                _point_in_geometry(x, y, feature["geometry"]) for feature in geojson["features"]
            ):
                continue
            inside += 1
            for scene in acquisition.scenes:
                if scene.geometry and _point_in_geometry(x, y, scene.geometry):
                    covered += 1
                    break
                if scene.bbox:
                    sx1, sy1, sx2, sy2 = scene.bbox
                    if sx1 <= x <= sx2 and sy1 <= y <= sy2:
                        covered += 1
                        break
    return round(100.0 * covered / inside, 1) if inside else 0.0


def search_scene_metadata(
    geojson: dict[str, Any],
    event_date: date,
    days_before: int,
    days_after: int,
    collection: str,
    endpoint: str | None = None,
    max_items: int = 100,
) -> list[SceneCandidate]:
    """Search public STAC metadata. Asset signing/loading happens in a separate processing layer."""
    try:
        from pystac_client import Client
    except ImportError as exc:
        raise RuntimeError("La dependencia pystac-client no está instalada.") from exc

    start = event_date - timedelta(days=days_before)
    end = event_date + timedelta(days=days_after)
    api_url = endpoint or os.getenv("STAC_API_URL", PLANETARY_STAC_API_URL)
    client = Client.open(api_url)
    search = client.search(
        collections=[collection],
        bbox=geojson_bbox(geojson),
        datetime=f"{start.isoformat()}/{end.isoformat()}",
        max_items=max_items,
    )
    return sorted(
        (_scene_from_item(item) for item in search.items()), key=lambda scene: scene.acquired_at
    )
