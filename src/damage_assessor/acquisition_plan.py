from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time
from urllib.parse import urljoin

import requests

from .geometry import geojson_bbox

PLAN_URL = "https://sentinels.copernicus.eu/copernicus/sentinel-1/acquisition-plans"


@dataclass(frozen=True)
class Sentinel1Pass:
    satellite: str
    starts_at: datetime
    ends_at: datetime
    mode: str
    polarization: str
    relative_orbit: int | None
    plan_url: str

    def to_row(self, now: datetime | None = None) -> dict[str, object]:
        reference = now or datetime.now(UTC)
        days = max(0, math.ceil((self.starts_at - reference).total_seconds() / 86_400))
        row = asdict(self)
        row["starts_at"] = self.starts_at.isoformat()
        row["ends_at"] = self.ends_at.isoformat()
        row["days_until"] = days
        return row


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
    return inside


def _footprint_intersects_bbox(
    footprint: list[tuple[float, float]], bbox: tuple[float, float, float, float]
) -> bool:
    min_x, min_y, max_x, max_y = bbox
    fp_x = [point[0] for point in footprint]
    fp_y = [point[1] for point in footprint]
    if max(fp_x) < min_x or min(fp_x) > max_x or max(fp_y) < min_y or min(fp_y) > max_y:
        return False
    corners = [(min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y)]
    center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
    return any(_point_in_polygon(point, footprint) for point in [*corners, center]) or any(
        min_x <= x <= max_x and min_y <= y <= max_y for x, y in footprint
    )


def _current_plan_links(html: str, cutoff: datetime) -> list[str]:
    paths = set(
        re.findall(
            r'href="([^"]*/s1[cd]_mp_user_(\d{8}t\d{6})_(\d{8}t\d{6}))"',
            html,
            flags=re.IGNORECASE,
        )
    )
    candidates: list[tuple[datetime, datetime, str, str]] = []
    for path, start_text, end_text in paths:
        start = datetime.strptime(start_text.upper(), "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
        end = datetime.strptime(end_text.upper(), "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
        satellite = "S1C" if "s1c_" in path.lower() else "S1D"
        if end >= cutoff:
            candidates.append((start, end, satellite, path))
    links: list[str] = []
    for satellite in ("S1C", "S1D"):
        matching = [candidate for candidate in candidates if candidate[2] == satellite]
        if matching:
            links.append(max(matching, key=lambda item: (item[0], item[1]))[3])
    return links


def _passes_from_kml(
    payload: bytes,
    plan_url: str,
    bbox: tuple[float, float, float, float],
    cutoff: datetime,
) -> list[Sentinel1Pass]:
    root = ET.fromstring(payload)
    namespace = {"k": "http://www.opengis.net/kml/2.2"}
    passes: list[Sentinel1Pass] = []
    for placemark in root.findall(".//k:Placemark", namespace):
        data = {
            node.attrib.get("name", ""): (node.findtext("k:value", default="", namespaces=namespace))
            for node in placemark.findall(".//k:Data", namespace)
        }
        if data.get("Mode") != "IW":
            continue
        coordinate_text = placemark.findtext(".//k:coordinates", default="", namespaces=namespace)
        footprint = [
            (float(parts[0]), float(parts[1]))
            for token in coordinate_text.split()
            if len(parts := token.split(",")) >= 2
        ]
        if len(footprint) < 3 or not _footprint_intersects_bbox(footprint, bbox):
            continue
        starts_at = datetime.fromisoformat(data["ObservationTimeStart"]).replace(tzinfo=UTC)
        ends_at = datetime.fromisoformat(data["ObservationTimeStop"]).replace(tzinfo=UTC)
        if starts_at < cutoff:
            continue
        relative_orbit = int(data["OrbitRelative"]) if data.get("OrbitRelative") else None
        passes.append(
            Sentinel1Pass(
                satellite=data.get("SatelliteId", "Sentinel-1"),
                starts_at=starts_at,
                ends_at=ends_at,
                mode=data.get("Mode", "IW"),
                polarization=data.get("Polarisation", ""),
                relative_orbit=relative_orbit,
                plan_url=plan_url,
            )
        )
    return passes


def next_sentinel1_pass(
    geojson: dict[str, object], event_date: date, timeout_seconds: int = 30
) -> Sentinel1Pass | None:
    now = datetime.now(UTC)
    event_cutoff = datetime.combine(event_date, time.min, tzinfo=UTC)
    cutoff = max(now, event_cutoff)
    response = requests.get(
        PLAN_URL,
        timeout=timeout_seconds,
        headers={"User-Agent": "ceibos-damage-assessor/0.2"},
    )
    response.raise_for_status()
    passes: list[Sentinel1Pass] = []
    for path in _current_plan_links(response.text, cutoff):
        url = urljoin(PLAN_URL, path)
        plan_response = requests.get(
            url,
            timeout=timeout_seconds,
            headers={"User-Agent": "ceibos-damage-assessor/0.2"},
        )
        plan_response.raise_for_status()
        passes.extend(_passes_from_kml(plan_response.content, url, geojson_bbox(geojson), cutoff))
    return min(passes, key=lambda item: item.starts_at) if passes else None
