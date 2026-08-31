from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import requests

POWER_DAILY_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_PARAMETERS = ("PRECTOTCORR", "GWETTOP")


@dataclass(frozen=True)
class PowerDailyObservation:
    day: date
    precipitation_mm: float | None
    surface_soil_wetness: float | None


@dataclass(frozen=True)
class PowerDailySeries:
    latitude: float
    longitude: float
    observations: tuple[PowerDailyObservation, ...]
    time_standard: str = "LST"

    def precipitation_before(self, reference_day: date, days: int) -> float | None:
        """Sum complete calendar days preceding a reference date."""
        if days < 1:
            raise ValueError("El período de acumulación debe ser mayor que cero.")
        by_day = {observation.day: observation for observation in self.observations}
        expected_days = [reference_day - timedelta(days=offset) for offset in range(1, days + 1)]
        values = [
            by_day[day].precipitation_mm
            for day in expected_days
            if day in by_day and by_day[day].precipitation_mm is not None
        ]
        return round(sum(values), 2) if len(values) == days else None

    def surface_wetness_on_or_before(
        self, reference_day: date, lookback_days: int = 2
    ) -> tuple[float, date] | None:
        """Return the latest available surface wetness without looking into the future."""
        by_day = {observation.day: observation for observation in self.observations}
        for offset in range(lookback_days + 1):
            day = reference_day - timedelta(days=offset)
            observation = by_day.get(day)
            if observation is not None and observation.surface_soil_wetness is not None:
                return observation.surface_soil_wetness, day
        return None


def _valid_number(value: Any, *, minimum: float, maximum: float | None = None) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric < minimum or (maximum is not None and numeric > maximum):
        return None
    return numeric


def fetch_power_daily(
    latitude: float,
    longitude: float,
    start: date,
    end: date,
    *,
    timeout: int = 30,
) -> PowerDailySeries:
    """Fetch daily precipitation and surface soil wetness from NASA POWER."""
    if start > end:
        raise ValueError("La fecha inicial no puede ser posterior a la fecha final.")
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("Las coordenadas del punto meteorológico no son válidas.")

    response = requests.get(
        POWER_DAILY_URL,
        params={
            "parameters": ",".join(POWER_PARAMETERS),
            "community": "AG",
            "longitude": longitude,
            "latitude": latitude,
            "start": start.strftime("%Y%m%d"),
            "end": end.strftime("%Y%m%d"),
            "format": "JSON",
            "time-standard": "LST",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()

    parameter_data = (payload.get("properties") or {}).get("parameter") or {}
    precipitation = parameter_data.get("PRECTOTCORR") or {}
    surface_wetness = parameter_data.get("GWETTOP") or {}
    day_keys = sorted(set(precipitation) | set(surface_wetness))
    observations = tuple(
        PowerDailyObservation(
            day=date.fromisoformat(f"{day_key[:4]}-{day_key[4:6]}-{day_key[6:8]}"),
            precipitation_mm=_valid_number(precipitation.get(day_key), minimum=0),
            surface_soil_wetness=_valid_number(surface_wetness.get(day_key), minimum=0, maximum=1),
        )
        for day_key in day_keys
        if len(day_key) == 8 and day_key.isdigit()
    )
    if not observations:
        messages = payload.get("messages") or []
        detail = f" ({'; '.join(str(message) for message in messages)})" if messages else ""
        raise ValueError(f"NASA POWER no devolvió observaciones para el período{detail}.")

    coordinates = (payload.get("geometry") or {}).get("coordinates") or []
    resolved_longitude = float(coordinates[0]) if len(coordinates) >= 2 else longitude
    resolved_latitude = float(coordinates[1]) if len(coordinates) >= 2 else latitude
    time_standard = str((payload.get("header") or {}).get("time_standard") or "LST")
    return PowerDailySeries(
        latitude=resolved_latitude,
        longitude=resolved_longitude,
        observations=observations,
        time_standard=time_standard,
    )
