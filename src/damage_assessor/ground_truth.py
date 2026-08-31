from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .severity import severity_class


def load_ground_truth(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    output: list[dict[str, Any]] = []
    for row in rows:
        score = float(row["escala_dano_0_10"])
        output.append(
            {
                **row,
                "escala_dano_0_10": score,
                "clase_severidad": severity_class(score),
                "superficie_sembrada_ha": float(row["superficie_sembrada_ha"]),
                "ha_dano": float(row["ha_dano"]),
            }
        )
    return output


def rows_for_geojson(
    rows: list[dict[str, Any]], geojson: dict[str, Any]
) -> list[dict[str, Any]]:
    requested = {
        str((feature.get("properties") or {}).get("pedido"))
        for feature in geojson["features"]
    }
    return [row for row in rows if str(row.get("pedido")) in requested]
