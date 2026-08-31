from __future__ import annotations

import csv
import html
import io
import json
import zipfile
from datetime import UTC, date, datetime
from typing import Any

from .analysis import RealChangeResult
from .catalog import ScenePair


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    buffer = io.StringIO()
    fieldnames = list(rows[0]) if rows else []
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def _report_html(
    event_date: date,
    sensor: str,
    pair: ScenePair,
    result: RealChangeResult | None,
    field_observations: list[dict[str, Any]],
) -> bytes:
    metrics = result.summary if result else {"estado": "Sin métrica de cambio para esta capa"}
    metric_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in metrics.items()
    )
    field_rows = "".join(
        f"<tr><td>{html.escape(str(row.get('pedido', '')))}</td>"
        f"<td>{html.escape(str(row.get('escala_dano_0_10', '')))}</td>"
        f"<td>{html.escape(str(row.get('clase_severidad', '')))}</td></tr>"
        for row in field_observations
    )
    document = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>Evaluación de daño agrícola</title>
<style>body{{font-family:Arial,sans-serif;max-width:1100px;margin:40px auto;color:#17313b}}
h1,h2{{color:#9b3a32}} .note{{background:#fff2cc;border-left:6px solid #d99b00;padding:14px}}
table{{border-collapse:collapse;width:100%;margin:16px 0}} th,td{{border-bottom:1px solid #ddd;padding:8px;text-align:left}}
th{{background:#f5f2ef}} img{{max-width:100%;border:1px solid #ddd}}</style></head>
<body><h1>Evaluación de daño agrícola</h1>
<p><strong>Evento:</strong> granizo · <strong>Fecha:</strong> {event_date.isoformat()} · <strong>Sensor:</strong> {html.escape(sensor)}</p>
<p><strong>Anterior:</strong> {html.escape(pair.before.scene_id)} · {pair.before.acquired_at.isoformat()}<br>
<strong>Posterior:</strong> {html.escape(pair.after.scene_id)} · {pair.after.acquired_at.isoformat()}</p>
<div class="note"><strong>Resultado preliminar.</strong> El área de cambio radar es un screening robusto no calibrado como daño; la severidad satelital permanece sin calibrar.</div>
<h2>Métricas</h2><table>{metric_rows}</table>
<h2>Antes / después</h2><p><img src="before.png" alt="Escena anterior"></p><p><img src="after.png" alt="Escena posterior"></p>
<h2>Severidad observada en campo</h2><table><tr><th>Lote</th><th>Score 0–10</th><th>Clase</th></tr>{field_rows}</table>
</body></html>"""
    return document.encode("utf-8")


def build_analysis_package(
    geojson: dict[str, Any],
    event_date: date,
    sensor: str,
    pair: ScenePair,
    before_png: bytes,
    after_png: bytes,
    result: RealChangeResult | None,
    field_observations: list[dict[str, Any]],
) -> bytes:
    buffer = io.BytesIO()
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "event_type": "hail",
        "event_date": event_date.isoformat(),
        "analysis_status": "SATELLITE_DATA_PRELIMINARY_NOT_VALIDATED",
        "sensor": sensor,
        "before_scene": pair.before.to_row(),
        "after_scene": pair.after.to_row(),
        "summary": result.summary if result else None,
        "severity_status": "FIELD_OBSERVED_ONLY_SATELLITE_MODEL_NOT_CALIBRATED",
    }
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "report.html", _report_html(event_date, sensor, pair, result, field_observations)
        )
        archive.writestr("before.png", before_png)
        archive.writestr("after.png", after_png)
        archive.writestr("selection.geojson", json.dumps(geojson, ensure_ascii=False, indent=2))
        archive.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))
        archive.writestr("field_severity.csv", _csv_bytes(field_observations))
        if result:
            archive.writestr("radar_change_screening.png", result.change_png)
            archive.writestr("change_by_lot.csv", _csv_bytes(result.lot_summaries))
        archive.writestr(
            "README.txt",
            "Datos satelitales. Resultado preliminar no validado. "
            "La severidad satelital no está calibrada; no usar como peritaje automático.\n",
        )
    return buffer.getvalue()
