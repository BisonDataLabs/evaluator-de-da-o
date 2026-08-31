from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeverityBand:
    code: str
    label: str
    min_exclusive: float
    max_inclusive: float


SEVERITY_BANDS = (
    SeverityBand("sin_dano", "Sin daño", -0.001, 0.0),
    SeverityBand("leve", "Leve", 0.0, 2.0),
    SeverityBand("moderado", "Moderado", 2.0, 5.0),
    SeverityBand("severo", "Severo", 5.0, 8.0),
    SeverityBand("extremo", "Extremo", 8.0, 10.0),
)


def severity_class(score_0_10: float) -> str:
    """Map a continuous 0–10 score to the initial five-class operational scheme."""
    score = float(score_0_10)
    if not 0 <= score <= 10:
        raise ValueError("La severidad debe estar entre 0 y 10.")
    for band in SEVERITY_BANDS:
        if band.min_exclusive < score <= band.max_inclusive:
            return band.label
    raise RuntimeError("No se pudo clasificar la severidad.")


def severity_scheme_rows() -> list[dict[str, str]]:
    return [
        {"rango_0_10": "0", "clase": "Sin daño", "uso": "Sin evidencia observada de daño"},
        {"rango_0_10": "1–2", "clase": "Leve", "uso": "Daño visible de baja magnitud"},
        {"rango_0_10": "3–5", "clase": "Moderado", "uso": "Daño claro con afectación funcional parcial"},
        {"rango_0_10": "6–8", "clase": "Severo", "uso": "Daño extendido o pérdida funcional importante"},
        {"rango_0_10": "9–10", "clase": "Extremo", "uso": "Daño generalizado o pérdida funcional dominante"},
    ]
