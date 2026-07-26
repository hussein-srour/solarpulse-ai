"""Canonical schema definitions for hourly solar and weather observations."""

from dataclasses import dataclass
from datetime import datetime
from typing import NotRequired, TypedDict


class HourlyRecord(TypedDict):
    """Typed representation of one canonical hourly observation."""

    timestamp: datetime
    site_id: str
    ac_energy_kwh: float
    ghi_w_m2: float
    ambient_temperature_c: float
    cloud_cover_pct: float
    relative_humidity_pct: float
    wind_speed_m_s: float
    dni_w_m2: NotRequired[float]
    dhi_w_m2: NotRequired[float]
    module_temperature_c: NotRequired[float]
    precipitation_mm: NotRequired[float]
    inverter_availability_pct: NotRequired[float]


class MeasuredGenerationRecord(TypedDict):
    """Typed representation of measured plant generation."""

    timestamp: datetime
    site_id: str
    ac_energy_kwh: float


@dataclass(frozen=True, slots=True)
class FieldDefinition:
    """Machine-readable definition of a canonical dataset field."""

    name: str
    required: bool
    data_type: str
    unit: str | None
    minimum: float | None = None
    maximum: float | None = None


FIELD_DEFINITIONS: tuple[FieldDefinition, ...] = (
    FieldDefinition("timestamp", True, "UTC datetime", None),
    FieldDefinition("site_id", True, "string", None),
    FieldDefinition("ac_energy_kwh", True, "float", "kWh", minimum=0),
    FieldDefinition("ghi_w_m2", True, "float", "W/m²", minimum=0),
    FieldDefinition("ambient_temperature_c", True, "float", "°C"),
    FieldDefinition("cloud_cover_pct", True, "float", "%", minimum=0, maximum=100),
    FieldDefinition("relative_humidity_pct", True, "float", "%", minimum=0, maximum=100),
    FieldDefinition("wind_speed_m_s", True, "float", "m/s", minimum=0),
    FieldDefinition("dni_w_m2", False, "float", "W/m²", minimum=0),
    FieldDefinition("dhi_w_m2", False, "float", "W/m²", minimum=0),
    FieldDefinition("module_temperature_c", False, "float", "°C"),
    FieldDefinition("precipitation_mm", False, "float", "mm", minimum=0),
    FieldDefinition(
        "inverter_availability_pct",
        False,
        "float",
        "%",
        minimum=0,
        maximum=100,
    ),
)

REQUIRED_COLUMNS: tuple[str, ...] = tuple(
    field.name for field in FIELD_DEFINITIONS if field.required
)
OPTIONAL_COLUMNS: tuple[str, ...] = tuple(
    field.name for field in FIELD_DEFINITIONS if not field.required
)
CANONICAL_COLUMNS: tuple[str, ...] = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
NUMERIC_FIELDS: tuple[FieldDefinition, ...] = tuple(
    field for field in FIELD_DEFINITIONS if field.data_type == "float"
)
