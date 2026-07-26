"""Typed, non-secret configuration for a solar installation."""

import json
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from solarpulse_ai.data.errors import SiteConfigurationError


class SiteConfig(BaseModel):
    """Location and physical metadata needed by weather integrations."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        str_strip_whitespace=True,
    )

    site_id: str = Field(min_length=1)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str
    installed_capacity_kwp: float = Field(gt=0)
    panel_tilt_degrees: float = Field(ge=0, le=90)
    panel_azimuth_degrees: float

    @field_validator("timezone")
    @classmethod
    def timezone_must_be_iana(cls, value: str) -> str:
        """Require a timezone available in the system IANA database."""
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as error:
            raise ValueError("timezone must be a valid IANA timezone") from error
        return value


def load_site_config(path: str | Path) -> SiteConfig:
    """Load and validate a JSON site configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise SiteConfigurationError(f"Site configuration does not exist: {config_path}")
    if not config_path.is_file():
        raise SiteConfigurationError(f"Site configuration path is not a file: {config_path}")

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SiteConfigurationError(
            f"Could not read site configuration {config_path}: {error}"
        ) from error

    try:
        return SiteConfig.model_validate(payload)
    except ValidationError as error:
        raise SiteConfigurationError(
            f"Invalid site configuration {config_path}: {error}"
        ) from error
