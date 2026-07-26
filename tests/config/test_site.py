"""Tests for typed solar-site configuration."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from solarpulse_ai.config.site import SiteConfig, load_site_config
from solarpulse_ai.data.errors import SiteConfigurationError


def _site_payload() -> dict[str, object]:
    return {
        "site_id": "example-site",
        "latitude": -6.7924,
        "longitude": 39.2083,
        "timezone": "Africa/Dar_es_Salaam",
        "installed_capacity_kwp": 10.0,
        "panel_tilt_degrees": 10.0,
        "panel_azimuth_degrees": 0.0,
    }


def test_valid_site_configuration_loads_from_json(tmp_path: Path) -> None:
    """A valid illustrative configuration is typed and loaded."""
    path = tmp_path / "site.json"
    path.write_text(json.dumps(_site_payload()), encoding="utf-8")

    site = load_site_config(path)

    assert site.site_id == "example-site"
    assert site.timezone == "Africa/Dar_es_Salaam"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("site_id", " "),
        ("latitude", -91),
        ("latitude", 91),
        ("longitude", -181),
        ("longitude", 181),
        ("installed_capacity_kwp", 0),
        ("panel_tilt_degrees", -1),
        ("panel_tilt_degrees", 91),
        ("timezone", "Not/A_Timezone"),
    ],
)
def test_invalid_site_values_are_rejected(field: str, value: object) -> None:
    """Site constraints reject unsafe or impossible metadata."""
    payload = _site_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        SiteConfig.model_validate(payload)


def test_missing_and_malformed_site_files_have_clear_errors(tmp_path: Path) -> None:
    """Configuration loading wraps file and JSON failures."""
    with pytest.raises(SiteConfigurationError, match="does not exist"):
        load_site_config(tmp_path / "missing.json")

    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(SiteConfigurationError, match="not a file"):
        load_site_config(directory)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(SiteConfigurationError, match="Could not read"):
        load_site_config(malformed)


def test_invalid_json_configuration_is_wrapped(tmp_path: Path) -> None:
    """Pydantic field errors are exposed as a data-layer configuration error."""
    path = tmp_path / "site.json"
    payload = _site_payload()
    payload["latitude"] = 100
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SiteConfigurationError, match="Invalid site configuration"):
        load_site_config(path)
