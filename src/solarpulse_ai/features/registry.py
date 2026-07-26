"""Site-configuration registry for multi-site feature generation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from solarpulse_ai.config.site import SiteConfig, load_site_config
from solarpulse_ai.data.errors import SiteConfigurationError


@dataclass(frozen=True, slots=True)
class SiteRegistry:
    """Validated site configurations keyed by site identifier."""

    sites: dict[str, SiteConfig]

    @classmethod
    def from_paths(cls, paths: Iterable[str | Path]) -> SiteRegistry:
        """Load paths while rejecting duplicate identifiers."""
        sites: dict[str, SiteConfig] = {}
        for path in paths:
            site = load_site_config(path)
            if site.site_id in sites:
                raise SiteConfigurationError(
                    f"Duplicate site configuration for site_id={site.site_id!r}"
                )
            sites[site.site_id] = site
        if not sites:
            raise SiteConfigurationError("At least one site configuration is required.")
        return cls(sites)

    def validate_dataset_sites(
        self, dataset_site_ids: Iterable[str], *, allow_unused: bool = False
    ) -> None:
        """Require an exact registry-to-dataset match unless unused sites are allowed."""
        dataset_sites = set(dataset_site_ids)
        configured_sites = set(self.sites)
        missing = dataset_sites - configured_sites
        unused = configured_sites - dataset_sites
        if missing:
            raise SiteConfigurationError(
                f"Missing site configuration(s): {', '.join(sorted(missing))}"
            )
        if unused and not allow_unused:
            raise SiteConfigurationError(
                f"Site configuration(s) not represented in dataset: {', '.join(sorted(unused))}"
            )

    def __getitem__(self, site_id: str) -> SiteConfig:
        """Return one site's validated configuration."""
        return self.sites[site_id]
