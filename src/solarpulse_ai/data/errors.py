"""Actionable errors raised by the SolarPulse data layer."""

from dataclasses import dataclass


class DataLayerError(Exception):
    """Base class for expected ingestion and validation failures."""


class CSVIngestionError(DataLayerError):
    """Raised when a CSV cannot be located, read, or written."""


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One validation problem, optionally linked to a column and CSV rows."""

    code: str
    message: str
    column: str | None = None
    rows: tuple[int, ...] = ()

    def __str__(self) -> str:
        """Render the issue in a concise, actionable form."""
        context: list[str] = []
        if self.column is not None:
            context.append(f"column={self.column}")
        if self.rows:
            context.append("CSV rows=" + ", ".join(str(row) for row in self.rows))
        suffix = f" ({'; '.join(context)})" if context else ""
        return f"[{self.code}] {self.message}{suffix}"


class DataValidationError(DataLayerError):
    """Raised when one or more canonical schema rules fail."""

    def __init__(self, issues: list[ValidationIssue]) -> None:
        """Store all discovered issues and construct a readable report."""
        self.issues = tuple(issues)
        report = "\n".join(f"- {issue}" for issue in self.issues)
        super().__init__(f"Hourly dataset validation failed:\n{report}")
