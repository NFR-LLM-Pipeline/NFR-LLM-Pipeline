from dataclasses import dataclass, field

# Data models

@dataclass
class Requirement:
    """One requirement from the PROMISE dataset"""
    req_id: str
    text: str
    project_id: str
    is_nfr: bool
    category: str  # mapped gold label: F / performance / security / maintainability / other

@dataclass
class IdentificationResult:
    """Prediction for whether a requirement is an NFR."""
    req_id: str
    predicted_is_nfr: bool

@dataclass
class ClassificationResult:
    """Prediction for the NFR category of a requirement."""
    req_id: str
    predicted_category: str

@dataclass
class PipelineResult:
    """Full pipeline output for a single project."""
    project_id: str
    identification: list[IdentificationResult] = field(default_factory=list)
    classification: list[ClassificationResult] = field(default_factory=list)