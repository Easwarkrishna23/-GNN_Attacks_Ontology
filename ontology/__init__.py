"""
Ontology-guided semantic defense package.

This package implements a lightweight OWL/RDF-compatible ontology builder and a
semantic defense module designed for GNN robustness experiments in this repo.

Key idea:
- Similarity pruning uses only geometric similarity of features (e.g., cosine/Jaccard).
- Ontology defense uses explicit typed concepts, constraints, and rule-based
  contradiction handling to produce edge trust and feature repair signals.
"""

from .ontology_builder import OntologyBuilder, OntologyArtifacts, OntologyConfig
from .ontology_rules import OntologyRuleEngine, RuleReport, SWRLRule
from .ontology_defense import OntologyGuidedDefense, DefenseVariant, DefenseOutput
from .ontology_export import OntologyExporter, ExportPaths

__all__ = [
    "OntologyBuilder",
    "OntologyArtifacts",
    "OntologyConfig",
    "OntologyRuleEngine",
    "RuleReport",
    "SWRLRule",
    "OntologyGuidedDefense",
    "DefenseVariant",
    "DefenseOutput",
    "OntologyExporter",
    "ExportPaths",
]

