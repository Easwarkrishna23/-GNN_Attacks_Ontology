"""
ontology_export.py

Exports ontology artifacts to Protégé-compatible formats:
- ontology.owl (RDF/XML serialization of OWL vocabulary)
- ontology.rdf (RDF/XML)
- ontology.ttl (Turtle)
- ontology.swrl (SWRL-like rule text for documentation)

We export a compact ontology:
- Classes: ResearchTopic, Feature, and per-dataset topic/subtopic classes
- Object properties: indicatesTopic, coOccursWith, contradicts
- Data properties: affinityScore, cooccurScore, inheritanceScore
- Individuals: Feature_<id> individuals (optional; enabled by default)

Note: SWRL rules can be embedded in OWL using the SWRL vocabulary, but that
requires a more verbose RDF encoding. For production work, you can extend this
exporter to encode rules as proper swrl:Imp individuals. Here we also export a
human-readable `.swrl` file to keep it simple and reviewable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .ontology_builder import OntologyArtifacts
from .ontology_rules import OntologyRuleEngine


@dataclass(frozen=True)
class ExportPaths:
    owl: str
    rdf: str
    ttl: str
    swrl: str


class OntologyExporter:
    def __init__(self, base_iri: str):
        if not base_iri.endswith("#"):
            base_iri = base_iri + "#"
        self.base_iri = base_iri

    def _require_rdflib(self):
        try:
            import rdflib  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "rdflib is required for OWL/RDF export. Install with `pip install rdflib`."
            ) from e

    def export_all(
        self,
        artifacts: OntologyArtifacts,
        out_dir: str,
        rule_engine: Optional[OntologyRuleEngine] = None,
        include_feature_individuals: bool = True,
    ) -> ExportPaths:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        owl_path = str(out / "ontology.owl")
        rdf_path = str(out / "ontology.rdf")
        ttl_path = str(out / "ontology.ttl")
        swrl_path = str(out / "ontology.swrl")

        self.export_owl(artifacts, owl_path, include_feature_individuals=include_feature_individuals)
        self.export_rdf(artifacts, rdf_path, include_feature_individuals=include_feature_individuals)
        self.export_turtle(artifacts, ttl_path, include_feature_individuals=include_feature_individuals)
        self.export_swrl_rules(artifacts, swrl_path, rule_engine=rule_engine)

        return ExportPaths(owl=owl_path, rdf=rdf_path, ttl=ttl_path, swrl=swrl_path)

    def _build_graph(self, artifacts: OntologyArtifacts, include_feature_individuals: bool):
        self._require_rdflib()
        from rdflib import Graph, Literal, Namespace, RDF, RDFS, OWL, XSD

        g = Graph()
        EX = Namespace(self.base_iri)

        g.bind("ex", EX)
        g.bind("owl", OWL)
        g.bind("rdfs", RDFS)
        g.bind("rdf", RDF)
        g.bind("xsd", XSD)

        # Base classes
        ResearchTopic = EX.ResearchTopic
        Feature = EX.Feature
        g.add((ResearchTopic, RDF.type, OWL.Class))
        g.add((Feature, RDF.type, OWL.Class))

        # Topic classes + subtopics
        for cname in artifacts.class_names:
            c = EX[cname]
            g.add((c, RDF.type, OWL.Class))
            g.add((c, RDFS.subClassOf, ResearchTopic))
            for sub in artifacts.topic_hierarchy.get(cname, []):
                s = EX[sub]
                g.add((s, RDF.type, OWL.Class))
                g.add((s, RDFS.subClassOf, c))
                if sub in artifacts.inheritance_score:
                    g.add((s, EX.inheritanceScore, Literal(float(artifacts.inheritance_score[sub]), datatype=XSD.float)))

        # Object properties
        indicatesTopic = EX.indicatesTopic
        coOccursWith = EX.coOccursWith
        contradicts = EX.contradicts
        for p in [indicatesTopic, coOccursWith, contradicts]:
            g.add((p, RDF.type, OWL.ObjectProperty))

        # Data properties
        affinityScore = EX.affinityScore
        cooccurScore = EX.cooccurScore
        for p in [affinityScore, cooccurScore, EX.inheritanceScore]:
            g.add((p, RDF.type, OWL.DatatypeProperty))

        # Feature individuals and their indicatesTopic mapping (top topic per feature)
        if include_feature_individuals:
            A = artifacts.feature_class_affinity  # (F,C)
            for fi, fname in enumerate(artifacts.feature_names):
                ind = EX[f"Feature_{fi}"]
                g.add((ind, RDF.type, Feature))
                g.add((ind, RDFS.label, Literal(fname)))
                # link to top-1 topic with score (for compactness)
                c = int(A[fi].argmax())
                topic = EX[artifacts.class_names[c]]
                g.add((ind, indicatesTopic, topic))
                g.add((ind, affinityScore, Literal(float(A[fi, c]), datatype=XSD.float)))

        # Co-occurrence and contradiction relationships among feature individuals (compact subset)
        if include_feature_individuals and artifacts.feature_cooccur.nnz > 0:
            co = artifacts.feature_cooccur.tocoo()
            # cap the number of exported relations for file size
            cap = min(int(co.data.size), 5000)
            for k in range(cap):
                i = int(co.row[k])
                j = int(co.col[k])
                w = float(co.data[k])
                fi = EX[f"Feature_{i}"]
                fj = EX[f"Feature_{j}"]
                g.add((fi, coOccursWith, fj))
                g.add((fi, cooccurScore, Literal(w, datatype=XSD.float)))

        if include_feature_individuals and len(artifacts.contradiction_pairs) > 0:
            # export a capped set of contradictions
            cap = 3000
            for k, (i, j) in enumerate(sorted(artifacts.contradiction_pairs)[:cap]):
                fi = EX[f"Feature_{int(i)}"]
                fj = EX[f"Feature_{int(j)}"]
                g.add((fi, contradicts, fj))

        return g

    def export_owl(self, artifacts: OntologyArtifacts, out_path: str, include_feature_individuals: bool = True) -> str:
        g = self._build_graph(artifacts, include_feature_individuals=include_feature_individuals)
        # RDF/XML is accepted by Protégé with .owl extension
        g.serialize(destination=out_path, format="xml")
        return out_path

    def export_rdf(self, artifacts: OntologyArtifacts, out_path: str, include_feature_individuals: bool = True) -> str:
        g = self._build_graph(artifacts, include_feature_individuals=include_feature_individuals)
        g.serialize(destination=out_path, format="xml")
        return out_path

    def export_turtle(self, artifacts: OntologyArtifacts, out_path: str, include_feature_individuals: bool = True) -> str:
        g = self._build_graph(artifacts, include_feature_individuals=include_feature_individuals)
        g.serialize(destination=out_path, format="turtle")
        return out_path

    def export_swrl_rules(
        self,
        artifacts: OntologyArtifacts,
        out_path: str,
        rule_engine: Optional[OntologyRuleEngine] = None,
    ) -> str:
        engine = rule_engine or OntologyRuleEngine()
        rules = engine.semantic_consistency_rules()
        lines = []
        lines.append("# SWRL-like rules (human-readable)")
        lines.append("# These rules document the semantic logic used by the defense.")
        lines.append("# For full SWRL-in-OWL encoding, extend ontology_export.py to emit swrl:Imp triples.")
        lines.append("")
        for r in rules:
            lines.append(r.to_swrl())
        Path(out_path).write_text("\n".join(lines), encoding="utf-8")
        return out_path

