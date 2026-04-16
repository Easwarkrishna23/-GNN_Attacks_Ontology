"""
CSV -> OWL (RDF/XML) export for Protégé.

Why this exists
---------------
Your project already computes per-attack / per-defense metrics as CSVs.
This module converts a simple experiment CSV (results/attack_results.csv) into:

  ontology/gnn_attack_reasoned.owl

The output OWL is intentionally "reasoner-ready":
- It contains NamedIndividuals for Dataset / Model / Attack / Defense / ImpactMetric.
- It links them using the starter ontology object properties (targetsDataset, affectsModel, mitigatedBy, measuredBy).
- It stores numeric metrics via datatype properties (accuracyBefore/AfterAttack/AfterDefense plus any extra numeric columns).

Note: This script does not *run* a DL reasoner itself. Instead it produces an OWL file that you can
open in Protégé and run HermiT/Pellet there (and/or add SWRL rules in Protégé).
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import xml.etree.ElementTree as ET


NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}


def _register_namespaces() -> None:
    # Keep prefixes stable in the serialized RDF/XML (helps readability in Protégé).
    for k, v in NS.items():
        ET.register_namespace(k, v)


def _indent(elem: ET.Element, level: int = 0) -> None:
    # Minimal pretty-printer for ElementTree output.
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            _indent(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_]+")


def safe_id(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("-", "_").replace(" ", "_").replace(":", "_").replace("/", "_")
    s = _SAFE_ID_RE.sub("_", s)
    s = s.strip("_")
    return s or "Unknown"


def _qname(ns_key: str, local: str) -> str:
    return f"{{{NS[ns_key]}}}{local}"


def _rdf_about(value: str) -> Dict[str, str]:
    return {_qname("rdf", "about"): value}


def _rdf_resource(value: str) -> Dict[str, str]:
    return {_qname("rdf", "resource"): value}


def _xsd_float_text(v: float) -> Tuple[Dict[str, str], str]:
    return ({_qname("rdf", "datatype"): NS["xsd"] + "#float"}, f"{float(v):.6f}")


def _xsd_str_text(v: str) -> Tuple[Dict[str, str], str]:
    return ({_qname("rdf", "datatype"): NS["xsd"] + "#string"}, str(v))


def _parse_float(x: str) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _existing_individual_ids(root: ET.Element) -> set:
    ids = set()
    for ind in root.findall(_qname("owl", "NamedIndividual")):
        about = ind.attrib.get(_qname("rdf", "about"))
        if about:
            ids.add(about)
    return ids


def _add_named_individual(
    root: ET.Element,
    iri: str,
    rdf_type_iri: str,
    comment: Optional[str] = None,
) -> ET.Element:
    ind = ET.SubElement(root, _qname("owl", "NamedIndividual"), _rdf_about(iri))
    ET.SubElement(ind, _qname("rdf", "type"), _rdf_resource(rdf_type_iri))
    if comment:
        c = ET.SubElement(ind, _qname("rdfs", "comment"))
        c.text = comment
    return ind


def _ensure_individual(
    root: ET.Element,
    existing: set,
    iri: str,
    rdf_type_iri: str,
    comment: Optional[str] = None,
) -> None:
    if iri in existing:
        return
    _add_named_individual(root, iri, rdf_type_iri, comment=comment)
    existing.add(iri)


def generate_reasoned_ontology(
    csv_path: str = "results/attack_results.csv",
    base_owl_path: str = "ontology/gnn_attacks_ontology_starter.owl",
    out_owl_path: str = "ontology/gnn_attack_reasoned.owl",
) -> str:
    """
    Convert `csv_path` into an OWL RDF/XML file by extending `base_owl_path`.

    Returns the output path.
    """
    _register_namespaces()

    csv_path = str(csv_path)
    base_owl_path = str(base_owl_path)
    out_owl_path = str(out_owl_path)

    base = Path(base_owl_path)
    if not base.exists():
        raise FileNotFoundError(f"Base OWL not found: {base_owl_path}")

    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    tree = ET.parse(base_owl_path)
    root = tree.getroot()

    # Default namespace is the ontology namespace with '#'.
    ont_ns = root.attrib.get("xmlns", "")
    if not ont_ns.endswith("#"):
        ont_ns = ont_ns + "#"

    existing = _existing_individual_ids(root)
    linked_defenses = set()

    def iri(local: str) -> str:
        if local.startswith("#"):
            return local
        return "#" + local

    # Helper to add (namespace-local) properties.
    def add_obj_prop(ind: ET.Element, prop_local: str, target_iri: str) -> None:
        ET.SubElement(ind, f"{{{ont_ns}}}{prop_local}", _rdf_resource(target_iri))

    def add_data_prop_float(ind: ET.Element, prop_local: str, v: float) -> None:
        attrs, text = _xsd_float_text(v)
        e = ET.SubElement(ind, f"{{{ont_ns}}}{prop_local}", attrs)
        e.text = text

    def add_data_prop_str(ind: ET.Element, prop_local: str, v: str) -> None:
        attrs, text = _xsd_str_text(v)
        e = ET.SubElement(ind, f"{{{ont_ns}}}{prop_local}", attrs)
        e.text = text

    # ---------------------------------------------------------------------
    # Semantic rule individuals (so every defense has an explicit rule link)
    # ---------------------------------------------------------------------
    # The starter ontology defines class OntologyRule and object property usesRule (Defense -> OntologyRule).
    # We create a small fixed rule-set with human-readable SWRL-like comments.
    rules = [
        (
            iri("Rule_SemanticContradiction"),
            "If a node contains feature combinations that contradict topic constraints, mark it anomalous and repair.",
        ),
        (
            iri("Rule_LowTopicAffinityEdge"),
            "If an edge connects nodes with very low topic/label affinity, down-weight or prune the edge.",
        ),
        (
            iri("Rule_CentralityOutlier"),
            "If an injected neighbor creates an unusual centrality pattern (degree/role shift), reduce its trust.",
        ),
    ]
    for rule_iri, comment in rules:
        _ensure_individual(root, existing, rule_iri, iri("OntologyRule"), comment=comment)

    # Add individuals from CSV rows.
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        run_id = safe_id(row.get("run_id", ""))  # required for uniqueness
        dataset = safe_id(row.get("dataset", "Cora"))
        model = safe_id(row.get("model", "GCN"))
        attack_name = safe_id(row.get("attack", "Baseline"))
        attack_type = (row.get("attack_type") or "").strip().lower() or "none"
        defense_name = (row.get("defense") or "").strip()

        # Canonical individuals we will link to.
        dataset_iri = iri(f"{dataset}Dataset")
        model_iri = iri(f"{model}Model")

        # Subclass selection (based on starter ontology class names).
        if attack_type == "evasion":
            attack_class_iri = iri("EvasionAttack")
        elif attack_type == "poisoning":
            attack_class_iri = iri("PoisoningAttack")
        else:
            attack_class_iri = iri("Attack")

        defense_iri = None
        if defense_name and defense_name.lower() not in {"none", "baseline", "no", "null"}:
            defense_iri = iri(safe_id(defense_name))

        # Ensure base individuals exist.
        _ensure_individual(root, existing, dataset_iri, iri("Dataset"))
        _ensure_individual(root, existing, model_iri, iri("Model"))
        if defense_iri:
            _ensure_individual(root, existing, defense_iri, iri("Defense"))
            # Link defenses to semantic rules once (explicit semantic logic, not just similarity).
            if defense_iri not in linked_defenses:
                for rule_iri, _comment in rules:
                    for ind in root.findall(_qname("owl", "NamedIndividual")):
                        if ind.attrib.get(_qname("rdf", "about")) == defense_iri:
                            add_obj_prop(ind, "usesRule", rule_iri)
                            break
                linked_defenses.add(defense_iri)

        # Create per-run individuals.
        attack_iri = iri(f"Run_{run_id}_Attack")
        metric_iri = iri(f"Run_{run_id}_Impact")

        # Attack individual.
        if attack_iri not in existing:
            aind = _add_named_individual(
                root,
                attack_iri,
                attack_class_iri,
                comment=f"Attack run {run_id}: {row.get('attack', '')} on {row.get('dataset', '')} ({row.get('model', '')}).",
            )
            existing.add(attack_iri)

            add_obj_prop(aind, "targetsDataset", dataset_iri)
            add_obj_prop(aind, "affectsModel", model_iri)
            if defense_iri:
                add_obj_prop(aind, "mitigatedBy", defense_iri)

            # Severity score: prefer explicit, else compute from accuracy drop.
            sev = _parse_float(row.get("severity_score", ""))
            if sev is None:
                a0 = _parse_float(row.get("accuracy_before", "")) or 0.0
                a1 = _parse_float(row.get("accuracy_after_attack", "")) or a0
                sev = max(0.0, a0 - a1)
            add_data_prop_float(aind, "hasSeverityScore", float(sev))

            # Store the original attack label for readability in Protégé.
            add_data_prop_str(aind, "attackLabel", row.get("attack", "") or "")
            add_data_prop_str(aind, "attackType", attack_type)
            if defense_name:
                add_data_prop_str(aind, "defenseLabel", defense_name)

        # Metric individual.
        if metric_iri not in existing:
            mind = _add_named_individual(root, metric_iri, iri("AccuracyMetric"))
            existing.add(metric_iri)

            # Starter ontology defines these exact properties:
            a_before = _parse_float(row.get("accuracy_before", ""))
            a_att = _parse_float(row.get("accuracy_after_attack", ""))
            a_def = _parse_float(row.get("accuracy_after_defense", ""))
            if a_before is not None:
                add_data_prop_float(mind, "accuracyBefore", a_before)
            if a_att is not None:
                add_data_prop_float(mind, "accuracyAfterAttack", a_att)
            if a_def is not None:
                add_data_prop_float(mind, "accuracyAfterDefense", a_def)

            # Add any other numeric columns as float-typed data properties.
            reserved = {
                "run_id",
                "dataset",
                "model",
                "attack",
                "attack_type",
                "defense",
                "accuracy_before",
                "accuracy_after_attack",
                "accuracy_after_defense",
                "severity_score",
            }
            for k, v in row.items():
                if k in reserved:
                    continue
                fv = _parse_float(v)
                if fv is None:
                    continue
                add_data_prop_float(mind, safe_id(k), float(fv))

        # Link attack -> metric
        # (measuredBy is an ObjectProperty in the starter ontology).
        # Note: we write it even if it already exists; Protégé will de-duplicate.
        for ind in root.findall(_qname("owl", "NamedIndividual")):
            if ind.attrib.get(_qname("rdf", "about")) == attack_iri:
                add_obj_prop(ind, "measuredBy", metric_iri)
                break

    _indent(root)
    out = Path(out_owl_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_owl_path, encoding="utf-8", xml_declaration=True)
    return out_owl_path


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Convert results/attack_results.csv into an OWL file for Protégé.")
    p.add_argument("--csv", default="results/attack_results.csv", help="Input CSV path.")
    p.add_argument("--base", default="ontology/gnn_attacks_ontology_starter.owl", help="Base OWL (starter ontology) path.")
    p.add_argument("--out", default="ontology/gnn_attack_reasoned.owl", help="Output OWL path.")
    args = p.parse_args(argv)

    out = generate_reasoned_ontology(csv_path=args.csv, base_owl_path=args.base, out_owl_path=args.out)
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
