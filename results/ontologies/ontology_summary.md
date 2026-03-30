# Ontology Variants Summary

We export only top-k semantic neighbors per node (not the full NxN ontology matrix).
Variants:
- semantic_only: cosine similarity on features only.
- label_guided_w0_9: 90% semantic similarity + 10% label agreement (for explanation only).
- label_guided_w0_7: 70% semantic similarity + 30% label agreement (for explanation only).

Files:
- ontology_topk_edges.csv: (variant, source, target, weight) for all nodes.
- ontology_examples.md: human-readable example for one target node.