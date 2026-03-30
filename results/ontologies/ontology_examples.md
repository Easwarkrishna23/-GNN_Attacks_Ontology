# Ontology Creation and Examples

## How Ontologies Are Created
- We compute semantic similarity between node feature vectors using cosine similarity.
- Optionally, we blend semantic similarity with label agreement (label-guided ontology).
- Finally, we row-normalize to obtain a stochastic ontology matrix O (rows sum to 1).

## Example: Target Node Ontology Neighbors
Target node: 1708 (label=3)

### Variant: semantic_only
- neighbor=1301 label=3 weight=0.002634
- neighbor=1949 label=3 weight=0.002222
- neighbor=62 label=0 weight=0.002178
- neighbor=1578 label=5 weight=0.002049
- neighbor=1806 label=3 weight=0.001953

### Variant: label_guided_w0_9
- neighbor=1301 label=3 weight=0.002144
- neighbor=1949 label=3 weight=0.001875
- neighbor=1806 label=3 weight=0.001700
- neighbor=2484 label=3 weight=0.001633
- neighbor=1340 label=3 weight=0.001633

### Variant: label_guided_w0_7
- neighbor=1301 label=3 weight=0.001685
- neighbor=1949 label=3 weight=0.001550
- neighbor=1806 label=3 weight=0.001462
- neighbor=2484 label=3 weight=0.001428
- neighbor=1340 label=3 weight=0.001428
