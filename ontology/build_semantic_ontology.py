from __future__ import annotations

from pathlib import Path

import numpy as np
from owlready2 import DataProperty, FunctionalProperty, ObjectProperty, Thing, get_ontology

TOPIC_NAMES = [
    "CaseBased",
    "GeneticAlgorithms",
    "NeuralNetworks",
    "ProbabilisticMethods",
    "ReinforcementLearning",
    "RuleLearning",
    "Theory",
]


def build_semantic_ontology(data, output_path: str) -> str:
    """
    Build a Cora-specific ontology with classes, subclasses, object/data properties,
    and per-node instances for semantic defense reasoning.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    onto = get_ontology("http://example.org/gnn_security.owl")

    with onto:
        class Domain(Thing):
            pass

        class Topic(Thing):
            pass

        class Node(Thing):
            pass

        class FeatureVector(Thing):
            pass

        class ResearchDomain(Domain):
            pass

        class ComputerScience(ResearchDomain):
            pass

        topic_classes = {}
        for t in TOPIC_NAMES:
            topic_classes[t] = type(t, (Topic,), {})

        class belongs_to(Node >> Topic, ObjectProperty):
            pass

        class has_feature_vector(Node >> FeatureVector, ObjectProperty):
            pass

        class similar_to(Node >> Node, ObjectProperty):
            pass

        class in_domain(Topic >> Domain, ObjectProperty):
            pass

        class topic_of_feature(FeatureVector >> Topic, ObjectProperty):
            pass

        class semantic_confidence(Node >> float, DataProperty, FunctionalProperty):
            pass

        class feature_density(FeatureVector >> float, DataProperty, FunctionalProperty):
            pass

        class consistency_score(Node >> float, DataProperty, FunctionalProperty):
            pass

        class minSemanticSupport(Topic >> float, DataProperty, FunctionalProperty):
            pass

        class compatibleWith(Topic >> Topic, ObjectProperty):
            pass

    # Create domain instance and topic instances.
    cs = onto.ComputerScience("ComputerScienceDomain")
    topic_inds = {}
    for t in TOPIC_NAMES:
        topic_inds[t] = onto.search_one(iri=f"*{t}Topic") or onto[t](f"{t}Topic")
        topic_inds[t].in_domain = [cs]

    # Topic compatibility map.
    compat = {
        "CaseBased": ["CaseBased", "RuleLearning"],
        "GeneticAlgorithms": ["GeneticAlgorithms", "NeuralNetworks"],
        "NeuralNetworks": ["NeuralNetworks", "GeneticAlgorithms", "ProbabilisticMethods", "ReinforcementLearning"],
        "ProbabilisticMethods": ["ProbabilisticMethods", "NeuralNetworks", "Theory"],
        "ReinforcementLearning": ["ReinforcementLearning", "NeuralNetworks"],
        "RuleLearning": ["RuleLearning", "CaseBased", "Theory"],
        "Theory": ["Theory", "ProbabilisticMethods", "RuleLearning"],
    }
    support = {
        "CaseBased": 0.55,
        "GeneticAlgorithms": 0.50,
        "NeuralNetworks": 0.48,
        "ProbabilisticMethods": 0.48,
        "ReinforcementLearning": 0.50,
        "RuleLearning": 0.55,
        "Theory": 0.50,
    }

    for t, inds in topic_inds.items():
        inds.minSemanticSupport = support[t]
        inds.compatibleWith = [topic_inds[c] for c in compat[t]]

    # Create node + feature instances.
    x = data.x.detach().cpu().numpy().astype(np.float32)
    y = data.y.detach().cpu().numpy().astype(int)
    edge_index = data.edge_index.detach().cpu().numpy()
    train_mask = data.train_mask.detach().cpu().numpy().astype(bool)

    # Neighborhood list for local semantic consistency.
    n = data.num_nodes
    nbrs = [[] for _ in range(n)]
    for u, v in zip(edge_index[0], edge_index[1]):
        u = int(u)
        v = int(v)
        if u != v:
            nbrs[u].append(v)

    # Feature-topic affinity from train nodes.
    aff = np.zeros((x.shape[1], len(TOPIC_NAMES)), dtype=np.float32)
    for c in range(len(TOPIC_NAMES)):
        idx = np.where(train_mask & (y == c))[0]
        if idx.size > 0:
            aff[:, c] = x[idx].mean(axis=0)
    aff_sum = aff.sum(axis=1, keepdims=True)
    aff_sum[aff_sum == 0.0] = 1.0
    aff = aff / aff_sum
    topic_scores = x @ aff
    ts_sum = topic_scores.sum(axis=1, keepdims=True)
    ts_sum[ts_sum == 0.0] = 1.0
    topic_scores = topic_scores / ts_sum
    dominant = np.argmax(topic_scores, axis=1)

    for i in range(n):
        node_i = onto.search_one(iri=f"*Node_{i}") or onto.Node(f"Node_{i}")
        feat_i = onto.search_one(iri=f"*Feature_{i}") or onto.FeatureVector(f"Feature_{i}")

        topic_name = TOPIC_NAMES[int(dominant[i])]
        node_i.belongs_to = [topic_inds[topic_name]]
        node_i.has_feature_vector = [feat_i]
        feat_i.topic_of_feature = [topic_inds[topic_name]]

        dens = float(np.mean(x[i] > 0.0))
        feat_i.feature_density = dens

        neigh = nbrs[i]
        if neigh:
            sim_local = float(np.mean([np.dot(topic_scores[i], topic_scores[j]) for j in neigh]))
        else:
            sim_local = float(np.max(topic_scores[i]))
        node_i.semantic_confidence = float(np.max(topic_scores[i]))
        node_i.consistency_score = sim_local

        # Link to one semantically strongest neighbor from graph.
        if neigh:
            best = max(neigh, key=lambda j: float(np.dot(topic_scores[i], topic_scores[j])))
            node_i.similar_to = [onto.search_one(iri=f"*Node_{int(best)}") or onto.Node(f"Node_{int(best)}")]

    onto.save(file=str(output), format="rdfxml")
    return str(output)
