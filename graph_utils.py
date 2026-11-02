import random


def generate_dynamic_directed_graph(num_nodes, min_out_degree=2, max_out_degree=5, seed=None):
    """Generate one directed communication graph without self-loops.

    Returns outgoing and incoming adjacency lists. Each node samples an
    out-degree in the configured range, capped by the available peers.
    """
    rng = random.Random(seed)

    out_neighbors = {i: set() for i in range(num_nodes)}
    in_neighbors = {i: set() for i in range(num_nodes)}

    for i in range(num_nodes):
        out_deg = rng.randint(min_out_degree, max_out_degree)
        candidates = [j for j in range(num_nodes) if j != i]
        chosen = rng.sample(candidates, k=min(out_deg, len(candidates)))

        for j in chosen:
            out_neighbors[i].add(j)
            in_neighbors[j].add(i)

    out_neighbors = {k: sorted(list(v)) for k, v in out_neighbors.items()}
    in_neighbors = {k: sorted(list(v)) for k, v in in_neighbors.items()}
    return out_neighbors, in_neighbors


def build_aggregation_weights(node_id, in_neighbors, self_weight=0.5):
    """Assign ``self_weight`` locally and split the remainder over in-neighbors."""

    incoming = in_neighbors[node_id]
    weights = {node_id: self_weight}

    if len(incoming) == 0:
        return {node_id: 1.0}

    remain = 1.0 - self_weight
    each = remain / len(incoming)
    for nbr in incoming:
        weights[nbr] = each

    return weights
