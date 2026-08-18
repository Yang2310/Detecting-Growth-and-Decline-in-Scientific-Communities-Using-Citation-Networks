from collections import Counter, defaultdict

import leidenalg
import networkx as nx
import numpy as np
import pandas as pd
from igraph import Graph
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer


def detect_communities(graph: nx.DiGraph, seed: int = 42):
    """Detect communities with Leiden on an undirected projection of the citation graph."""
    graph_undirected = graph.to_undirected()
    graph_undirected.remove_edges_from(nx.selfloop_edges(graph_undirected))

    igraph_graph = Graph.TupleList(list(graph_undirected.edges()), directed=False)
    partition = leidenalg.find_partition(
        igraph_graph,
        leidenalg.ModularityVertexPartition,
        n_iterations=5,
        seed=seed,
    )

    assignments = {}
    for community_id, community in enumerate(partition):
        for node_id in community:
            assignments[igraph_graph.vs[node_id]["name"]] = community_id

    sizes = Counter(assignments.values())
    return assignments, dict(sizes)


def map_community_papers(assignments: dict, metadata: pd.DataFrame):
    """Map community IDs to paper IDs that have metadata."""
    valid_ids = set(metadata["paper_id"])
    community_papers = defaultdict(list)
    for paper_id, community_id in assignments.items():
        if paper_id in valid_ids:
            community_papers[community_id].append(paper_id)
    return community_papers


def extract_community_topics(
    community_id: int,
    paper_ids: list[str],
    metadata: pd.DataFrame,
    n_topics: int = 3,
):
    """Extract representative topic keywords for a community with TF-IDF + LDA."""
    if len(paper_ids) < 10:
        return None

    community_metadata = metadata[metadata["paper_id"].isin(paper_ids)]
    texts = (community_metadata["title"] + " " + community_metadata["abstract"]).tolist()

    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    n_topics_actual = min(n_topics, len(texts) // 5, 5)
    n_topics_actual = max(2, n_topics_actual)

    lda = LatentDirichletAllocation(
        n_components=n_topics_actual,
        random_state=42,
        max_iter=50,
    )
    lda.fit(tfidf_matrix)

    topics = []
    topic_weights = []
    for topic in lda.components_:
        top_indices = topic.argsort()[-8:][::-1]
        topics.append([feature_names[i] for i in top_indices])
        topic_weights.append([topic[i] for i in top_indices])

    all_keywords = [keyword for topic in topics for keyword in topic[:3]]
    keyword_counts = Counter(all_keywords)
    community_keywords = [word for word, _ in keyword_counts.most_common(8)]

    return {
        "community_id": community_id,
        "topics": topics,
        "topic_weights": topic_weights,
        "community_keywords": community_keywords,
        "keyword_counts": keyword_counts,
        "num_papers": len(paper_ids),
    }


def analyze_major_community_topics(
    community_papers,
    metadata: pd.DataFrame,
    limit: int = 20,
):
    """Run topic analysis for the largest detected communities."""
    results = {}
    major = sorted(community_papers.items(), key=lambda item: len(item[1]), reverse=True)

    for community_id, paper_ids in major[:limit]:
        result = extract_community_topics(community_id, paper_ids, metadata)
        if result:
            results[community_id] = result

    return results


def build_cross_community_flow(citation_graph: nx.DiGraph, assignments: dict):
    """Build an inter-community citation-flow matrix."""
    community_ids = sorted(set(assignments.values()))
    index = {community_id: i for i, community_id in enumerate(community_ids)}
    matrix = np.zeros((len(community_ids), len(community_ids)), dtype=int)

    for source, target in citation_graph.edges():
        if source not in assignments or target not in assignments:
            continue

        source_community = assignments[source]
        target_community = assignments[target]
        if source_community != target_community:
            matrix[index[source_community], index[target_community]] += 1

    return matrix, community_ids


def identify_bridge_communities(flow_matrix, community_ids, community_topics):
    """Rank communities by hub and mediation behaviour across citation flows."""
    bridge_scores = []
    matrix = flow_matrix.astype(np.float64)

    for i, community_id in enumerate(community_ids):
        out_flow = matrix[i].sum()
        in_flow = matrix[:, i].sum()
        diversity = np.count_nonzero(matrix[i])
        hub_score = out_flow * diversity / max(1, len(community_ids) - 1)

        mediation = 0.0
        for j in range(len(community_ids)):
            if j == i:
                continue
            for k in range(len(community_ids)):
                if k == i or k == j:
                    continue
                if matrix[j][k] > 0 and matrix[j][i] > 0 and matrix[i][k] > 0:
                    log_mediation = (
                        np.log1p(matrix[j][k])
                        + np.log1p(matrix[j][i])
                        + np.log1p(matrix[i][k])
                    )
                    mediation += np.exp(log_mediation)

        keywords = community_topics.get(community_id, {}).get("community_keywords", [])[:2]
        bridge_scores.append(
            {
                "community": community_id,
                "hub_score": hub_score,
                "inflow": in_flow,
                "mediation": mediation,
                "bridge_score": hub_score + mediation * 0.1,
                "community_name": "|".join(keywords),
            }
        )

    return pd.DataFrame(bridge_scores).sort_values("bridge_score", ascending=False)
