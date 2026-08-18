from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer


def ensure_results_dir(path: str | Path = "results") -> Path:
    results_dir = Path(path)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def plot_network_statistics(graph: nx.DiGraph, output_dir: str | Path = "results"):
    """Reproduce the original three-panel network overview."""
    output_dir = ensure_results_dir(output_dir)
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.bar(["Nodes", "Edges"], [graph.number_of_nodes(), graph.number_of_edges()])
    plt.title("Network Basic Structure")

    plt.subplot(132)
    plt.hist(list(dict(graph.in_degree()).values()), bins=50, log=True)
    plt.xlabel("In-degree")
    plt.ylabel("Frequency (log)")
    plt.title("Citation In-degree Distribution")

    self_loops = nx.number_of_selfloops(graph)
    total_edges = graph.number_of_edges()
    plt.subplot(133)
    plt.pie(
        [self_loops, total_edges - self_loops],
        labels=["Self-citations", "Normal citations"],
        autopct="%1.1f%%",
    )
    plt.title("Self-citation Ratio")

    plt.tight_layout()
    plt.savefig(output_dir / "network_statistics.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_paper_temporal_distribution(metadata: pd.DataFrame, output_dir: str | Path = "results"):
    output_dir = ensure_results_dir(output_dir)
    min_year = int(metadata["year"].min())
    max_year = int(metadata["year"].max())

    plt.figure(figsize=(10, 5))
    plt.hist(metadata["year"], bins=range(min_year, max_year + 2), align="left")
    plt.xticks(range(min_year, max_year + 1))
    plt.xlabel("Year")
    plt.ylabel("Number of Papers")
    plt.title("Temporal Distribution of Papers")
    plt.grid(axis="y", alpha=0.75)
    plt.tight_layout()
    plt.savefig(output_dir / "paper_temporal_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_community_size_distribution(community_sizes: dict, output_dir: str | Path = "results"):
    output_dir = ensure_results_dir(output_dir)
    bins = [
        (1, 5, "1-5"),
        (6, 10, "6-10"),
        (11, 50, "11-50"),
        (50, 150, "50-200"),
        (150, 500, "150-500"),
        (500, float("inf"), "500+"),
    ]

    labels, counts, percentages = [], [], []
    total = max(len(community_sizes), 1)
    for low, high, label in bins:
        if high == float("inf"):
            count = sum(size >= low for size in community_sizes.values())
        else:
            count = sum(low <= size <= high for size in community_sizes.values())
        labels.append(label)
        counts.append(count)
        percentages.append(count / total * 100)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, counts)
    for index, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{counts[index]}\n({percentages[index]:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    plt.xlabel("Community Size (nodes)")
    plt.ylabel("Number of Communities")
    plt.title("Distribution of Community Sizes")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "community_size_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_community_network(
    graph: nx.DiGraph,
    assignments: dict,
    community_sizes: dict,
    community_topics: dict,
    output_dir: str | Path = "results",
    limit: int = 20,
):
    """Reproduce the original topical network of the largest communities."""
    output_dir = ensure_results_dir(output_dir)
    top_ids = [cid for cid, _ in sorted(community_sizes.items(), key=lambda item: item[1], reverse=True)[:limit]]
    community_graph = nx.Graph()

    for community_id in top_ids:
        if community_id not in community_topics:
            continue
        keywords = community_topics[community_id]["community_keywords"][:3]
        community_graph.add_node(
            community_id,
            label=f"C{community_id}: " + ", ".join(keywords),
            size=community_topics[community_id]["num_papers"] * 0.5,
        )

    for source, target in graph.edges():
        if source not in assignments or target not in assignments:
            continue
        source_id = assignments[source]
        target_id = assignments[target]
        if source_id == target_id or source_id not in top_ids or target_id not in top_ids:
            continue
        if source_id not in community_graph or target_id not in community_graph:
            continue
        if community_graph.has_edge(source_id, target_id):
            community_graph[source_id][target_id]["weight"] += 1
        else:
            community_graph.add_edge(source_id, target_id, weight=1)

    if community_graph.number_of_edges() < 5:
        print("Warning: Too few edges between top communities. Skipping community_network.png")
        return

    plt.figure(figsize=(20, 15))
    positions = nx.spring_layout(community_graph, seed=42, k=0.5)
    node_sizes = [community_graph.nodes[node]["size"] for node in community_graph.nodes()]
    node_colors = [hash(node) % 30 for node in community_graph.nodes()]
    nx.draw_networkx_nodes(
        community_graph,
        positions,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.tab20,
        alpha=0.8,
    )
    edge_widths = [community_graph[u][v]["weight"] * 0.002 for u, v in community_graph.edges()]
    nx.draw_networkx_edges(
        community_graph,
        positions,
        width=edge_widths,
        edge_color="gray",
        alpha=0.6,
    )
    labels = {node: community_graph.nodes[node]["label"] for node in community_graph.nodes()}
    nx.draw_networkx_labels(
        community_graph,
        positions,
        labels=labels,
        font_size=6,
        font_family="sans-serif",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )
    plt.title("Community Network with Topical Labels", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "community_network.png", dpi=300)
    plt.close()


def plot_community_evolution(community_id, community_data: pd.DataFrame, output_dir: str | Path = "results"):
    """Preserve the original two-panel evolution plot and annotations."""
    output_dir = ensure_results_dir(output_dir)
    data = community_data.reset_index(drop=True).sort_values("window_center")
    if len(data) < 2:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
    ax1.plot(data["window_center"], data["introspection_ratio"], "^-", label="Introspection Ratio")
    ax1.plot(data["window_center"], data["inflow_ratio"], "s-", label="In-flow Ratio")
    ax1.plot(data["window_center"], data["outflow_ratio"], "D-", label="Out-flow Ratio")
    ax1.set_ylabel("Ratio")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.set_title(f"Knowledge Flow Ratios of Community {community_id}")

    ax2.plot(data["window_center"], data["node_count"], "o-", linewidth=2, markersize=8, label="Community Size")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Community Size")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.set_title(f"Size Evolution of Community {community_id}")

    global_class = data.iloc[0]["global_community"]
    ax2.text(
        0.95,
        0.95,
        f"Global Classification: {global_class}",
        transform=ax2.transAxes,
        fontsize=12,
        ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

    for index in range(len(data)):
        row = data.iloc[index]
        clean_label = "|".join(part for part in str(row["window_class"]).split("|") if part != "Active")
        ax2.annotate(
            clean_label,
            xy=(row["window_center"], row["node_count"] * 1.05),
            ha="center",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", alpha=0.7),
        )
        if index > 0:
            previous = data.iloc[index - 1]
            if row["window_class"] != previous["window_class"]:
                ax1.axvline(x=row["window_center"], color="gray", linestyle=":", alpha=0.7)
                ax2.axvline(x=row["window_center"], color="gray", linestyle=":", alpha=0.7)
                ax2.text(
                    row["window_center"],
                    ax2.get_ylim()[1] * 0.95,
                    f"Change: {previous['window_class']} → {row['window_class']}",
                    ha="center",
                    va="top",
                    rotation=90,
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.5),
                )

    plt.tight_layout()
    plt.savefig(output_dir / f"community_{community_id}_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_community_topic_evolution(
    community_id,
    community_data: pd.DataFrame,
    metadata: pd.DataFrame,
    community_papers,
    paper_to_year: dict,
    output_dir: str | Path = "results",
):
    output_dir = ensure_results_dir(output_dir)
    data = community_data.sort_values("window_center")
    all_keywords = set()
    topic_data = {}

    for _, row in data.iterrows():
        start_year, end_year = row["year_start"], row["year_end"]
        key = f"{start_year}-{end_year}"
        paper_ids = [
            paper_id
            for paper_id in community_papers[community_id]
            if start_year <= paper_to_year.get(paper_id, -1) <= end_year
        ]
        if len(paper_ids) < 5:
            topic_data[key] = {}
            continue

        window_metadata = metadata[metadata["paper_id"].isin(paper_ids)]
        texts = (window_metadata["title"] + " " + window_metadata["abstract"]).tolist()
        try:
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8,
            )
            matrix = vectorizer.fit_transform(texts)
            names = vectorizer.get_feature_names_out()
            scores = np.asarray(matrix.mean(axis=0)).flatten()
            values = {names[i]: score for i, score in enumerate(scores) if score > 0.01}
            topic_data[key] = values
            all_keywords.update(values)
        except Exception as exc:
            print(f"Error extracting topics for community {community_id} in {key}: {exc}")
            topic_data[key] = {}

    keywords = sorted(all_keywords)
    windows = sorted(topic_data)
    if not keywords or not windows:
        return

    matrix = np.zeros((len(keywords), len(windows)))
    for column, window in enumerate(windows):
        for row, keyword in enumerate(keywords):
            matrix[row, column] = topic_data[window].get(keyword, 0)

    if len(keywords) > 20:
        top_indices = np.argsort(matrix.sum(axis=1))[-20:][::-1]
        matrix = matrix[top_indices]
        keywords = [keywords[i] for i in top_indices]

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        matrix,
        cmap="YlGnBu",
        linewidths=0.5,
        linecolor="lightgray",
        xticklabels=windows,
        yticklabels=keywords,
        cbar_kws={"label": "Keyword Importance (TF-IDF)"},
    )
    plt.title(f"Topic Evolution in Community {community_id}", fontsize=14)
    plt.xlabel("Time Window", fontsize=12)
    plt.ylabel("Keywords", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / f"community_{community_id}_topic_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_growth_patterns(growth_features: pd.DataFrame, output_dir: str | Path = "results"):
    output_dir = ensure_results_dir(output_dir)
    counts = growth_features["growth_pattern"].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(12, 8))
    bars = plt.barh(counts.index, counts.values)
    total = len(growth_features)
    for index, count in enumerate(counts.values):
        plt.text(count, index, f" {count} ({count / total:.1%})", va="center", ha="left")
    plt.xlim(0, counts.max() * 1.2 if len(counts) else 1)
    plt.title("Distribution of Community Growth Patterns", fontsize=16)
    plt.xlabel("Number of Communities", fontsize=12)
    plt.ylabel("Growth Pattern", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "growth_patterns.png", dpi=300)
    plt.close()
