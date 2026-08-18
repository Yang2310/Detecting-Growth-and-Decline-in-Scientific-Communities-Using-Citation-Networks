from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


def ensure_results_dir(path: str | Path = "results") -> Path:
    results_dir = Path(path)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def plot_network_statistics(graph, output_dir: str | Path = "results"):
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
        [self_loops, max(total_edges - self_loops, 0)],
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
        (51, 150, "51-150"),
        (151, 500, "151-500"),
        (501, float("inf"), "500+"),
    ]

    labels, counts = [], []
    for low, high, label in bins:
        labels.append(label)
        counts.append(
            sum(
                1
                for size in community_sizes.values()
                if size >= low and (high == float("inf") or size <= high)
            )
        )

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts)
    plt.xlabel("Community Size (nodes)")
    plt.ylabel("Number of Communities")
    plt.title("Distribution of Community Sizes")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "community_size_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_community_evolution(community_id, community_data: pd.DataFrame, output_dir: str | Path = "results"):
    output_dir = ensure_results_dir(output_dir)
    data = community_data.sort_values("window_center").reset_index(drop=True)
    if len(data) < 2:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
    ax1.plot(data["window_center"], data["introspection_ratio"], "^-", label="Introspection Ratio")
    ax1.plot(data["window_center"], data["inflow_ratio"], "s-", label="In-flow Ratio")
    ax1.plot(data["window_center"], data["outflow_ratio"], "D-", label="Out-flow Ratio")
    ax1.set_ylabel("Ratio")
    ax1.set_title(f"Knowledge Flow Ratios of Community {community_id}")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)

    ax2.plot(data["window_center"], data["node_count"], "o-", linewidth=2, label="Community Size")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Community Size")
    ax2.set_title(f"Size Evolution of Community {community_id}")
    ax2.grid(True, linestyle="--", alpha=0.7)

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

    from sklearn.feature_extraction.text import TfidfVectorizer

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
        except ValueError:
            topic_data[key] = {}

    keywords = sorted(all_keywords)
    windows = sorted(topic_data)
    if not keywords or not windows:
        return

    matrix = np.zeros((len(keywords), len(windows)))
    for j, window in enumerate(windows):
        for i, keyword in enumerate(keywords):
            matrix[i, j] = topic_data[window].get(keyword, 0)

    if len(keywords) > 20:
        top_indices = np.argsort(matrix.sum(axis=1))[-20:][::-1]
        matrix = matrix[top_indices]
        keywords = [keywords[i] for i in top_indices]

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        matrix,
        cmap="YlGnBu",
        linewidths=0.5,
        xticklabels=windows,
        yticklabels=keywords,
        cbar_kws={"label": "Keyword Importance (TF-IDF)"},
    )
    plt.title(f"Topic Evolution in Community {community_id}")
    plt.xlabel("Time Window")
    plt.ylabel("Keywords")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / f"community_{community_id}_topic_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_growth_patterns(growth_features: pd.DataFrame, output_dir: str | Path = "results"):
    output_dir = ensure_results_dir(output_dir)
    counts = growth_features["growth_pattern"].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    plt.barh(counts.index, counts.values)
    plt.xlabel("Number of Communities")
    plt.ylabel("Growth Pattern")
    plt.title("Distribution of Community Growth Patterns")
    plt.tight_layout()
    plt.savefig(output_dir / "growth_patterns.png", dpi=300, bbox_inches="tight")
    plt.close()
