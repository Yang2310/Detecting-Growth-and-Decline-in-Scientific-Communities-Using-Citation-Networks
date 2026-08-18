"""Extended analyses preserved from the original thesis pipeline.

This module contains the secondary analyses and visualizations that were
previously embedded in the monolithic ``data.py`` script. Functions accept
explicit inputs so the refactored pipeline keeps the original functionality
without relying on global variables.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _ensure_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def analyze_community_roles(
    temporal_df: pd.DataFrame,
    output_dir: str | Path = "results",
) -> pd.DataFrame:
    """Plot the number of communities assigned to core roles over time."""
    output_dir = _ensure_output_dir(output_dir)
    core_roles = ["Exporter", "Insular", "Hub", "Declining", "Growing"]

    tagged = temporal_df.copy()
    for role in core_roles:
        tagged[role] = tagged["window_class"].apply(lambda value: int(role in value))

    role_over_time = tagged.groupby("window_center")[core_roles].sum()

    plt.figure(figsize=(12, 7))
    for role in core_roles:
        plt.plot(
            role_over_time.index,
            role_over_time[role],
            marker="o",
            linewidth=2,
            label=role,
        )
    plt.title("Community Core Roles Over Time")
    plt.xlabel("Year")
    plt.ylabel("Number of Communities")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(title="Role")
    plt.tight_layout()
    plt.savefig(output_dir / "core_roles_over_time.png", dpi=300)
    plt.close()

    return role_over_time


def plot_emergence_disappearance_trends(
    temporal_df: pd.DataFrame,
    output_dir: str | Path = "results",
) -> pd.DataFrame:
    """Visualize transitions into and out of the Active state."""
    output_dir = _ensure_output_dir(output_dir)
    data = temporal_df.sort_values(["community", "window_center"]).copy()
    data["prev_status"] = data.groupby("community")["status"].shift(1)
    data = data.dropna(subset=["prev_status"])

    data["emerging_count"] = (
        (data["prev_status"] != "Active") & (data["status"] == "Active")
    ).astype(int)
    data["disappearing_count"] = (
        (data["prev_status"] == "Active") & (data["status"] != "Active")
    ).astype(int)

    trends = data.groupby("window_center")[["emerging_count", "disappearing_count"]].sum()

    plt.figure(figsize=(14, 7))
    plt.plot(trends.index, trends["emerging_count"], "o-", label="Emerging Communities")
    plt.plot(
        trends.index,
        trends["disappearing_count"],
        "s--",
        label="Disappearing Communities",
    )
    plt.title("Macro Trends: Emergence and Disappearance of Communities Over Time")
    plt.xlabel("Year")
    plt.ylabel("Number of Communities")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "community_macro_trends.png", dpi=300)
    plt.close()

    return trends


def analyze_knowledge_flow(
    temporal_df: pd.DataFrame,
    output_dir: str | Path = "results",
) -> pd.DataFrame:
    """Cluster active communities by their outflow-ratio trajectories."""
    output_dir = _ensure_output_dir(output_dir)
    active = temporal_df[temporal_df["status"] == "Active"].copy()
    if active.empty:
        return pd.DataFrame()

    flow_matrix = active.pivot_table(
        index="community",
        columns="window_center",
        values="outflow_ratio",
        fill_value=0,
    )

    figsize_height = max(10, len(flow_matrix.index) * 0.4)
    cluster = sns.clustermap(
        flow_matrix,
        cmap="coolwarm",
        figsize=(12, figsize_height),
        standard_scale=1,
    )
    cluster.fig.suptitle("Knowledge Flow Patterns Across ACTIVE Communities", y=1.01)
    cluster.fig.savefig(output_dir / "knowledge_flow_clustermap.png", dpi=300, bbox_inches="tight")
    plt.close(cluster.fig)
    return flow_matrix


def visualize_cross_community_flow(
    flow_matrix: np.ndarray,
    community_ids: list,
    community_topics: dict,
    output_path: str | Path,
) -> None:
    """Plot the strongest cross-community citation flows as a heatmap."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = []
    for community_id in community_ids:
        keywords = community_topics.get(community_id, {}).get("community_keywords", [])
        label = f"C{community_id}"
        if keywords:
            label += f"\n{keywords[0]}"
        labels.append(label)

    flow_df = pd.DataFrame(flow_matrix, index=labels, columns=labels)
    if flow_df.empty:
        return

    min_flow = flow_df.sum(axis=1).quantile(0.7)
    active = flow_df.sum(axis=1) > min_flow
    filtered = flow_df.loc[active, active]
    if filtered.empty:
        return

    plt.figure(figsize=(16, 14))
    sns.heatmap(
        np.log1p(filtered.values),
        annot=True,
        fmt=".1f",
        cmap="viridis",
        xticklabels=filtered.index,
        yticklabels=filtered.columns,
        linewidths=0.5,
        cbar_kws={"label": "Log(1 + Citation Count)"},
    )
    plt.title("Cross-Community Knowledge Flow")
    plt.xlabel("Target Community")
    plt.ylabel("Source Community")
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_community_state_distribution(
    temporal_df: pd.DataFrame,
    output_dir: str | Path = "results",
) -> pd.DataFrame:
    """Plot Active, Latent, and Dormant community counts over time."""
    output_dir = _ensure_output_dir(output_dir)
    state_counts = temporal_df.groupby(["window_center", "status"]).size().unstack(fill_value=0)
    for column in ["Dormant", "Latent", "Active"]:
        if column not in state_counts:
            state_counts[column] = 0
    state_counts = state_counts[["Dormant", "Latent", "Active"]]

    plt.figure(figsize=(14, 8))
    plt.stackplot(
        state_counts.index,
        state_counts["Dormant"],
        state_counts["Latent"],
        state_counts["Active"],
        labels=["Dormant", "Latent", "Active"],
    )
    plt.title("Distribution of Community States Over Time")
    plt.xlabel("Year")
    plt.ylabel("Number of Communities")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlim(state_counts.index.min(), state_counts.index.max())
    plt.tight_layout()
    plt.savefig(output_dir / "community_state_distribution.png", dpi=300)
    plt.close()
    return state_counts


def plot_community_lifecycle_heatmap(
    temporal_df: pd.DataFrame,
    output_dir: str | Path = "results",
) -> pd.DataFrame:
    """Track each community's Dormant/Latent/Active state over time."""
    output_dir = _ensure_output_dir(output_dir)
    status_map = {"Dormant": 0, "Latent": 1, "Active": 2}
    data = temporal_df.copy()
    data["status_numeric"] = data["status"].map(status_map)

    matrix = data.pivot_table(
        index="community",
        columns="window_center",
        values="status_numeric",
    )
    matrix["max_status"] = matrix.max(axis=1)
    matrix["active_duration"] = (matrix > 0).sum(axis=1)
    matrix = matrix.sort_values(
        by=["max_status", "active_duration"],
        ascending=[False, False],
    )
    matrix = matrix.drop(columns=["max_status", "active_duration"])

    height = max(10, len(matrix.index) * 0.1)
    plt.figure(figsize=(20, height))
    ax = sns.heatmap(
        matrix,
        cmap=["#B0C4DE", "#FFD700", "#2E8B57"],
        linewidths=0.5,
        cbar_kws={"ticks": [0, 1, 2], "shrink": 0.3},
    )
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticklabels(["Dormant", "Latent", "Active"])
    plt.title("Lifecycle of Each Community Over Time")
    plt.xlabel("Year")
    plt.ylabel("Community ID (Sorted by Activity Level)")
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "community_lifecycle_heatmap.png", dpi=300)
    plt.close()
    return matrix


def identify_emerging_community_sources(
    temporal_df: pd.DataFrame,
    flow_matrix: np.ndarray,
    community_ids: list,
    community_topics: dict,
) -> pd.DataFrame:
    """Identify communities that become Active and summarize their main knowledge sources."""
    data = temporal_df.sort_values(["community", "window_center"]).copy()
    data["prev_status"] = data.groupby("community")["status"].shift(1)
    events = data[
        data["prev_status"].notna()
        & (data["prev_status"] != "Active")
        & (data["status"] == "Active")
    ]
    emerging_ids = events["community"].unique()

    id_to_index = {community_id: index for index, community_id in enumerate(community_ids)}
    records = []
    for community_id in emerging_ids:
        if community_id not in id_to_index:
            continue
        target_index = id_to_index[community_id]
        inflows = flow_matrix[:, target_index]
        sources = []
        for source_index, count in enumerate(inflows):
            if count <= 0:
                continue
            source_id = community_ids[source_index]
            source_name = "|".join(
                community_topics.get(source_id, {}).get("community_keywords", [])[:2]
            )
            sources.append((source_name, int(count)))
        sources.sort(key=lambda item: item[1], reverse=True)

        records.append(
            {
                "emerging_community": community_id,
                "emerging_name": "|".join(
                    community_topics.get(community_id, {}).get("community_keywords", [])[:2]
                ),
                "main_sources": "; ".join(
                    f"{name}({count})" for name, count in sources[:3]
                ),
                "total_inflow": int(np.sum(inflows)),
            }
        )

    return pd.DataFrame(records)


def print_analysis_report(
    temporal_df: pd.DataFrame,
    growth_features: pd.DataFrame,
    bridge_df: pd.DataFrame,
    emerging_df: pd.DataFrame,
) -> None:
    """Print the summary report produced by the original script."""
    print("\n" + "=" * 60)
    print("Community Evolution Analysis Report")
    print("=" * 60)
    print(f"Total communities analyzed: {temporal_df['community'].nunique()}")
    print(f"Time windows covered: {temporal_df['window_center'].nunique()}")
    print("\nGrowth pattern distribution:")
    print(growth_features["growth_pattern"].value_counts())

    print("\n" + "=" * 60)
    print("All visualizations saved to 'results' directory")
    print("=" * 60)

    print("\n=== Cross-Domain Analysis Results ===")
    if not bridge_df.empty:
        top = bridge_df.iloc[0]
        print(f"Top Bridge Community: C{top['community']} ({top['community_name']})")
        print(f"  - Hub Score: {top['hub_score']:.1f}")
        print(f"  - Mediation Count: {top['mediation']}")

    if not emerging_df.empty:
        print("\nEmerging Communities and Their Knowledge Sources:")
        for _, row in emerging_df.iterrows():
            print(f"- {row['emerging_name']}: Sources: {row['main_sources']}")
    else:
        print("\nNo emerging communities identified")
