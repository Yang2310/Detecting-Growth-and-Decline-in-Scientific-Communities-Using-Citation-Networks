from collections import defaultdict

import numpy as np
import pandas as pd


def calculate_coin_metrics(
    year_start: int,
    year_end: int,
    citation_graph,
    paper_to_year: dict,
    paper_to_community: dict,
):
    """Calculate the original COIN-style metrics for a time window."""
    valid_nodes = {
        paper_id
        for paper_id, year in paper_to_year.items()
        if year_start <= year <= year_end
    }

    edges_in_window = [
        (source, target)
        for source, target in citation_graph.edges()
        if source in valid_nodes and target in paper_to_year
    ]

    metrics = defaultdict(
        lambda: {
            "introspection": 0,
            "inflow": 0,
            "outflow": 0,
            "total_citations": 0,
            "node_count": 0,
        }
    )

    for paper_id, year in paper_to_year.items():
        if year_start <= year <= year_end and paper_id in paper_to_community:
            metrics[paper_to_community[paper_id]]["node_count"] += 1

    for source, target in edges_in_window:
        if source not in paper_to_community or target not in paper_to_community:
            continue
        source_community = paper_to_community[source]
        target_community = paper_to_community[target]
        metrics[source_community]["total_citations"] += 1
        if source_community == target_community:
            metrics[source_community]["introspection"] += 1
        else:
            metrics[source_community]["outflow"] += 1
            metrics[target_community]["inflow"] += 1

    for values in metrics.values():
        total = values["total_citations"]
        if total > 0:
            values["introspection_ratio"] = values["introspection"] / total
            values["inflow_ratio"] = values["inflow"] / total
            values["outflow_ratio"] = values["outflow"] / total
        else:
            values["introspection_ratio"] = 0
            values["inflow_ratio"] = 0
            values["outflow_ratio"] = 0
        values["influence_score"] = values["node_count"] + values["inflow"] * 0.5

    return dict(metrics)


def build_temporal_dataframe(
    citation_graph,
    metadata: pd.DataFrame,
    assignments: dict,
    window_size: int = 2,
):
    """Reproduce the original sliding-window community analysis."""
    paper_to_year = dict(zip(metadata["paper_id"], metadata["year"]))
    years = sorted(metadata["year"].dropna().unique())
    all_communities = set(assignments.values())
    previous_size = defaultdict(int)
    records = []

    min_year = int(min(years))
    max_year = int(max(years))

    for start_year in range(min_year, max_year - window_size + 2):
        end_year = start_year + window_size - 1
        if end_year > max_year:
            break

        coin_metrics = calculate_coin_metrics(
            start_year,
            end_year,
            citation_graph,
            paper_to_year,
            assignments,
        )

        active_count = 0
        for community_id in all_communities:
            values = coin_metrics.get(
                community_id,
                {
                    "node_count": 0,
                    "total_citations": 0,
                    "introspection": 0,
                    "inflow": 0,
                    "outflow": 0,
                    "introspection_ratio": 0,
                    "inflow_ratio": 0,
                    "outflow_ratio": 0,
                    "influence_score": 0,
                },
            )

            current_size = values["node_count"]
            if current_size >= 5:
                status = "Active"
                active_count += 1
            elif current_size > 0:
                status = "Latent"
            else:
                status = "Dormant"

            previous = previous_size[community_id]
            growth_rate = (current_size - previous) / previous if previous > 0 else 0.0

            records.append(
                {
                    "community": community_id,
                    "year_start": start_year,
                    "year_end": end_year,
                    "window_center": (start_year + end_year) / 2,
                    "node_count": current_size,
                    "status": status,
                    "total_citations": values["total_citations"],
                    "introspection": values["introspection"],
                    "inflow": values["inflow"],
                    "outflow": values["outflow"],
                    "introspection_ratio": values["introspection_ratio"],
                    "inflow_ratio": values["inflow_ratio"],
                    "outflow_ratio": values["outflow_ratio"],
                    "influence_score": values["influence_score"],
                    "growth_rate": growth_rate,
                }
            )
            previous_size[community_id] = current_size

        print(f"  Window {start_year}-{end_year}: {active_count} active communities")

    return pd.DataFrame(records), paper_to_year


def classify_global_community(community_data: pd.DataFrame):
    """Preserve the original global community classification thresholds."""
    active = community_data[community_data["status"] == "Active"].copy()
    if len(active) < 3:
        return ["Peripheral"] if len(active) == 0 else ["Insufficient Active Data"]

    time_points = active["window_center"].values
    papers = active["node_count"].values
    introspection = active["introspection_ratio"].values
    outflow = active["outflow_ratio"].values

    paper_slope = np.polyfit(time_points, papers, 1, w=np.sqrt(papers))[0]
    intro_slope = np.polyfit(time_points, introspection, 1, w=np.sqrt(papers))[0]
    outflow_slope = np.polyfit(time_points, outflow, 1, w=np.sqrt(papers))[0]
    avg_intro_outflow = np.mean([i / (o + 1e-6) for i, o in zip(introspection, outflow)])

    labels = []
    if np.median(outflow) > 0.30 and np.median(introspection) > 0.50:
        labels.append("Exporter")
    if np.median(active["inflow_ratio"]) > 0.25 and np.median(introspection) < 0.45:
        labels.append("Hub")
    if avg_intro_outflow > 3.6:
        labels.append("Insular")
    if paper_slope > 0.4:
        labels.append("Growing")
    elif paper_slope < -0.25:
        labels.append("Declining")
    if intro_slope > 0.08 and outflow_slope < -0.08:
        labels.append("Stagnating")
    elif outflow_slope > 0.1 and intro_slope < 0.05:
        labels.append("Opening")
    return labels or ["Stable"]


def classify_single_window(row):
    """Preserve the original per-window role/growth/size classification."""
    if row["status"] == "Dormant":
        return ["Dormant"]
    if row["status"] == "Latent":
        return ["Latent", "Small"]

    labels = []
    if row["outflow_ratio"] > 0.35 and row["introspection_ratio"] > 0.25:
        labels.append("Exporter")
    if row["inflow_ratio"] > 0.3 and row["introspection_ratio"] < 0.40:
        labels.append("Hub")
    if row["introspection_ratio"] / (row["outflow_ratio"] + 1e-6) > 4.0:
        labels.append("Insular")

    if row["growth_rate"] > 0.3:
        labels.append("Growing")
    elif row["growth_rate"] < -0.2:
        labels.append("Declining")

    if row["node_count"] < 50:
        labels.append("Small")
    elif row["node_count"] < 300:
        labels.append("Medium")
    else:
        labels.append("Large")

    if not any(role in labels for role in ["Exporter", "Hub", "Insular"]):
        labels.insert(0, "Stable")
    return labels


def add_classifications(temporal_df: pd.DataFrame) -> pd.DataFrame:
    result = temporal_df.copy()
    global_classes = {
        community_id: classify_global_community(group)
        for community_id, group in result.groupby("community")
    }
    result["global_community"] = result["community"].map(
        lambda value: "|".join(global_classes.get(value, ["Unknown"]))
    )
    result["window_class"] = result.apply(
        lambda row: "|".join(classify_single_window(row)), axis=1
    )
    return result


def analyze_growth_patterns(temporal_df: pd.DataFrame) -> pd.DataFrame:
    """Preserve the original growth-pattern CSV columns and classifications."""
    features = temporal_df.groupby("community").agg(
        {
            "node_count": ["mean", "std", "max"],
            "growth_rate": ["mean", "std"],
        }
    )
    features.columns = [
        "mean_size",
        "size_std",
        "max_size",
        "mean_growth",
        "growth_std",
    ]
    features = features.reset_index()

    def classify(row):
        if row["mean_size"] < 20:
            return "Small"
        if row["mean_growth"] > 0.3:
            return "Fast Growing"
        if row["mean_growth"] < -0.2:
            return "Declining"
        if row["growth_std"] > 0.4:
            return "Volatile"
        return "Stable"

    features["growth_pattern"] = features.apply(classify, axis=1)
    return features
