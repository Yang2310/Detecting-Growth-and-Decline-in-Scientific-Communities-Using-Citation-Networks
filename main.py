from pathlib import Path

from src.community_analysis import (
    analyze_major_community_topics,
    build_cross_community_flow,
    detect_communities,
    identify_bridge_communities,
    map_community_papers,
)
from src.data_loading import (
    align_citations_with_metadata,
    build_citation_graph,
    load_abstracts,
    load_citations,
)
from src.temporal_analysis import (
    add_classifications,
    analyze_growth_patterns,
    build_temporal_dataframe,
)
from src.visualization import (
    ensure_results_dir,
    plot_community_evolution,
    plot_community_size_distribution,
    plot_community_topic_evolution,
    plot_growth_patterns,
    plot_paper_temporal_distribution,
)


DATA_DIR = Path("data")
RESULTS_DIR = Path("results")


def main():
    ensure_results_dir(RESULTS_DIR)

    print("Loading citation data...")
    citations = load_citations(DATA_DIR / "cit-HepTh.txt")
    metadata = load_abstracts(DATA_DIR / "cit-HepTh-abstracts")
    citations = align_citations_with_metadata(citations, metadata)
    graph = build_citation_graph(citations)

    print(f"Papers with metadata: {len(metadata):,}")
    print(f"Citation edges after alignment: {len(citations):,}")

    plot_paper_temporal_distribution(metadata, RESULTS_DIR)

    print("Detecting scientific communities...")
    assignments, community_sizes = detect_communities(graph)
    print(f"Communities detected: {len(community_sizes):,}")
    plot_community_size_distribution(community_sizes, RESULTS_DIR)

    community_papers = map_community_papers(assignments, metadata)
    community_topics = analyze_major_community_topics(community_papers, metadata)

    print("Running temporal community analysis...")
    temporal_df, paper_to_year = build_temporal_dataframe(
        graph,
        metadata,
        assignments,
        window_size=2,
    )
    temporal_df = add_classifications(temporal_df)
    temporal_df.to_csv(RESULTS_DIR / "community_evolution.csv", index=False)

    growth_features = analyze_growth_patterns(temporal_df)
    growth_features.to_csv(RESULTS_DIR / "growth_patterns.csv", index=False)
    plot_growth_patterns(growth_features, RESULTS_DIR)

    flow_matrix, community_ids = build_cross_community_flow(graph, assignments)
    bridge_df = identify_bridge_communities(
        flow_matrix,
        community_ids,
        community_topics,
    )
    bridge_df.to_csv(RESULTS_DIR / "bridge_communities.csv", index=False)

    top_communities = (
        temporal_df.groupby("community")["node_count"]
        .max()
        .nlargest(5)
        .index
    )

    for community_id in top_communities:
        community_data = temporal_df[temporal_df["community"] == community_id]
        plot_community_evolution(community_id, community_data, RESULTS_DIR)
        plot_community_topic_evolution(
            community_id,
            community_data,
            metadata,
            community_papers,
            paper_to_year,
            RESULTS_DIR,
        )

    print("Analysis complete. Outputs saved to results/.")


if __name__ == "__main__":
    main()
