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
from src.extended_analysis import (
    analyze_community_roles,
    analyze_knowledge_flow,
    identify_emerging_community_sources,
    plot_community_lifecycle_heatmap,
    plot_community_state_distribution,
    plot_emergence_disappearance_trends,
    print_analysis_report,
    visualize_cross_community_flow,
)
from src.temporal_analysis import (
    add_classifications,
    analyze_growth_patterns,
    build_temporal_dataframe,
)
from src.visualization import (
    ensure_results_dir,
    plot_community_evolution,
    plot_community_network,
    plot_community_size_distribution,
    plot_community_topic_evolution,
    plot_growth_patterns,
    plot_network_statistics,
    plot_paper_temporal_distribution,
)


DATA_DIR = Path("data")
RESULTS_DIR = Path("results")


def main():
    ensure_results_dir(RESULTS_DIR)

    print("Loading citation data...")
    citations = load_citations(DATA_DIR / "cit-HepTh.txt")
    metadata = load_abstracts(DATA_DIR / "cit-HepTh-abstracts")

    valid_ids = set(metadata["paper_id"])
    original_edges = len(citations)
    citations = align_citations_with_metadata(citations, metadata)
    graph = build_citation_graph(citations)

    print(f"Number of valid paper IDs: {len(valid_ids)}")
    print(
        f"Filtered citations: Original {original_edges} → Remaining {len(citations)} "
        f"({len(citations) / original_edges:.1%})"
    )
    print(f"Total nodes in citation network: {graph.number_of_nodes():,}")
    print(f"Total edges in citation network: {graph.number_of_edges():,}")

    plot_network_statistics(graph, RESULTS_DIR)
    plot_paper_temporal_distribution(metadata, RESULTS_DIR)

    print("\nDetecting scientific communities...")
    assignments, community_sizes = detect_communities(graph)
    print(f"Communities detected: {len(community_sizes)}")
    print(f"Average community size: {sum(community_sizes.values()) / len(community_sizes):.2f}")
    print(f"Largest community: {max(community_sizes.values())} nodes")
    print(f"Smallest community: {min(community_sizes.values())} nodes")

    print("Top 5 largest communities:")
    for community_id, size in sorted(
        community_sizes.items(), key=lambda item: item[1], reverse=True
    )[:5]:
        print(f"  Community {community_id}: {size} nodes")

    plot_community_size_distribution(community_sizes, RESULTS_DIR)

    community_papers = map_community_papers(assignments, metadata)
    community_topics = analyze_major_community_topics(community_papers, metadata)

    print("\nMajor community topics:")
    for community_id, topic_info in list(community_topics.items())[:8]:
        print(f"\nCommunity {community_id} ({topic_info['num_papers']} papers):")
        print(f"  Key keywords: {', '.join(topic_info['community_keywords'][:6])}")
        for index, topic_keywords in enumerate(topic_info["topics"][:2]):
            print(f"  Topic {index + 1}: {', '.join(topic_keywords[:3])}")

    plot_community_network(
        graph,
        assignments,
        community_sizes,
        community_topics,
        RESULTS_DIR,
    )

    print("\nPerforming temporal analysis with 2-year sliding windows...")
    temporal_df, paper_to_year = build_temporal_dataframe(
        graph,
        metadata,
        assignments,
        window_size=2,
    )
    temporal_df = add_classifications(temporal_df)
    print(f"Generated temporal analysis with {len(temporal_df)} data points")
    print(f"Unique communities tracked: {temporal_df['community'].nunique()}")

    temporal_df.to_csv(RESULTS_DIR / "community_evolution.csv", index=False)

    flow_matrix, community_ids = build_cross_community_flow(graph, assignments)
    bridge_df = identify_bridge_communities(
        flow_matrix,
        community_ids,
        community_topics,
    )
    bridge_df.to_csv(RESULTS_DIR / "bridge_communities.csv", index=False)

    print("\nTop 5 Bridge Communities:")
    print(bridge_df.head(5))

    emerging_df = identify_emerging_community_sources(
        temporal_df,
        flow_matrix,
        community_ids,
        community_topics,
    )
    if not emerging_df.empty:
        emerging_df.to_csv(RESULTS_DIR / "emerging_communities.csv", index=False)
        print(f"\nIdentified and analyzed {len(emerging_df)} emerging communities.")
    else:
        print("\nNo emerging communities identified based on the original transition criteria.")

    role_over_time = analyze_community_roles(temporal_df, RESULTS_DIR)
    _ = role_over_time

    growth_features = analyze_growth_patterns(temporal_df)
    growth_features.to_csv(RESULTS_DIR / "growth_patterns.csv", index=False)
    plot_growth_patterns(growth_features, RESULTS_DIR)

    plot_emergence_disappearance_trends(temporal_df, RESULTS_DIR)
    plot_community_state_distribution(temporal_df, RESULTS_DIR)
    plot_community_lifecycle_heatmap(temporal_df, RESULTS_DIR)
    analyze_knowledge_flow(temporal_df, RESULTS_DIR)

    visualize_cross_community_flow(
        flow_matrix,
        community_ids,
        community_topics,
        RESULTS_DIR / "final_knowledge_flow.png",
    )

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

    print_analysis_report(
        temporal_df,
        growth_features,
        bridge_df,
        emerging_df,
    )

    print("Saved analysis results to CSV files")
    print("\nProject analysis completed successfully!")


if __name__ == "__main__":
    main()
