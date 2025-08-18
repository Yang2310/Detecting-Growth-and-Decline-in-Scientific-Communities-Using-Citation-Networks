import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import leidenalg
from igraph import Graph
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import defaultdict, Counter


# Loading reference relationships
cit_df = pd.read_csv(
    "data/cit-HepTh.txt",
    sep="\t",
    comment='#',
    header=None,
    names=["source", "target"],
    dtype={"source": str, "target": str}
)

cit_df["source"] = cit_df["source"].str.strip()
cit_df["target"] = cit_df["target"].str.strip()

# Loading metadata
def load_abstracts(root_dir):
    """Loading metadata files stratified by year"""
    metadata = []

    # Iterate through all the year folders in the root directory
    for year_dir in os.listdir(root_dir):
        year_path = os.path.join(root_dir, year_dir)

        # Ensure it's a directory and the year part is numeric
        if os.path.isdir(year_path) and year_dir.isdigit():
            year = int(year_dir)

            # Traverse all .abs files in this year directory
            for filename in os.listdir(year_path):
                if filename.endswith(".abs"):
                    # Remove leading zeros from paper ID
                    paper_id = filename.split(".")[0].strip().lstrip('0')
                    file_path = os.path.join(year_path, filename)

                    # Read file content
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    
                    # Read file content
                    title = ""
                    authors = ""
                    abstract = ""
                    
                    lines = content.split("\n")

                    # Parse title line
                    for i, line in enumerate(lines):
                        if line.startswith("Title: "):
                            title = line.replace("Title: ", "").strip()
                            break
                    
                    # Parse authors line 
                    for i, line in enumerate(lines):
                        if line.startswith("Authors: "):
                            authors = line.replace("Authors: ", "").strip()
                            break
                    
                    # Parse abstract
                    separator_count = 0
                    abstract_lines = []
                    for line in lines:
                        if line.strip() == "\\\\":
                            separator_count += 1
                            if separator_count == 3:
                                break
                            continue 

                        if separator_count == 2:
                            abstract_lines.append(line.strip())
                    
                    abstract = " ".join(abstract_lines).strip()

                    # Only add valid records with title and abstract
                    if title and abstract:
                        metadata.append({
                            "paper_id": paper_id,
                            "year": year,
                            "title": title,
                            "authors": authors,
                            "abstract": abstract
                        })
                    else:
                        print(f"Parsing failed: {file_path} - Title: '{title}', Abstract length: {len(abstract)}")

    
    return pd.DataFrame(metadata)

metadata_df = load_abstracts("data/cit-HepTh-abstracts")

# Create set of valid IDs
valid_ids = set(metadata_df['paper_id'])
print(f"Number of valid paper IDs: {len(valid_ids)}")

# Filter citations to only include valid IDs
original_edges = len(cit_df)
cit_df = cit_df[
    cit_df["source"].isin(valid_ids) & 
    cit_df["target"].isin(valid_ids)
]
print(f"Filtered citations: Original {original_edges} → Remaining {len(cit_df)} ({len(cit_df)/original_edges:.1%})")

print("\nFirst few record titles:")
for i in range(min(3, len(metadata_df))):
    print(f"{i+1}. {metadata_df.iloc[i]['title']}")
    print(f"{metadata_df.iloc[i]['abstract']}\n")

# Create results directory if it doesn't exist
if not os.path.exists("results"):
    os.makedirs("results")
    print("Created results directory")

# Build the base citation network
G = nx.from_pandas_edgelist(cit_df, 'source', 'target', create_using=nx.DiGraph())

#Plot basic network statistics
plt.figure(figsize=(12, 4))

# Subplot 1: Number of nodes and edges
plt.subplot(131)
plt.bar(['Nodes', 'Edges'], [G.number_of_nodes(), G.number_of_edges()], color=['skyblue', 'salmon'])
plt.title("Network Basic Structure")

# Subplot 2: In-degree distribution (log scale)
plt.subplot(132)
in_degrees = dict(G.in_degree()).values()
plt.hist(in_degrees, bins=50, log=True, color='teal')
plt.xlabel('In-degree')
plt.ylabel('Frequency (log)')
plt.title("Citation In-degree Distribution")

# Subplot 3: Self-citation ratio check
self_loops = nx.number_of_selfloops(G)
total_edges = G.number_of_edges()
plt.subplot(133)
plt.pie([self_loops, total_edges-self_loops], 
        labels=['Self-citations', 'Normal citations'],
        autopct='%1.1f%%', 
        colors=['gold', 'lightcoral'])
plt.title("Self-citation Ratio")

plt.tight_layout()
plt.savefig("results/network_statistics.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved network_statistics.png in results directory")


# Plotting time distribution histograms
plt.figure(figsize=(10, 5))
plt.hist(metadata_df['year'], bins=range(1992, 2005), edgecolor='black', align='left')
plt.xticks(range(1992, 2004))
plt.xlabel('Year')
plt.ylabel('Number of Papers')
plt.title('Temporal Distribution of Papers')
plt.grid(axis='y', alpha=0.75)
plt.savefig("results/paper_temporal_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved paper_temporal_distribution.png in results directory")



# =============================================================================
# Community Detection
# =============================================================================

# Convert to undirected graph for community detection
G_undir = G.to_undirected()
G_undir.remove_edges_from(nx.selfloop_edges(G_undir))

edges = list(G_undir.edges())
g_igraph = Graph.TupleList(edges, directed=False)

# Run Leiden algorithm
partition = leidenalg.find_partition(
    g_igraph, 
    leidenalg.ModularityVertexPartition,
    n_iterations=5,            # Increase iterations to improve quality
    seed=42                    # Fix random seed for reproducibility
)

# Create community mapping dictionary
comms = {}
for i, community in enumerate(partition):
    for node_id in community:
        node_name = g_igraph.vs[node_id]['name']
        comms[node_name] = i

# Analyze community statistics
comm_sizes = {}
for node, cid in comms.items():
   comm_sizes[cid] = comm_sizes.get(cid, 0) + 1

print(f"\nCommunities detected: {len(comm_sizes)}")
print(f"Average community size: {sum(comm_sizes.values()) / len(comm_sizes):.2f}")
print(f"Largest community: {max(comm_sizes.values())} nodes")
print(f"Smallest community: {min(comm_sizes.values())} nodes")

# Show largest communities
sorted_comms = sorted(comm_sizes.items(), key=lambda x: x[1], reverse=True)
print("Top 5 largest communities:")
for i, (cid, size) in enumerate(sorted_comms[:5]):
   print(f"  Community {cid}: {size} nodes")

# Visualization
# Define size bins
size_bins = [
    (1, 5, "1-5"),
    (6, 10, "6-10"), 
    (11, 50, "11-50"),
    (50, 150, "50-200"),
    (150, 500, "150-500"),
    (500, float('inf'), "500+")
]

# Count communities in each size range
size_distribution = []
for min_size, max_size, label in size_bins:
    if max_size == float('inf'):
        count = sum(1 for size in comm_sizes.values() if size >= min_size)
    else:
        count = sum(1 for size in comm_sizes.values() if min_size <= size <= max_size)
    
    percentage = count / len(comm_sizes) * 100
    size_distribution.append({
        'range': label,
        'count': count,
        'percentage': percentage
    })

# Convert to DataFrame
size_dist_df = pd.DataFrame(size_distribution)

# Create visualization
plt.figure(figsize=(10, 6))
bars = plt.bar(size_dist_df['range'], size_dist_df['count'], 
               color='skyblue', edgecolor='navy', alpha=0.7)

# Add count and percentage labels on each bar
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{int(height)}\n({size_dist_df.iloc[i]["percentage"]:.1f}%)', 
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.xlabel('Community Size (nodes)')
plt.ylabel('Number of Communities')
plt.title('Distribution of Community Sizes')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("results/community_size_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved community size distribution to results/community_size_distribution.png")


# =============================================================================
# Thematic analysis
# =============================================================================


# Create mapping from communities to papers
comm_papers = defaultdict(list)
for paper_id, comm_id in comms.items():
    if paper_id in metadata_df['paper_id'].values:
        comm_papers[comm_id].append(paper_id)

print(f"Created paper mappings for {len(comm_papers)} communities")

def extract_comm_topics(comm_id, paper_ids, metadata_df, n_topics=3):
    # Extract main topics for communities
    if len(paper_ids) < 10:
        print(f"  Community {comm_id}: Too few papers ({len(paper_ids)}) for topic analysis")
        return None
    
    # Get all abstracts and titles
    comm_metadata = metadata_df[metadata_df['paper_id'].isin(paper_ids)]
    abstracts = comm_metadata['abstract'].tolist()
    titles = comm_metadata['title'].tolist()

    # Combine titles and abstracts for analysis
    texts = []
    for title, abstract in zip(titles, abstracts):
        combined_text = title + " " + abstract
        texts.append(combined_text)

    try:
        # TF-IDF feature extraction
        vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )

        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        # Perform topic modeling using LDA
        n_topics_actual = min(n_topics, len(texts) // 5, 5)
        if n_topics_actual < 2:
            n_topics_actual = 2
        
        lda = LatentDirichletAllocation(
            n_components=n_topics_actual,
            random_state=42,
            max_iter=50
        )
        
        lda.fit(tfidf_matrix)

        # Extract topic keywords and weights
        topics = []
        topic_weights = []
        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-8:][::-1]
            top_keywords = [feature_names[i] for i in top_indices]
            top_weights = [topic[i] for i in top_indices]
            topics.append(top_keywords)
            topic_weights.append(top_weights)
        
        # Get most frequent keywords as community labels
        all_keywords = []
        for topic in topics:
            all_keywords.extend(topic[:3])
        
        keyword_counts = Counter(all_keywords)
        top_community_keywords = [kw for kw, count in keyword_counts.most_common(8)]
        
        print(f"  Community {comm_id}: {len(paper_ids)} papers, keywords: {', '.join(top_community_keywords[:4])}")
        
        return {
            'topics': topics,
            'topic_weights': topic_weights,
            'community_keywords': top_community_keywords,
            'keyword_counts': keyword_counts,
            'num_papers': len(paper_ids)
        }
    
    except Exception as e:
        print(f"  Error analyzing community {comm_id}: {e}")
        return None
    
# Topics for major communities
comm_topics = {}
major_communities = sorted(comm_papers.items(), key=lambda x: len(x[1]), reverse=True)

for i, (comm_id, paper_ids) in enumerate(major_communities[:20]):  # Analyze top 15 largest communities
    topic_result = extract_comm_topics(comm_id, paper_ids, metadata_df)
    if topic_result:
        comm_topics[comm_id] = topic_result

print("\nMajor community topics:")
for comm_id, topic_info in list(comm_topics.items())[:8]:
    print(f"\nCommunity {comm_id} ({topic_info['num_papers']} papers):")
    print(f"  Key keywords: {', '.join(topic_info['community_keywords'][:6])}")
    
    # Main topics
    for j, topic_keywords in enumerate(topic_info['topics'][:2]):
        print(f"  Topic {j+1}: {', '.join(topic_keywords[:3])}")


# Visualization
# Select the top 15 largest communities for visualization
top_comm_ids = [cid for cid, _ in sorted_comms[:20]]

# Create community graph
G_comm = nx.Graph()

# Add community nodes
for comm_id in top_comm_ids:
    if comm_id in comm_topics:
        keywords = comm_topics[comm_id]['community_keywords'][:3]
        label = f"C{comm_id}: " + ", ".join(keywords)
        node_size = comm_topics[comm_id]['num_papers'] * 0.5  # Adjust node size based on community size
        G_comm.add_node(comm_id, label=label, size=node_size)

# Add edges between communities (based on cross-community citations)
for source, target in G.edges():
    if source in comms and target in comms:
        comm_source = comms[source]
        comm_target = comms[target]
        if comm_source != comm_target and comm_source in top_comm_ids and comm_target in top_comm_ids:
            if G_comm.has_edge(comm_source, comm_target):
                G_comm[comm_source][comm_target]['weight'] += 1
            else:
                G_comm.add_edge(comm_source, comm_target, weight=1)

# Skip visualization if too few edges
if G_comm.number_of_edges() < 5:
    print("Warning: Too few edges between top communities. Skipping visualization.")
else:
    plt.figure(figsize=(20, 15))
    pos = nx.spring_layout(G_comm, seed=42, k=0.5)  # Layout algorithm
    
    # Node sizes and colors
    node_sizes = [G_comm.nodes[node]['size'] for node in G_comm.nodes()]
    node_colors = [hash(node) % 30 for node in G_comm.nodes()]  # Generate different colors for communities
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G_comm, pos, 
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.tab20,
        alpha=0.8
    )
    
    # Draw edges (width based on citation count)
    edge_weights = [G_comm[u][v]['weight'] * 0.002 for u, v in G_comm.edges()]
    nx.draw_networkx_edges(
        G_comm, pos, 
        width=edge_weights,
        edge_color='gray',
        alpha=0.6
    )
    
    # Add node labels (community ID and topics)
    labels = {node: G_comm.nodes[node]['label'] for node in G_comm.nodes()}
    nx.draw_networkx_labels(
        G_comm, pos, 
        labels=labels,
        font_size=6,
        font_family='sans-serif',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
    )
    
    # Add graph title
    plt.title("Community Network with Topical Labels", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("results/community_network.png", dpi=300)
    plt.show()
    print("Saved community_network.png in results directory")



# =============================================================================
# COIN model analysis
# =============================================================================

# Create mapping from paper ID to year and community
paper_to_year = dict(zip(metadata_df['paper_id'], metadata_df['year']))
paper_to_community = comms
years = sorted(metadata_df['year'].dropna().unique())

print(f"\nTime range: {min(years)} - {max(years)}")
print(f"Total papers with year info: {len(paper_to_year)}")
print(f"Papers in communities: {len(paper_to_community)}")

print("\nData Integrity Check")

# 1. Basic statistics
print(f"Total nodes in citation network: {len(G.nodes):,}")
print(f"Total edges in citation network: {len(G.edges):,}")
print(f"Year data entries: {len(paper_to_year):,}")
print(f"Community data entries: {len(paper_to_community):,}")

# 2. ID format check
print(f"\nID Format Samples:")
print(f"Citation network node samples: {list(G.nodes)[:5]}")
print(f"Year data ID samples: {list(paper_to_year.keys())[:5]}")
print(f"Community data ID samples: {list(paper_to_community.keys())[:5]}")

# 3. Intersection analysis
network_nodes = set(G.nodes)
year_nodes = set(paper_to_year.keys())
community_nodes = set(paper_to_community.keys())

print(f"\nData Intersection Analysis:")
print(f"Network ∩ Year data: {len(network_nodes & year_nodes):,} ({len(network_nodes & year_nodes)/len(network_nodes):.1%})")
print(f"Network ∩ Community data: {len(network_nodes & community_nodes):,} ({len(network_nodes & community_nodes)/len(network_nodes):.1%})")
print(f"Year data ∩ Community data: {len(year_nodes & community_nodes):,}")

# 4. Missing data analysis
missing_years = network_nodes - year_nodes
missing_communities = network_nodes - community_nodes

print(f"\nMissing Data Samples:")
print(f"Nodes missing year information: {list(missing_years)[:10]}")
print(f"Nodes missing community information: {list(missing_communities)[:10]}")

# 5. Year distribution check
print(f"\nYear Distribution:")
year_counts = pd.Series(list(paper_to_year.values())).value_counts().sort_index()
for year, count in year_counts.items():
    print(f"  {year}: {count:,} papers")


def calculate_coin_metrics(year_start, year_end, citation_graph, paper_to_year, paper_to_community):
    """Calculate COIN metrics for each community in given time window"""
    # Get valid nodes within the current time window
    valid_nodes = set(
        pid for pid, year in paper_to_year.items()
        if year_start <= year <= year_end
    )
    
    # Filter edges within time window
    edges_in_window = []
    for source, target in citation_graph.edges():
        # Source node must be in current time window
        if source in valid_nodes:
            # Target node must exist in metadata
            if target in paper_to_year:  # Ensure target node has year information
                edges_in_window.append((source, target))
    
    print(f"\nProcessing {len(edges_in_window)} edges in window {year_start}-{year_end}")

    # Initialize COIN metrics storage
    community_metrics = defaultdict(lambda: {
        'introspection': 0,
        'inflow': 0,
        'outflow': 0,
        'total_citations': 0,
        'node_count': 0
    })

    # Count papers per community in window
    for paper_id, year in paper_to_year.items():
        if year_start <= year <= year_end and paper_id in paper_to_community:
            comm_id = paper_to_community[paper_id]
            community_metrics[comm_id]['node_count'] += 1

    # Analyze citation relationships
    for source, target in edges_in_window:
        if source in paper_to_community and target in paper_to_community:
            source_comm = paper_to_community[source]
            target_comm = paper_to_community[target]
            
            community_metrics[source_comm]['total_citations'] += 1
            
            if source_comm == target_comm:
                community_metrics[source_comm]['introspection'] += 1
            else:
                community_metrics[source_comm]['outflow'] += 1
                community_metrics[target_comm]['inflow'] += 1

    # Calculate COIN ratios
    for comm_id in community_metrics:
        metrics = community_metrics[comm_id]
        total_cit = metrics['total_citations']
        
        if total_cit > 0:
            metrics['introspection_ratio'] = metrics['introspection'] / total_cit
            metrics['inflow_ratio'] = metrics['inflow'] / total_cit
            metrics['outflow_ratio'] = metrics['outflow'] / total_cit
        else:
            metrics['introspection_ratio'] = 0
            metrics['inflow_ratio'] = 0
            metrics['outflow_ratio'] = 0
            
        # Calculate influence score
        metrics['influence_score'] = metrics['node_count'] + metrics['inflow'] * 0.5
        
    return dict(community_metrics)

# # Simple test
# test_edges = [
#     ('p1', 'p2'), ('p2', 'p1'),  # Community A introspection
#     ('p3', 'p4'),               # Community B introspection
#     ('p3', 'p1'),               # B->A (B outflow, A inflow)
# ]

# test_paper_to_year = {'p1': 2000, 'p2': 2000, 'p3': 2000, 'p4': 2000}
# test_paper_to_community = {'p1': 'A', 'p2': 'A', 'p3': 'B', 'p4': 'B'}

# test_graph = nx.DiGraph()
# test_graph.add_edges_from(test_edges)

# print(f"\nTest data: {len(test_edges)} citations")
# print(f"Community A: p1,p2  Community B: p3,p4")
# print(f"Citation relationships: {test_edges}")

# # Run test
# results = calculate_coin_metrics(2000, 2000, test_graph, test_paper_to_year, test_paper_to_community)

# for comm, metrics in results.items():
#     print(f"Community {comm}: papers={metrics['total_papers']}, "
#           f"introspection={metrics['introspection']}, inflow={metrics['inflow']}, outflow={metrics['outflow']}")

# # Test visualization
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# pos = {'p1': (0, 1), 'p2': (0, 0), 'p3': (2, 1), 'p4': (2, 0)}
# colors = ['lightblue', 'lightblue', 'lightgreen', 'lightgreen']
# nx.draw(test_graph, pos, ax=ax1, node_color=colors, with_labels=True, arrows=True)
# ax1.set_title('Citation Network (Community A=blue, B=green)')

# # COIN metrics comparison
# communities = list(results.keys())
# introspection = [results[i]['introspection'] for i in communities]
# inflow = [results[i]['inflow'] for i in communities]
# outflow = [results[i]['outflow'] for i in communities]

# x = range(len(communities))
# ax2.bar([i-0.2 for i in x], introspection, 0.2, label='introspection', alpha=0.7)
# ax2.bar(x, inflow, 0.2, label='inflow', alpha=0.7)  
# ax2.bar([i+0.2 for i in x], outflow, 0.2, label='outflow', alpha=0.7)
# ax2.set_xticks(x)
# ax2.set_xticklabels(communities)
# ax2.legend()
# ax2.set_title('COIN Metrics')

# plt.tight_layout()
# plt.show()

# print(f"\nvalidation:")
# print(f"Community A introspection (p1<->p2): {results['A']['introspection'] == 2}")
# print(f"Community A inflow (p3->p1): {results['A']['inflow'] == 1}")  
# print(f"Community B outflow (p3->p1): {results['B']['outflow'] == 1}")
# print(f"COIN ratio sums: A={results['A']['introspection_ratio'] + results['A']['inflow_ratio']:.2f}, B={results['B']['outflow_ratio']:.2f}")


def build_cross_community_flow(citation_graph, comm_assignments):
    """Build an inter-community knowledge flow matrix"""
    
    # Initialize flow matrix
    comm_ids = sorted(set(comm_assignments.values()))
    flow_matrix = np.zeros((len(comm_ids), len(comm_ids)), dtype=int)
    comm_index = {cid: idx for idx, cid in enumerate(comm_ids)}
    
    # Count cross-community citations
    for src, tgt in citation_graph.edges():
        if src in comm_assignments and tgt in comm_assignments:
            src_comm = comm_assignments[src]
            tgt_comm = comm_assignments[tgt]
            
            if src_comm != tgt_comm:  # Only count cross-community citations
                src_idx = comm_index[src_comm]
                tgt_idx = comm_index[tgt_comm]
                flow_matrix[src_idx][tgt_idx] += 1
                
    return flow_matrix, comm_ids

def identify_bridge_communities(flow_matrix, comm_ids, comm_topics):
    """Identify knowledge bridge communities"""
    bridge_scores = []
    flow_matrix = flow_matrix.astype(np.float64)  # Convert to float to avoid integer overflow
    
    for i, comm_id in enumerate(comm_ids):
        # Calculate knowledge hub index
        out_flow = flow_matrix[i].sum()  # Total outflow
        in_flow = flow_matrix[:, i].sum()  # Total inflow
        diversity = np.count_nonzero(flow_matrix[i])  # Number of connected communities
        
        # Hub score = outflow * connection diversity
        hub_score = out_flow * diversity / max(1, (len(comm_ids) - 1))  # Avoid division by zero
        
        # Calculate knowledge mediation index (with numerical safeguards)
        mediation = 0.0
        for j in range(len(comm_ids)):
            if j != i:
                for k in range(len(comm_ids)):
                    if k != i and k != j:
                        # Numerical safeguard: only calculate when all values > 0
                        if flow_matrix[j][k] > 0 and flow_matrix[j][i] > 0 and flow_matrix[i][k] > 0:
                            # Use logarithms to avoid large number multiplication
                            log_mediation = (
                                np.log1p(flow_matrix[j][k]) +
                                np.log1p(flow_matrix[j][i]) +
                                np.log1p(flow_matrix[i][k]))
                            mediation += np.exp(log_mediation)
        
        bridge_score = hub_score + mediation * 0.1
        community_name = '|'.join(comm_topics.get(comm_id, {}).get('community_keywords', [])[:2])
        
        bridge_scores.append({
            'community': comm_id,
            'hub_score': hub_score,
            'mediation': mediation,
            'bridge_score': bridge_score,
            'community_name': community_name
        })
    
    return pd.DataFrame(bridge_scores).sort_values('bridge_score', ascending=False)


# Perform temporal window analysis
print("\nPerforming temporal analysis with 2-year sliding windows...")
window_size = 2
temporal_data = []
all_community_ids = set(comms.values())
min_year = int(min(years))
max_year = int(max(years))

# Track previous window size for each community
prev_window_size = defaultdict(int)

for start_year in range(min_year, max_year - window_size + 2):
    end_year = start_year + window_size - 1
    if end_year > max_year:
        break
        
    # Get COIN metrics for this time window
    coin_metrics = calculate_coin_metrics(start_year, end_year, G, paper_to_year, paper_to_community)
    
    # Inside the time window loop 
    flow_matrix, comm_ids = build_cross_community_flow(G, comms)

    # Store flow matrix for subsequent analysis
    if 'flow_matrices' not in locals():
        flow_matrices = {}
    flow_matrices[(start_year, end_year)] = flow_matrix


    active_communities_count = 0
    for comm_id in all_community_ids:
        # Safely get metrics using .get(), providing default values if the community is inactive in this window
        metrics = coin_metrics.get(comm_id, {
            'node_count': 0, 'total_citations': 0, 'introspection': 0,
            'inflow': 0, 'outflow': 0, 'introspection_ratio': 0,
            'inflow_ratio': 0, 'outflow_ratio': 0, 'influence_score': 0
        })

        current_size = metrics['node_count']
        
        # Define community status
        status = ''
        if current_size >= 5:  # Active threshold
            status = 'Active'
            active_communities_count += 1
        elif current_size > 0:
            status = 'Latent'  # Dormant/budding
        else:
            status = 'Dormant'  # Inactive

        # Calculate growth rate
        prev_size = prev_window_size[comm_id]
        growth_rate = (current_size - prev_size) / prev_size if prev_size > 0 else 0.0

        temporal_data.append({
            'community': comm_id,
            'year_start': start_year,
            'year_end': end_year,
            'window_center': (start_year + end_year) / 2,
            'node_count': current_size,
            'status': status,  
            'total_citations': metrics['total_citations'],
            'introspection': metrics['introspection'],
            'inflow': metrics['inflow'],
            'outflow': metrics['outflow'],
            'introspection_ratio': metrics['introspection_ratio'],
            'inflow_ratio': metrics['inflow_ratio'],
            'outflow_ratio': metrics['outflow_ratio'],
            'influence_score': metrics['influence_score'],
            'growth_rate': growth_rate
        })
        
        # Update previous window size
        prev_window_size[comm_id] = current_size

    print(f"  Window {start_year}-{end_year}: {active_communities_count} active communities")

temporal_df = pd.DataFrame(temporal_data)
print(f"Generated temporal analysis with {len(temporal_df)} data points")
print(f"Unique communities tracked: {temporal_df['community'].nunique()}")

# Identify bridge communities
bridge_df = identify_bridge_communities(flow_matrix, comm_ids, comm_topics)
print("\nTop 5 Bridge Communities:")
print(bridge_df.head(5))


# =============================================================================
# Community Classification
# =============================================================================

def classify_global_community(comm_data):
    # [Modified] Perform trend analysis only on active periods of the community
    active_data = comm_data[comm_data['status'] == 'Active'].copy()

    # If there are fewer than 3 active data points, long-term trends cannot be determined
    if len(active_data) < 3:
        if len(active_data) == 0:
            return ['Peripheral']  # Edge community that was never active
        return ['Insufficient Active Data']
    
    # All subsequent calculations are based on active_data
    time_points = active_data['window_center'].values
    papers = active_data['node_count'].values
    intro = active_data['introspection_ratio'].values
    outflow = active_data['outflow_ratio'].values
    
    
    paper_slope = np.polyfit(time_points, papers, 1, w=np.sqrt(papers))[0]
    intro_slope = np.polyfit(time_points, intro, 1, w=np.sqrt(papers))[0]
    outflow_slope = np.polyfit(time_points, outflow, 1, w=np.sqrt(papers))[0]
    
    safe_ratio = lambda a, b: a/(b+1e-6)
    avg_intro_outflow = np.mean([safe_ratio(i, o) for i, o in zip(intro, outflow)])
    
    # Core role classification
    classifications = []
    
    if np.median(outflow) > 0.30 and np.median(intro) > 0.50:
        classifications.append('Exporter')
    
    if np.median(active_data['inflow_ratio']) > 0.25 and np.median(intro) < 0.45:
        classifications.append('Hub')
    
    if avg_intro_outflow > 3.6:
        classifications.append('Insular')
    
    # Status classification
    if paper_slope > 0.4:
        classifications.append('Growing')
    elif paper_slope < -0.25:
        classifications.append('Declining')
    
    if intro_slope > 0.08 and outflow_slope < -0.08:
        classifications.append('Stagnating')
    elif outflow_slope > 0.1 and intro_slope < 0.05:
        classifications.append('Opening')
    
    return classifications if classifications else ['Stable']


# Apply classifications to each community
community_classes = {}
for comm_id in temporal_df['community'].unique():
    comm_data = temporal_df[temporal_df['community'] == comm_id]
    community_classes[comm_id] = classify_global_community(comm_data)

# Add classification results to main dataframe
temporal_df['global_community'] = temporal_df['community'].map(
    lambda x: '|'.join(community_classes.get(x, ['Unknown'])))


def classify_single_window(window_data):
    """Classify based on a single time window, with status and size."""
    status = window_data['status']
    node_count = window_data['node_count']
    
    # 1. Handle inactive statuses
    if status == 'Dormant':
        return ['Dormant']
    if status == 'Latent':
        return ['Latent', 'Small']

    # 2. Process active status
    classifications = []
    
    # Core role classification
    outflow_ratio = window_data['outflow_ratio']
    inflow_ratio = window_data['inflow_ratio']
    introspection_ratio = window_data['introspection_ratio']
    
    # 1. Role classification
    if outflow_ratio > 0.35 and introspection_ratio > 0.25:
        classifications.append('Exporter')
    if inflow_ratio > 0.3 and introspection_ratio < 0.40:
        classifications.append('Hub')
    if introspection_ratio / (outflow_ratio + 1e-6) > 4.0:
        classifications.append('Insular')
    

    # Growth status classification
    growth_rate = window_data['growth_rate']
    if growth_rate > 0.3:
        classifications.append('Growing')
    elif growth_rate < -0.2:
        classifications.append('Declining')

    # Size classification
    if node_count < 50:
        classifications.append('Small')
    elif node_count < 300:
        classifications.append('Medium')
    else:
        classifications.append('Large')
        
    # Add 'Stable' as default role if no core role identified
    if not any(role in classifications for role in ['Exporter', 'Hub', 'Insular']):
        classifications.insert(0, 'Stable')

    return classifications

# Apply window-level classification (independent for each time window)
temporal_df['window_class'] = temporal_df.apply(
    lambda row: '|'.join(classify_single_window(row)), 
    axis=1
)


# =============================================================================
# Analysis and visualisation
# =============================================================================
# Ensure we have the temporal data DataFrame
print("\nStarting community evolution analysis and visualization...")

# 1. Main community evolution trajectory visualization
def plot_community_evolution(comm_id, comm_data):
    """Plot evolution timeline for a single community (split into two subplots)"""
    if len(comm_data) < 2:
        print(f"Skipping community {comm_id}: insufficient data points")
        return
    
    # Reset index to ensure continuous positional indexing
    comm_data = comm_data.reset_index(drop=True)
    # Ensure data is sorted chronologically
    comm_data = comm_data.sort_values('window_center')
    
    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
    
    # First subplot: Ratio trends
    ax1.plot(comm_data['window_center'], comm_data['introspection_ratio'], '^-', color='red', markersize=8, label='Introspection Ratio')
    ax1.plot(comm_data['window_center'], comm_data['inflow_ratio'], 's-', color='green', markersize=8, label='In-flow Ratio')
    ax1.plot(comm_data['window_center'], comm_data['outflow_ratio'], 'D-', color='purple', markersize=8, label='Out-flow Ratio')
    ax1.set_ylabel('Ratio')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_title(f'Knowledge Flow Ratios of Community {comm_id}')

    # Second subplot: Community size evolution
    # Plot community size
    ax2.plot(comm_data['window_center'], comm_data['node_count'], 'o-', color='royalblue', linewidth=2, markersize=8, label='Community Size')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Community Size')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_title(f'Size Evolution of Community {comm_id}')
    
    # Add global classification label
    global_class = comm_data.iloc[0]['global_community']
    ax2.text(
        0.95, 0.95, 
        f"Global Classification: {global_class}",
        transform=ax2.transAxes,
        fontsize=12,
        ha='right',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
    )
    
    # Annotate window-level classifications
    for pos_idx in range(len(comm_data)):
        row = comm_data.iloc[pos_idx]
        
        # [Modified] Clean labels by removing redundant 'Active'
        raw_label = '|'.join(classify_single_window(row))  # Ensure latest classification logic is used
        label_parts = raw_label.split('|')
        filtered_parts = [part for part in label_parts if part != 'Active']
        clean_label = '|'.join(filtered_parts)
        
        # Add window classification above data point
        ax2.annotate(
            clean_label,  # <-- Use cleaned label
            xy=(row['window_center'], row['node_count'] * 1.05),
            ha='center', 
            va='bottom',
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", alpha=0.7)
        )
        
        # Add trend change markers
        if pos_idx > 0:
            prev_row = comm_data.iloc[pos_idx-1]
            if row['window_class'] != prev_row['window_class']:
                # Add vertical line on both subplots
                ax1.axvline(x=row['window_center'], color='gray', linestyle=':', alpha=0.7)
                ax2.axvline(x=row['window_center'], color='gray', linestyle=':', alpha=0.7)
                # Add text annotation on second subplot
                ax2.text(
                    row['window_center'], 
                    ax2.get_ylim()[1] * 0.95,
                    f"Change: {prev_row['window_class']} → {row['window_class']}",
                    ha='center', 
                    va='top', 
                    rotation=90,
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.5)
                )
        
    plt.tight_layout()
    plt.savefig(f"results/community_{comm_id}_evolution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved evolution plot for community {comm_id}")

def plot_community_topic_evolution(comm_id, temporal_data, metadata_df):
    """Plot the topic evolution for a community using a heatmap matrix"""
    # Ensure data is sorted chronologically
    comm_data = temporal_data.sort_values('window_center')
    
    # Collect keywords from all time windows
    all_keywords = set()
    topic_data = {}
    
    for _, row in comm_data.iterrows():
        start_year = row['year_start']
        end_year = row['year_end']
        window_key = f"{start_year}-{end_year}"
        
        # Get papers in this community for current time window
        window_paper_ids = [
            pid for pid in comm_papers[comm_id]
            if start_year <= paper_to_year.get(pid, -1) <= end_year
        ]
        
        if len(window_paper_ids) < 5:  # Skip if too few papers
            topic_data[window_key] = {}
            continue
        
        # Extract text data
        window_metadata = metadata_df[metadata_df['paper_id'].isin(window_paper_ids)]
        texts = (window_metadata['title'] + " " + window_metadata['abstract']).tolist()
        
        # Extract keywords using TF-IDF (without topic modeling)
        try:
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Get average TF-IDF scores for each keyword
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
            
            # Store keywords and their scores
            window_topics = {}
            for i, score in enumerate(tfidf_scores):
                if score > 0.01:  # Only keep meaningful scores
                    keyword = feature_names[i]
                    window_topics[keyword] = score
                    all_keywords.add(keyword)
            
            topic_data[window_key] = window_topics
        except Exception as e:
            print(f"Error extracting topics for community {comm_id} in {window_key}: {e}")
            topic_data[window_key] = {}
    
    # Create heatmap matrix
    all_keywords = sorted(all_keywords)  # Sort keywords
    windows = sorted(topic_data.keys())  # Sort time windows
    
    # Create matrix: rows=keywords, columns=time windows
    heatmap_matrix = np.zeros((len(all_keywords), len(windows)))
    
    for j, window in enumerate(windows):
        window_topics = topic_data[window]
        for i, keyword in enumerate(all_keywords):
            if keyword in window_topics:
                heatmap_matrix[i, j] = window_topics[keyword]
    
    # Keep only top 20 most important keywords (avoid clutter)
    if len(all_keywords) > 20:
        # Calculate total keyword importance score
        keyword_scores = np.sum(heatmap_matrix, axis=1)
        top_indices = np.argsort(keyword_scores)[-20:][::-1]
        heatmap_matrix = heatmap_matrix[top_indices, :]
        all_keywords = [all_keywords[i] for i in top_indices]
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Use seaborn to plot heatmap
    ax = sns.heatmap(
        heatmap_matrix,
        cmap="YlGnBu",  # Yellow to blue gradient
        linewidths=0.5,
        linecolor="lightgray",
        xticklabels=windows,
        yticklabels=all_keywords,
        cbar_kws={'label': 'Keyword Importance (TF-IDF)'}
    )
    
    # Set title and labels
    plt.title(f"Topic Evolution in Community {comm_id}", fontsize=14)
    plt.xlabel("Time Window", fontsize=12)
    plt.ylabel("Keywords", fontsize=12)
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"results/community_{comm_id}_topic_evolution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved topic evolution plot for community {comm_id}")


# 2. Community role distribution analysis
def analyze_community_roles(temporal_df):
    """Analyze changes in community roles over time (simplified version)"""
    # Core role list
    core_roles = ['Exporter', 'Insular', 'Hub', 'Declining', 'Growing']
    
    # Create new columns marking whether each community has each core attribute in each time window
    for role in core_roles:
        temporal_df[role] = temporal_df['window_class'].apply(
            lambda x: 1 if role in x else 0
        )
    
    # Count communities with each core attribute per time window
    role_over_time = temporal_df.groupby('window_center')[core_roles].sum()
    
    # Plot role changes over time
    plt.figure(figsize=(12, 7))
    for role in core_roles:
        plt.plot(role_over_time.index, role_over_time[role], 
                 marker='o', linewidth=2, label=role)
    
    plt.title('Community Core Roles Over Time')
    plt.xlabel('Year')
    plt.ylabel('Number of Communities')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Role')
    plt.tight_layout()
    plt.savefig("results/core_roles_over_time.png", dpi=300)
    plt.close()
    print("Saved core roles over time plot")
    
    # Remove temporary columns
    temporal_df.drop(columns=core_roles, inplace=True)
    
    return role_over_time

# 3. Growth pattern analysis
def analyze_growth_patterns(temporal_df):
    """Analyze community growth patterns"""
    # Group by community and calculate growth features
    growth_features = temporal_df.groupby('community').agg({
        'node_count': ['mean', 'std', 'max'],
        'growth_rate': ['mean', 'std']
    })
    
    # Rename columns
    growth_features.columns = [
        'mean_size', 'size_std', 'max_size',
        'mean_growth', 'growth_std'
    ]
    growth_features = growth_features.reset_index()
    
    # Classify growth patterns
    def classify_growth(row):
        if row['mean_size'] < 20:
            return 'Small'
        if row['mean_growth'] > 0.3:
            return 'Fast Growing'
        if row['mean_growth'] < -0.2:
            return 'Declining'
        if row['growth_std'] > 0.4:
            return 'Volatile'
        return 'Stable'
    
    growth_features['growth_pattern'] = growth_features.apply(classify_growth, axis=1)
    
    # Calculate the number of communities for each growth pattern and sort in descending order
    pattern_counts = growth_features['growth_pattern'].value_counts().sort_values(ascending=False)

    # Create a horizontal bar plot using Seaborn for better aesthetics
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        x=pattern_counts.values, 
        y=pattern_counts.index, 
        hue=pattern_counts.index,  # Assign y-axis variable to hue
        palette='viridis', 
        orient='h',
        legend=False  # Turn off automatic legend from hue
    )

    plt.title('Distribution of Community Growth Patterns', fontsize=16)
    plt.xlabel('Number of Communities', fontsize=12)
    plt.ylabel('Growth Pattern', fontsize=12)

    # Add count and percentage labels to each bar
    total_communities = len(growth_features)
    for i, count in enumerate(pattern_counts.values):
        label_text = f' {count} ({count / total_communities:.1%})'
        ax.text(count, i, label_text, color='black', va='center', ha='left')

    plt.xlim(0, pattern_counts.max() * 1.2)  # Set x-axis limit with padding
    plt.grid(axis='x', linestyle='--', alpha=0.6)  # Add grid lines on x-axis
    plt.tight_layout()
    plt.savefig("results/growth_patterns.png", dpi=300)
    plt.close()
    print("Saved growth patterns plot (as a bar chart)")

    return growth_features

# 4. Stagnating community detection
def plot_emergence_disappearance_trends(temporal_df):
    """
    Analyze and visualize trends in community emergence and disappearance
    """
    # Prepare the data: sort by community and time window
    df = temporal_df.sort_values(['community', 'window_center']).copy()
    
    # Create a column for previous status to detect transitions
    df['prev_status'] = df.groupby('community')['status'].shift(1)
    df.dropna(subset=['prev_status'], inplace=True)  # Remove rows without previous status
    
    # Identify emerging and disappearing communities
    is_emerging = (df['prev_status'] != 'Active') & (df['status'] == 'Active')
    is_disappearing = (df['prev_status'] == 'Active') & (df['status'] != 'Active')
    
    # Create count columns for aggregation
    df['emerging_count'] = is_emerging.astype(int)
    df['disappearing_count'] = is_disappearing.astype(int)
    
    # Aggregate counts by time window
    trends = df.groupby('window_center')[['emerging_count', 'disappearing_count']].sum()
    
    # Create the visualization
    plt.figure(figsize=(14, 7))
    plt.plot(trends.index, trends['emerging_count'], 'o-', label='Emerging Communities', color='green')
    plt.plot(trends.index, trends['disappearing_count'], 's--', label='Disappearing Communities', color='red')
    
    plt.title('Macro Trends: Emergence and Disappearance of Communities Over Time', fontsize=16)
    plt.xlabel('Year')
    plt.ylabel('Number of Communities')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig("results/community_macro_trends.png", dpi=300)
    plt.close()
    print("Saved community macro trends plot")

    
# 5. Knowledge flow analysis
def analyze_knowledge_flow(temporal_df):
    """Analyze knowledge flow patterns for ACTIVE communities."""
    
    # [New] Filter to include only data points from active periods
    active_flow_df = temporal_df[temporal_df['status'] == 'Active'].copy()

    # If no active communities exist, skip visualization
    if active_flow_df.empty:
        print("No active communities to analyze for knowledge flow clustermap. Skipping.")
        return pd.DataFrame()  # Return an empty DataFrame

    # [Modified] Create knowledge flow matrix using filtered data
    flow_matrix = active_flow_df.pivot_table(
        index='community',
        columns='window_center',
        values='outflow_ratio',
        fill_value=0
    )
    
    # Cluster analysis (subsequent code remains unchanged)
    # Set appropriate height to prevent label overlap
    figsize_height = max(10, len(flow_matrix.index) * 0.4) 
    sns.clustermap(
        flow_matrix, 
        cmap='coolwarm',
        figsize=(12, figsize_height),
        standard_scale=1  # Standardize by row
    )
    plt.title('Knowledge Flow Patterns Across ACTIVE Communities')  # Updated title
    plt.tight_layout()
    plt.savefig("results/knowledge_flow_clustermap.png", dpi=300)
    plt.close()
    print("Saved knowledge flow clustermap for active communities")
    
    return flow_matrix


def visualize_cross_community_flow(flow_matrix, comm_ids, comm_topics, filename):
    plt.figure(figsize=(16, 14))  # Increase figure size
    
    # More robust label generation
    comm_labels = []
    for cid in comm_ids:
        # Get community keywords (safe handling)
        keywords = comm_topics.get(cid, {}).get('community_keywords', [])
        label = f"C{cid}"
        if keywords:
            label += f"\n{keywords[0]}"
        comm_labels.append(label)
    
    # Create DataFrame for filtering
    flow_df = pd.DataFrame(flow_matrix, index=comm_labels, columns=comm_labels)
    
    # Filter low-flow communities (optional)
    min_flow = flow_df.sum(axis=1).quantile(0.7)  # Keep only top 30% active communities
    active_comms = flow_df.sum(axis=1) > min_flow
    flow_df = flow_df.loc[active_comms, active_comms]
    
    # Apply log transformation for better value distribution
    log_flow = np.log1p(flow_df.values)  # log(1+x) to avoid log(0)
    
    # Create heatmap
    sns.heatmap(
        log_flow,
        annot=True, 
        fmt=".1f",  # Display one decimal place
        cmap="viridis",
        xticklabels=flow_df.index,
        yticklabels=flow_df.columns,
        linewidths=0.5,
        cbar_kws={'label': 'Log(1 + Citation Count)'}
    )
    
    plt.title("Cross-Community Knowledge Flow")
    plt.xlabel("Target Community")
    plt.ylabel("Source Community")
    plt.xticks(rotation=90, fontsize=9)  # Rotate x-axis labels
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved cross-community flow visualization: {filename}")


def plot_community_state_distribution(temporal_df):
    """
    Visualize the distribution of Active, Latent, and Dormant communities over time using a stacked area chart.
    """
    # 1. Count communities per status in each time window
    state_counts = temporal_df.groupby(['window_center', 'status']).size().unstack(fill_value=0)
    
    # 2. Ensure column order for better readability
    state_counts = state_counts[['Dormant', 'Latent', 'Active']]
    
    # 3. Create stacked area plot
    plt.figure(figsize=(14, 8))
    plt.stackplot(
        state_counts.index, 
        state_counts['Dormant'], 
        state_counts['Latent'], 
        state_counts['Active'], 
        labels=['Dormant', 'Latent', 'Active'],
        colors=['#B0C4DE', '#FFD700', '#2E8B57']  # LightSteelBlue, Gold, SeaGreen
    )
    
    plt.title('Distribution of Community States Over Time', fontsize=16)
    plt.xlabel('Year')
    plt.ylabel('Number of Communities')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(state_counts.index.min(), state_counts.index.max())
    plt.tight_layout()
    plt.savefig("results/community_state_distribution.png", dpi=300)
    plt.close()
    print("Saved community state distribution plot (stacked area chart).")


def plot_community_lifecycle_heatmap(temporal_df):
    """
    Visualize the lifecycle of each community using a heatmap to track state changes over time.
    """
    # 1. Map categorical states to numerical values for visualization
    status_map = {'Dormant': 0, 'Latent': 1, 'Active': 2}
    df = temporal_df.copy()
    df['status_numeric'] = df['status'].map(status_map)
    
    # 2. Create pivot table: communities as rows, time windows as columns
    lifecycle_matrix = df.pivot_table(
        index='community', 
        columns='window_center', 
        values='status_numeric'
    )
    
    # 3. [Key] Sort communities to create a structured visualization
    # Calculate each community's highest status for primary sorting
    lifecycle_matrix['max_status'] = lifecycle_matrix.max(axis=1)
    # Calculate active duration as secondary sort criterion
    lifecycle_matrix['active_duration'] = (lifecycle_matrix > 0).sum(axis=1)
    # Sort by highest status then active duration (both descending)
    lifecycle_matrix.sort_values(
        by=['max_status', 'active_duration'], 
        ascending=[False, False], 
        inplace=True
    )
    # Remove temporary sorting columns
    lifecycle_matrix.drop(columns=['max_status', 'active_duration'], inplace=True)

    # 4. Create heatmap visualization
    # Dynamically adjust height to prevent Y-axis label overlap
    figsize_height = max(10, len(lifecycle_matrix.index) * 0.1)
    plt.figure(figsize=(20, figsize_height))
    
    ax = sns.heatmap(
        lifecycle_matrix,
        cmap=['#B0C4DE', '#FFD700', '#2E8B57'],  # Consistent colors with state distribution
        linewidths=.5,
        cbar_kws={'ticks': [0, 1, 2], 'shrink': 0.3}  # Set colorbar ticks
    )
    
    # Customize colorbar labels
    cbar = ax.collections[0].colorbar
    cbar.set_ticklabels(['Dormant', 'Latent', 'Active'])

    plt.title('Lifecycle of Each Community Over Time', fontsize=16)
    plt.xlabel('Year')
    plt.ylabel('Community ID (Sorted by Activity Level)')
    plt.yticks(rotation=0, fontsize=8)  # Display Y-axis labels horizontally
    
    plt.tight_layout()
    plt.savefig("results/community_lifecycle_heatmap.png", dpi=300)
    plt.close()
    print("Saved community lifecycle heatmap.")

# Analyze knowledge absorption in emerging fields
if 'emerging_comms' not in locals():
    emerging_comms = []
    
for comm_id in temporal_df['community'].unique():
    comm_data = temporal_df[temporal_df['community'] == comm_id]
    
    # Emerging field criteria: late-stage fast growth + high inflow
    if len(comm_data) > 2:
        late_growth = comm_data['growth_rate'].iloc[-1] > 0.4
        high_inflow = comm_data['inflow_ratio'].mean() > 0.35
        
        if late_growth and high_inflow:
            emerging_comms.append(comm_id)

df_sorted = temporal_df.sort_values(['community', 'window_center']).copy()
df_sorted['prev_status'] = df_sorted.groupby('community')['status'].shift(1)
df_sorted.dropna(subset=['prev_status'], inplace=True)

emerging_events = df_sorted[
    (df_sorted['prev_status'] != 'Active') & (df_sorted['status'] == 'Active')
]
emerging_comm_ids = emerging_events['community'].unique()


# Analyze knowledge sources for emerging communities
emerging_analysis = []
if len(emerging_comm_ids) > 0:
    print(f"\nAnalyzing knowledge sources for {len(emerging_comm_ids)} emerging communities...")
    # Get the full cross-community knowledge flow matrix
    flow_matrix, comm_ids = build_cross_community_flow(G, comms)
    # Create reverse lookup dictionary for community IDs
    comm_id_to_index = {cid: i for i, cid in enumerate(comm_ids)}

    for comm_id in emerging_comm_ids:
        if comm_id not in comm_id_to_index: 
            continue  # Skip if community not in flow matrix
            
        comm_idx = comm_id_to_index[comm_id]
        # Get all incoming citations to this community from the matrix
        inflows = flow_matrix[:, comm_idx]

        # Identify main knowledge sources
        top_sources = []
        for src_idx, count in enumerate(inflows):
            if count > 0:
                source_comm = comm_ids[src_idx]
                # Get top 2 keywords as source name
                source_name = '|'.join(comm_topics.get(source_comm, {}).get('community_keywords', [])[:2])
                top_sources.append((source_name, count))
        
        # Sort sources by citation count descending
        top_sources.sort(key=lambda x: x[1], reverse=True)
        
        emerging_analysis.append({
            'emerging_community': comm_id,
            'emerging_name': '|'.join(comm_topics.get(comm_id, {}).get('community_keywords', [])[:2]),
            'main_sources': '; '.join([f"{name}({count})" for name, count in top_sources[:3]]),  # Top 3 sources
            'total_inflow': sum(inflows)  # Total incoming citations
        })

# Save analysis results
emerging_df = pd.DataFrame(emerging_analysis)
if not emerging_df.empty:
    emerging_df.to_csv("results/emerging_communities.csv", index=False)
    print(f"\nIdentified and analyzed {len(emerging_df)} emerging communities.")
else:
    print("\nNo emerging communities identified based on the new criteria.")


# Save bridge community results
bridge_df.to_csv("results/bridge_communities.csv", index=False)

# Visualize overall knowledge flow
visualize_cross_community_flow(flow_matrix, comm_ids, comm_topics, "results/final_knowledge_flow.png")

# Main execution flow
if 'temporal_df' in locals() and not temporal_df.empty:
    print("\nStarting analysis and visualization...")
    
    # Analyze community role distribution
    role_over_time = analyze_community_roles(temporal_df)
    
    # Analyze growth patterns
    growth_features = analyze_growth_patterns(temporal_df)
    
    # Detect stagnating communities
    plot_emergence_disappearance_trends(temporal_df)

    plot_community_state_distribution(temporal_df)
    plot_community_lifecycle_heatmap(temporal_df)
    
    # Analyze knowledge flow
    flow_matrix = analyze_knowledge_flow(temporal_df)
    
    # Plot evolution for each major community
    top_communities = temporal_df.groupby('community')['node_count'].max().nlargest(5).index
    for comm_id in top_communities:
        comm_data = temporal_df[temporal_df['community'] == comm_id]
        plot_community_evolution(comm_id, comm_data)
        plot_community_topic_evolution(comm_id, comm_data, metadata_df)

    # Generate analysis report
    print("\n" + "="*60)
    print("Community Evolution Analysis Report")
    print("="*60)
    print(f"Total communities analyzed: {temporal_df['community'].nunique()}")
    print(f"Time windows covered: {temporal_df['window_center'].nunique()}")
    
    print("\nGrowth pattern distribution:")
    print(growth_features['growth_pattern'].value_counts())
    
    print("\n" + "="*60)
    print("All visualizations saved to 'results' directory")
    print("="*60)
    
    # Add cross-domain analysis to final report
    print("\n=== Cross-Domain Analysis Results ===")
    print(f"Top Bridge Community: C{bridge_df.iloc[0]['community']} ({bridge_df.iloc[0]['community_name']})")
    print(f"  - Hub Score: {bridge_df.iloc[0]['hub_score']:.1f}")
    print(f"  - Mediation Count: {bridge_df.iloc[0]['mediation']}")

    if not emerging_df.empty:
        print("\nEmerging Communities and Their Knowledge Sources:")
        for _, row in emerging_df.iterrows():
            print(f"- {row['emerging_name']}: Sources: {row['main_sources']}")
    else:
        print("\nNo emerging communities identified")

    # Save analysis results
    temporal_df.to_csv("results/community_evolution.csv", index=False)
    growth_features.to_csv("results/growth_patterns.csv", index=False)
    
    print("Saved analysis results to CSV files")
else:
    print("Error: temporal_df not found or empty. Analysis aborted.")

print("\nProject analysis completed successfully!")