import pandas as pd
import os

# Loading reference relationships
cit_df = pd.read_csv(
    "data/cit-HepTh.txt",
    sep="\t",
    comment='#',
    header=None,
    names=["source", "target"],
    dtype={"source": str, "target": str}
)

# Conversion to 7-digit format
cit_df["source"] = cit_df["source"].str.strip().str.zfill(7)
cit_df["target"] = cit_df["target"].str.strip().str.zfill(7)

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
                    paper_id = filename.split(".")[0].zfill(7)
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
                    abstract_started = False
                    abstract_lines = []
                    for line in lines:
                        if line.strip() == "\\\\":
                            abstract_started = True
                            continue
                        if abstract_started:
                            if line.strip() == "\\\\":  # 摘要结束标志
                                break
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



import matplotlib.pyplot as plt
import networkx as nx

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
plt.show()


# Plotting time distribution histograms
plt.figure(figsize=(10, 5))
plt.hist(metadata_df['year'], bins=range(1992, 2004), edgecolor='black', align='left')
plt.xticks(range(1992, 2004))
plt.xlabel('Year')
plt.ylabel('Number of Papers')
plt.title('Temporal Distribution of Papers')
plt.grid(axis='y', alpha=0.75)
plt.show()



# =============================================================================
# Community Detection
# =============================================================================
import community as community_louvain

# Convert to undirected graph for community detection
G_undir = G.to_undirected()
G_undir.remove_edges_from(nx.selfloop_edges(G_undir))

comms = community_louvain.best_partition(G_undir, resolution=1.0)

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


# =============================================================================
# Thematic analysis
# =============================================================================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import defaultdict, Counter

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

for i, (comm_id, paper_ids) in enumerate(major_communities[:15]):  # Analyze top 15 largest communities
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
    # Filter edges within time window
    edges_in_window = []
    for source, target in citation_graph.edges():
        source_year = paper_to_year.get(source)
        if source_year and year_start <= source_year <= year_end:
            edges_in_window.append((source, target))
    
    print(f"\nProcessing {len(edges_in_window)} edges in window {year_start}-{year_end}")

    # Initialize COIN metrics storage
    community_metrics = defaultdict(lambda: {
        'introspection': 0,
        'inflow': 0,
        'outflow': 0,
        'total_papers': 0,
        'total_citations': 0
    })

    # Count papers per community in window
    for paper_id, year in paper_to_year.items():
        if year_start <= year <= year_end and paper_id in paper_to_community:
            comm_id = paper_to_community[paper_id]
            community_metrics[comm_id]['total_papers'] += 1

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
            
        # 计算影响力指数
        metrics['influence_score'] = metrics['total_papers'] + metrics['inflow'] * 0.5
        
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

# Perform temporal window analysis
print("\nPerforming temporal analysis with 2-year sliding windows...")
window_size = 2
temporal_data = []
min_year = int(min(years))
max_year = int(max(years))

for start_year in range(min_year, max_year - window_size + 2):
    end_year = start_year + window_size - 1
    if end_year > max_year:
        break
        
    # Get COIN metrics for this time window
    coin_metrics = calculate_coin_metrics(start_year, end_year, G, paper_to_year, paper_to_community)
    
    # Only analyze communities with sufficient activity
    active_communities = 0
    for comm_id, metrics in coin_metrics.items():
        if metrics['total_papers'] >= 5:
            active_communities += 1
            temporal_data.append({
                'community': comm_id,
                'year_start': start_year,
                'year_end': end_year,
                'window_center': (start_year + end_year) / 2,
                'total_papers': metrics['total_papers'],
                'total_citations': metrics['total_citations'],
                'introspection': metrics['introspection'],
                'inflow': metrics['inflow'],
                'outflow': metrics['outflow'],
                'introspection_ratio': metrics['introspection_ratio'],
                'inflow_ratio': metrics['inflow_ratio'],
                'outflow_ratio': metrics['outflow_ratio'],
                'influence_score': metrics['influence_score']
            })
    
    print(f"  Window {start_year}-{end_year}: {active_communities} active communities")

temporal_df = pd.DataFrame(temporal_data)
print(f"Generated temporal analysis with {len(temporal_df)} data points")
print(f"Unique communities tracked: {temporal_df['community'].nunique()}")