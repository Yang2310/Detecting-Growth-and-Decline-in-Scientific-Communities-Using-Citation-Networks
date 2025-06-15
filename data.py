import pandas as pd
import os

# Loading reference relationships
cit_df = pd.read_csv(
    "data/cit-HepTh.txt",
    sep="\t",
    header=None,
    names=["source", "target"]
)

# Loading submission time
dates_df = pd.read_csv(
    "data/cit-HepTh-dates.txt",
    sep="\t",
    header=None,
    names=["paper_id", "date"]
)
dates_df["year"] = pd.to_datetime(dates_df["date"]).dt.year

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
                    paper_id = filename.split(".")[0]
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
print(f"Successfully loaded {len(metadata_df)} records")
print("\nFirst few record titles:")
for i in range(min(3, len(metadata_df))):
    print(f"{i+1}. {metadata_df.iloc[i]['title']}")



import matplotlib.pyplot as plt
import networkx as nx

# Build the base citation network
G = nx.from_pandas_edgelist(cit_df, 'source', 'target', create_using=nx.DiGraph())

# Plot basic network statistics
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


# Consolidation of all time data
combined_years = pd.concat([
    dates_df["year"],
    metadata_df["year"]
]).dropna()

# Plotting time distribution histograms
plt.figure(figsize=(10, 5))
plt.hist(combined_years, bins=range(1992, 2004), edgecolor='black', align='left')
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

    