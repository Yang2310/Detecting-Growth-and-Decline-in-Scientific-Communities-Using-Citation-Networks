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
                        content = f.read().split("\n")
                        try:
                            title = content[0].replace("Title: ", "").strip()
                            abstract = content[2].replace("Abstract: ", "").strip()
                        except IndexError:
                            print(f"格式错误：{file_path}")
                            continue

                    metadata.append({
                        "paper_id": paper_id,
                        "year": year,
                        "title": title,
                        "abstract": abstract
                    })
    
    return pd.DataFrame(metadata)

metadata_df = load_abstracts("data/cit-HepTh-abstracts")



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
