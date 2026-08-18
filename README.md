# Scientific Community Evolution Analysis

A modular Python analysis pipeline for detecting, tracking, and interpreting scientific communities in a citation network.

**English** | [简体中文](README.zh-CN.md)

This project originated as my MSc thesis at the University of Galway. I built the full analysis workflow from raw citation and paper metadata through network construction, community detection, temporal analysis, topic modeling, classification, and visualization.

## What I Built

The project turns raw citation data into an end-to-end community evolution analysis:

```text
Citation data + paper metadata
            ↓
      Data cleaning
            ↓
   Citation network build
            ↓
   Leiden community detection
            ↓
     TF-IDF + LDA topics
            ↓
 COIN-style citation-flow metrics
            ↓
   2-year sliding windows
            ↓
Community lifecycle / growth analysis
            ↓
Knowledge-flow & bridge analysis
            ↓
      CSV + visual reports
```

The pipeline analyzes the arXiv High-Energy Physics Theory (Hep-Th) citation network from 1992–2003 and studies how research communities grow, decline, become isolated, exchange knowledge, and change topics over time.

## Skills Demonstrated

This repository is intended to show practical project capability, not only a research result.

- **Python data processing** — loading, cleaning, aligning, and transforming citation and metadata records
- **ETL-style workflow design** — structured flow from raw data to analysis-ready datasets and generated outputs
- **Graph & network analysis** — directed citation networks, Leiden community detection, inter-community citation flow
- **NLP / topic modeling** — TF-IDF feature extraction and LDA-based topic interpretation
- **Temporal analytics** — sliding-window metrics, growth rates, lifecycle tracking, and long-term trend classification
- **Data visualization** — network diagrams, heatmaps, growth charts, lifecycle views, and temporal evolution plots
- **Software organization** — modular source files, reusable functions, explicit pipeline entry point, dependency management, and error handling

## Core Analysis

### Community Detection

The directed citation network is projected for Leiden community detection. The pipeline calculates community sizes and identifies the largest research groups for further analysis.

### Topic Modeling

For major communities, titles and abstracts are processed with TF-IDF and LDA to extract representative topic keywords and provide semantic labels for otherwise structural graph clusters.

### Temporal Evolution

Communities are tracked across **2-year sliding windows** using COIN-style citation-flow metrics:

- **Introspection** — citations remaining within the same community
- **Inflow** — citations received from other communities
- **Outflow** — citations sent to other communities
- **Influence score** — combines active community size with citation inflow

Each community is also classified by state and trajectory, including Active, Latent, Dormant, Growing, Declining, Hub, Exporter, Insular, Stagnating, and Opening patterns.

### Cross-Community Knowledge Flow

The pipeline builds an inter-community flow matrix to identify:

- bridge communities
- knowledge hubs
- emerging communities
- major knowledge sources for newly active communities
- changes in cross-community flow over time

## Example Results

### Community Evolution

![Community evolution](results/community_0_evolution.png)

Tracks citation-flow ratios and community size across time windows, including changes in community classification.

### Topic Evolution

![Topic evolution](results/community_0_topic_evolution.png)

Shows how the most important research keywords within one community change over time.

### Community Lifecycle

![Community lifecycle](results/community_lifecycle_heatmap.png)

Shows transitions between Dormant, Latent, and Active states across detected communities.

## Project Structure

```text
.
├── main.py                     # end-to-end pipeline entry point
├── requirements.txt            # Python dependencies
├── data.py                     # original thesis implementation retained for reference
├── src/
│   ├── data_loading.py         # citation / metadata loading and cleaning
│   ├── community_analysis.py   # Leiden, topic modeling, cross-community flow
│   ├── temporal_analysis.py    # COIN metrics, sliding windows, classifications
│   ├── extended_analysis.py    # lifecycle, roles, emerging and bridge analyses
│   └── visualization.py        # reusable plotting functions
└── results/                    # generated figures and CSV analysis outputs
```

The original thesis script is retained as a reference implementation, while the modular version separates data, analysis, temporal logic, and visualization into reusable components.

## Generated Outputs

The pipeline produces both machine-readable analysis files and presentation-ready visualizations, including:

```text
network_statistics.png
paper_temporal_distribution.png
community_size_distribution.png
community_network.png
community_<id>_evolution.png
community_<id>_topic_evolution.png
core_roles_over_time.png
growth_patterns.png
community_macro_trends.png
knowledge_flow_clustermap.png
community_state_distribution.png
community_lifecycle_heatmap.png
final_knowledge_flow.png

community_evolution.csv
growth_patterns.csv
bridge_communities.csv
emerging_communities.csv   # generated when emerging communities are detected
```

## Tech Stack

**Python · Pandas · NumPy · NetworkX · igraph · leidenalg · scikit-learn · Matplotlib · Seaborn**

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the Hep-Th citation file and abstract metadata under:

```text
data/
├── cit-HepTh.txt
└── cit-HepTh-abstracts/
```

Then run:

```bash
python main.py
```

Generated analysis files are written to `results/`.

## Project Context

This was developed as an MSc research project, but the implementation demonstrates a broader workflow that is transferable to other data and software tasks: ingesting raw data, building a processing pipeline, applying analytical methods, generating reusable outputs, and organizing the work into maintainable modules.
