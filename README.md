# Scientific Community Evolution Analysis

MSc thesis project at the University of Galway analyzing how scientific communities grow, remain stable, and decline over time using the arXiv High-Energy Physics Theory (Hep-Th) citation network.

## Overview

This project combines citation-network analysis, dynamic community tracking, topic modeling, and knowledge-flow analysis to study structural and semantic changes in scientific communities.

- **Dataset:** arXiv HEP-Th citation network (1992–2003)
- **Community detection:** Leiden algorithm
- **Dynamic analysis:** COIN-style metrics with 2-year sliding windows
- **Topic modeling:** TF-IDF + LDA
- **Knowledge-flow analysis:** cross-community citation flow, bridge communities, emerging-community sources
- **Outputs:** CSV analysis results plus network, lifecycle, growth, topic, and temporal visualizations

## Tech Stack

Python · Pandas · NumPy · NetworkX · igraph · leidenalg · scikit-learn · Matplotlib · Seaborn

## Analysis Pipeline

1. Load citation relationships and arXiv paper metadata.
2. Clean and align citation records with valid paper IDs.
3. Build the directed citation network and summarize its structure.
4. Detect scientific communities with Leiden.
5. Extract representative community topics with TF-IDF + LDA.
6. Calculate COIN-style introspection, inflow, and outflow metrics.
7. Track community states and growth across sliding time windows.
8. Classify community roles and long-term trajectories.
9. Analyze cross-community knowledge flow, bridge communities, and emerging communities.
10. Generate CSV outputs and visualizations for structural and semantic evolution.

## Example Results

### Community Evolution

![Community evolution](results/community_0_evolution.png)

Tracks community size and citation-flow ratios across time windows, including role/classification changes.

### Topic Evolution

![Topic evolution](results/community_0_topic_evolution.png)

Shows how dominant research keywords shift within the same community over time.

### Community Lifecycle

![Community lifecycle](results/community_lifecycle_heatmap.png)

Tracks transitions between Dormant, Latent, and Active states for detected communities.

## Repository Structure

```text
.
├── main.py
├── requirements.txt
├── data.py                     # original monolithic thesis implementation; retained during parity validation
├── src/
│   ├── data_loading.py         # citation and metadata loading/cleaning
│   ├── community_analysis.py   # Leiden, topic modeling, cross-community flow
│   ├── temporal_analysis.py    # COIN metrics, sliding windows, classifications
│   ├── extended_analysis.py    # lifecycle, role, emerging/bridge and flow analyses
│   └── visualization.py        # reusable plotting functions
└── results/                    # generated figures and CSV outputs
```

## Generated Outputs

The modular pipeline preserves the original thesis analysis targets, including:

- `network_statistics.png`
- `paper_temporal_distribution.png`
- `community_size_distribution.png`
- `community_network.png`
- `community_<id>_evolution.png`
- `community_<id>_topic_evolution.png`
- `core_roles_over_time.png`
- `growth_patterns.png`
- `community_macro_trends.png`
- `knowledge_flow_clustermap.png`
- `community_state_distribution.png`
- `community_lifecycle_heatmap.png`
- `final_knowledge_flow.png`
- `community_evolution.csv`
- `growth_patterns.csv`
- `bridge_communities.csv`
- `emerging_communities.csv` when emerging communities are detected

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the Hep-Th citation data and abstract metadata under `data/`, then run:

```bash
python main.py
```

Generated analysis files are written to `results/`.

## Refactor Note

The original `data.py` is intentionally retained on the refactor branch while feature parity is checked. The modular version reorganizes the implementation for readability and reuse; it is not intended to remove thesis analyses or outputs.
