# Scientific Community Evolution Analysis

MSc thesis project at the University of Galway analyzing how scientific communities grow, remain stable, and decline over time using the arXiv High-Energy Physics Theory (Hep-Th) citation network.

## Overview

This project combines citation-network analysis, dynamic community tracking, and topic modeling to study structural and semantic changes in scientific communities.

- **Dataset:** arXiv HEP-Th citation network (1992–2003)
- **Community detection:** Leiden algorithm
- **Dynamic analysis:** COIN model with sliding time windows
- **Topic modeling:** TF-IDF + LDA
- **Outputs:** community-size, citation-flow, temporal-evolution, and topic-evolution visualizations

## Tech Stack

Python · Pandas · NumPy · NetworkX · igraph · leidenalg · scikit-learn · Matplotlib · Seaborn

## What the Project Does

1. Loads citation relationships and paper metadata.
2. Cleans and aligns citation records with valid paper IDs.
3. Builds the citation network and summarizes its structure.
4. Detects scientific communities with the Leiden algorithm.
5. Tracks community evolution across sliding time windows.
6. Extracts representative topics with TF-IDF and LDA.
7. Generates visualizations for structural and semantic evolution.

## Example Results

### Community Evolution

![Community evolution](results/community_0_evolution.png)

Tracks the structural evolution of a detected scientific community over time.

### Topic Evolution

![Topic evolution](results/community_0_topic_evolution.png)

Shows how dominant research topics shift within the same community across time windows.

## Repository Structure

The original thesis implementation is currently contained in `data.py`. A refactor is in progress to split the pipeline into reusable modules for data loading, community detection, temporal analysis, topic modeling, and visualization.

## Notes

This repository is intended as a research and portfolio project. The analysis framework can be adapted to other temporal citation or interaction networks with similar metadata.
