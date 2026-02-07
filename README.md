# TDA Research Tool

This repository implements a research-focused Topological Data Analysis (TDA) framework for mixed 2D and 3D point clouds.

The system supports persistent homology up to dimension H₂, advanced diagram representations, multiple diagram distances, and topology-aware feature constructions suitable for mathematical analysis and machine learning.

## Features

- Persistent homology (H₀, H₁, H₂) via Vietoris–Rips complexes
- Euclidean and geodesic (kNN-based) filtrations
- Persistence diagrams and barcodes
- Persistence images (multi-scale)
- Persistence entropy and total persistence
- Persistence landscapes and silhouette functions
- Wasserstein, bottleneck, and sliced Wasserstein distances
- Density-based filtrations
- Persistent cohomology and circular coordinates
- Mapper construction with spectral graph invariants
- Mixed 2D / 3D synthetic datasets
- Ensemble-based classification using topological features

## Project Structure

TDA-research-tool/
requirements.txt
app.py
topo.py

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
Deployment

The application is designed to run directly on Streamlit Community Cloud by connecting this repository and selecting app.py as the entry point.

Notes

This project is intended as a computational and exploratory research tool.
All topology-related computations are performed explicitly and transparently, without black-box abstractions.
