# TESI_NLP_project

This repository contains a pipeline for running **Latent Dirichlet Allocation (LDA)** topic modeling on TESI (Traumatic Events Screening Inventory) interview transcripts from the Early Life Stress, Puberty, and Neural Trajectories (ELS) study in the Stanford Neurodevelopment Affect and Psychopathology Laboratory (PI: Ian Gotlib.

The workflow uses [DLATK](https://dlatk.github.io/dlatk/) as the main toolbox for feature extraction and topic modeling, with Mallet as the LDA backend.

---

## Contents
- `scripts/lda_pipeline.py` – End-to-end script for:
  1. Building an SQLite database from raw TESI transcripts  
  2. Extracting 1-gram features with DLATK  
  3. Estimating LDA topics with Mallet  
  4. Exporting:  
     - topic–word distributions  
     - document–topic distributions (with and without participant IDs)  
     - top keywords per topic  
  5. Generating wordcloud visualizations of the learned topics  
- `outputs/wordclouds_py/` – Example wordclouds for the 30-topic solution
- `clustering.rmd` – Rmd notebook for clustering the 30 topics into higher-order categories (e.g., 5 interpretable clusters: grief, sports/routine, conflict, games/school, mistreatment)
- `scripts/lda_pipeline_pca.py`
  Sentence-level pipeline that:
  1. splits transcripts into sentences,  
  2. builds DLATK features per sentence,  
  3. estimates ~500-topic MALLET LDA at the sentence level,  
  4. aggregates to transcript (participant) level, and  
  5. applies PCA to obtain <=200 orthogonal components for downstream modeling.


---

## Requirements
- Python 3.11  
- DLATK  
- Mallet (installed at `~/.local/mallet/bin/mallet`)  
- `pandas`, `matplotlib`, `wordcloud`
