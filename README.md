**# TESI_NLP_project**

This repository contains a pipeline for running Latent Dirichlet Allocation (LDA) topic modeling on TESI (Traumatic Events Screening Inventory) interview transcripts.

The workflow uses DLATK as the main toolbox for feature extraction and topic modeling, with Mallet as the LDA backend.

**Contents**
	•	scripts/lda_pipeline.py – End-to-end script for:
	
	1.	Building an SQLite database from raw TESI transcripts
	
	2.	Extracting 1-gram features with DLATK
	
	3.	Estimating LDA topics with Mallet
	
	4.	Exporting:
	
	•	topic–word distributions
	
	•	document–topic distributions (with and without participant IDs)
	
	•	top keywords per topic
	
	5.	Generating wordcloud visualizations of the learned topics
	
	•	outputs/wordclouds_py/ – example wordcloud for the 30-topic solution

**Requirements**
	•	Python 3.11
	
	•	DLATK
	
	•	Mallet (installed at ~/.local/mallet/bin/mallet)
	
	•	pandas, matplotlib, wordcloud
