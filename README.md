# MASAR: Drone-to-Satellite Geo-Localization System

## Overview

MASAR is an AI-powered drone geo-localization system designed to determine the geographic location of a drone image by matching it with satellite imagery. The system is based on the TransGeo architecture and utilizes FAISS similarity search for efficient cross-view image retrieval.

The project also integrates Explainable Artificial Intelligence (XAI) techniques to improve transparency by visualizing the image regions that contribute most to localization decisions.

---

## Objectives

* Match drone-view images with corresponding satellite images.
* Improve retrieval efficiency using FAISS indexing.
* Evaluate localization performance using Recall@K metrics.
* Enhance model interpretability using XAI visualizations.
* Assess robustness through stress testing and load testing.

---

## Dataset

The project uses the University-1652 dataset, a benchmark dataset developed for cross-view geo-localization research.

Dataset Characteristics:

* 1,652 university buildings
* 72 universities
* Drone-view imagery
* Satellite-view imagery
* Ground-view imagery

The dataset was selected because it provides a realistic environment for training and evaluating drone-to-satellite localization models.

---

## Project Structure

```text
MASAR/
│
├── main.py
├── setup_dataset.py
├── preprocessing.py
├── data_loader.py
├── train.py
├── retrieval.py
├── evaluation.py
├── xai.py
├── testing.py
└── README.md
```

---

## Module Description

### setup_dataset.py

Handles dataset extraction, environment setup, path configuration, and TransGeo preparation.

### preprocessing.py

Contains image preprocessing, resizing, normalization, and augmentation pipelines.

### data_loader.py

Implements dataset classes and image-loading utilities for drone and satellite imagery.

### train.py

Loads and trains the TransGeo model using contrastive learning.

### retrieval.py

Performs feature extraction and FAISS-based similarity retrieval.

### evaluation.py

Computes Recall@1, Recall@5, Recall@10, inference latency, and evaluation visualizations.

### xai.py

Generates GradCAM saliency maps and feature importance visualizations.

### testing.py

Contains stress testing, load testing, performance monitoring, and deployment strategy simulations.

---

## Technologies Used

* Python
* PyTorch
* Vision Transformers (ViT)
* TransGeo
* FAISS
* NumPy
* SciPy
* Matplotlib
* PIL
* Google Colab

---

## Evaluation Metrics

The system is evaluated using:

* Recall@1
* Recall@5
* Recall@10
* Retrieval Latency
* Inference Time
* Stress Test Performance
* Load Test Performance

---

## Explainable AI Features

The project includes:

* GradCAM Saliency Maps
* Feature Importance Maps
* Visual Explanation Overlays

These techniques help explain why the model retrieves a particular satellite image for a given drone query.

---

## MASAR Development Team
Retaj Alhazmi

Manar Ahmed

Ghadeer Sami

Ghala Shahir

Maria Alzahery

AI System Design Project
