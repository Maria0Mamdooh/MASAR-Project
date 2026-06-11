# MASAR-Project : Drone-to-Satellite Geo-Localization System

## Overview

MASAR is an AI-powered geo-localization system that identifies the geographic location of a drone image by matching it with satellite imagery. The system is based on the TransGeo architecture and uses FAISS similarity search to retrieve the most relevant satellite images.

The project also incorporates Explainable Artificial Intelligence (XAI) techniques to visualize the image regions that influence localization decisions.

## Main Components

* Dataset Setup & Environment Configuration
* Data Preprocessing & Loading
* TransGeo Model Training
* FAISS-Based Image Retrieval
* Performance Evaluation
* Explainable AI (XAI)
* Stress Testing & Load Testing

## Dataset

The project uses the University-1652 dataset, a benchmark dataset for cross-view geo-localization containing drone-view, satellite-view, and ground-view images collected from 72 universities and 1,652 buildings.

## Project Structure

text
MASAR/
│
├── main.py
├── setup_dataset.py
├── preprocessing.py
├── data_loader.py
├── train.py
├── retrieval.py
├── xai.py
├── testing.py
└── README.md


## Technologies Used

* Python
* PyTorch
* Vision Transformers (ViT)
* TransGeo
* FAISS
* NumPy
* Matplotlib
* Google Colab

## Evaluation Metrics

* Recall@1
* Recall@5
* Recall@10
* Retrieval Latency
* Inference Time

## MASAR Development Team
Retaj Alhazmi
Manar Ahmed
Ghadeer Sami
Maria Alzahery
Ghala Shahir

AI System Design Project
