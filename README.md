# 📘 Real-World Forecasting Foundations  
### *The Official Repository of Forecast Academy*

Welcome to the production repository for **Forecast Academy – Real-World Forecasting Foundations**.

This repo contains all notebooks, utilities, and curated artifacts needed to build a **production-grade forecasting system** from scratch.  
Unlike tutorials that focus only on model syntax, this course covers the **entire forecasting lifecycle** — from business framing to diagnostics, visualization, modeling strategy, and evaluation.

---

## 🎯 What You Will Build

By the end of this course, you will have constructed a complete forecasting pipeline for the **M5 (Walmart) Dataset**, capable of scaling to **30,000+ SKUs**.

You will learn how to:

- Diagnose portfolio structure using the **Lie Detector Six** framework  
- Engineer clean timelines that avoid leakage and handle messy real-world calendars  
- Visualize thousands of SKUs instantly using **tsforge**  
- Use GenAI as a strategic partner with the **SPICE Framework**  
- Create a clear **Strategy on a Page** that aligns Data Science with Business  
- Build a reproducible forecasting workflow you can use at work immediately  

This repository mirrors a **real-world DS production environment**, giving you experience that translates directly into industry practice.

---

## 📂 Repository Structure

```text
real-world-forecasting-foundations/
├── notebooks/                  # The Classroom: Follow modules here
│   ├── module_01/              # Strategy, Diagnostics, Data Prep
│   └── module_02/              # Baselines & Evaluation (Coming Soon)
│
├── data/                       # Local data store (mostly git-ignored)
│   ├── raw/                    # Place the Kaggle M5 CSVs here
│   └── artifacts/              # ⭐ Precomputed “Save Points”
│
├── scripts/                    # Setup & admin tools
│   └── download_data.py        # ⭐ RUN THIS FIRST (downloads artifacts)
│
├── utils/                      # Course glue (paths, styling)
│   └── paths.py                # Cross-platform path helper
│
├── docs/                       # M5 dataset overview, schema, reference
│   
│
└── environment.yml             # Conda environment for the course
```

## 🔑 Why this structure?

Mirrors how professional teams organize forecasting projects

Keeps data cleanly separated and safely git-ignored

Makes the repo scalable and beginner-friendly

Allows “save points” so learners can jump into any module

## 🚀 Quick Start Guide

Follow these steps to set up your environment exactly as used in the course.

**1. Clone the Repository**
```git clone https://github.com/YourUsername/real-world-forecasting-foundations.git
cd real-world-forecasting-foundations
```
**2. Create the Conda Environment**

We use conda to avoid dependency conflicts and ensure reproducibility.
```
conda env create -f environment.yml
conda activate forecast-academy
```
**3. Download Course Artifacts (Save Points)**

We don’t store large datasets in GitHub.
Instead, run:
```
python scripts/download_data.py
```

This will download curated artifacts for all the modules in case you plan to jump around.


You're ready to begin.




## 🛠 Troubleshooting & FAQ
**Q: I get a FileNotFoundError when loading data.**

A: Make sure you ran:
```
python scripts/download_data.py
```

Also ensure you're running notebooks from inside the notebooks/ directory.
The utils.paths helper resolves paths based on the repo root.

**Q: Plotly charts aren’t showing in Jupyter.**

A: You may need to install widget extensions:
```
jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget
```
**Q: Can I use this pipeline for my own company’s data?**

A: Yes. The architecture is dataset-agnostic.
Just ensure your private data uses:
* unique_id
* ds (date)
* y (target)

This aligns with Nixtla + tsforge conventions.

## **© 2025 Forecast Academy**

All Rights Reserved.
This repository is part of the official Forecast Academy curriculum.