# HealthcareAI_MultiTaskProjects

This repository contains three end-to-end AI/ML projects with fully runnable Colab notebooks (GPU ready):

- Task 1: Arrhythmia classification from ECG using 1D CNN
- Task 2: Fine-tuning Bio_ClinicalBERT for clinical sentence classification (22 labels)
- Task 3: Abstractive summarization fine-tuning with LLaMA 3.1 (or fallback)

Open the notebooks directly in Google Colab for a GPU-accelerated experience. Each notebook installs its dependencies, downloads data, trains, evaluates, and plots results with extensive markdown explanations.

## Repository Structure

- Task1_ECG_CNN.ipynb
- Task2_BioClinicalBERT.ipynb
- Task3_LLaMA_Summarization.ipynb
- requirements.txt
- dataset_links.md

## Getting Started

1. Open a notebook in Colab and select GPU in Runtime > Change runtime type.
2. Run the first setup cell to install all dependencies.
3. Follow the notebook prompts to download datasets (Google Drive, Kaggle) as needed.

## Notes

- Pushing to GitHub requires a token; see the instructions at the end of each notebook.
