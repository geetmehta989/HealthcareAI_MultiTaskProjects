# Dataset Links and Information

## Task 1: Arrhythmia Classification Using CNN
- **Dataset**: Heartbeat Dataset
- **Source**: Google Drive
- **URL**: https://drive.google.com/file/d/1xAs-CjlpuDqUT2EJUVR5cPuqTUdw2uQg/view?usp=sharing
- **Description**: ECG signal data for arrhythmia classification
- **Download Method**: Using `gdown` library in the notebook
- **Expected Format**: CSV or similar structured format with ECG signals

## Task 2: Fine-Tune Bio_ClinicalBERT
- **Dataset**: Clinical Note Sentences (22 categories)
- **Source**: JSON format
- **Description**: Clinical text data for sentence classification into 22 medical categories
- **Model**: `emilyalsentzer/Bio_ClinicalBERT` from Hugging Face
- **Expected Format**: JSON with text and label fields

## Task 3: LLaMA 3.1 Text Summarization
- **Dataset**: CNN/DailyMail Summarization Dataset
- **Source**: Kaggle
- **URL**: https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail
- **Description**: News articles with human-written summaries for abstractive summarization
- **Download Method**: Kaggle API
- **Expected Format**: CSV with article and summary columns

## Setup Instructions

### For Google Colab:
1. Mount Google Drive for Task 1 dataset access
2. Install Kaggle API and configure credentials for Task 3
3. Ensure GPU runtime is enabled for all tasks

### Authentication Requirements:
- **Google Drive**: Mount drive or use `gdown` with public links
- **Kaggle**: API token (kaggle.json) required for dataset download
- **Hugging Face**: Optional token for model access (if needed)

## Data Preprocessing Notes:
- **Task 1**: ECG signals may require normalization and windowing
- **Task 2**: Text tokenization and label encoding required
- **Task 3**: Article-summary pairs need sequence length management