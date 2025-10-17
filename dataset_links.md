# Dataset Links and Download Instructions

This document contains all the necessary links and instructions for downloading the datasets used in the three healthcare AI projects.

## 📊 Task 1: ECG Arrhythmia Classification

### Dataset: Heartbeat Dataset
- **Source**: Google Drive
- **Direct Link**: https://drive.google.com/file/d/1xAs-CjlpuDqUT2EJUVR5cPuqTUdw2uQg/view?usp=sharing
- **File ID**: `1xAs-CjlpuDqUT2EJUVR5cPuqTUdw2uQg`
- **Format**: CSV/JSON
- **Size**: ~50MB
- **Description**: ECG heartbeat signals for arrhythmia classification

### Download Instructions:
```python
# Method 1: Using gdown (recommended for Colab)
!pip install gdown
import gdown
url = 'https://drive.google.com/uc?id=1xAs-CjlpuDqUT2EJUVR5cPuqTUdw2uQg'
gdown.download(url, 'heartbeat_dataset.zip', quiet=False)

# Method 2: Using Colab's built-in file upload
# Upload the file directly through Colab's file interface
```

## 📊 Task 2: Clinical Note Classification

### Dataset: Clinical Notes JSON
- **Source**: To be provided or generated
- **Format**: JSON
- **Description**: Clinical note sentences with 22 category labels
- **Categories**: Various medical conditions and procedures

### Download Instructions:
```python
# The dataset will be generated or provided as part of the notebook
# If external dataset is needed, it will be specified in the notebook
```

## 📊 Task 3: Text Summarization

### Dataset: CNN/DailyMail Summarization
- **Source**: Kaggle
- **Link**: https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail
- **Format**: CSV
- **Size**: ~1.2GB
- **Description**: News articles with human-written summaries

### Download Instructions:
```python
# Method 1: Using Kaggle API (recommended)
!pip install kaggle
# Upload your kaggle.json API key to Colab
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download the dataset
!kaggle datasets download -d gowrishankarp/newspaper-text-summarization-cnn-dailymail
!unzip newspaper-text-summarization-cnn-dailymail.zip

# Method 2: Direct download (if API not available)
# The notebook will include alternative download methods
```

## 🔧 Setup Instructions for Each Task

### Task 1 Setup:
1. Download the heartbeat dataset using gdown
2. Extract and preprocess ECG signals
3. Split into train/validation/test sets
4. Normalize and prepare for CNN training

### Task 2 Setup:
1. Load clinical notes dataset
2. Encode categorical labels
3. Tokenize using Bio_ClinicalBERT tokenizer
4. Prepare for sequence classification

### Task 3 Setup:
1. Download CNN/DailyMail dataset from Kaggle
2. Preprocess articles and summaries
3. Tokenize using LLaMA tokenizer
4. Prepare for sequence-to-sequence training

## 📝 Notes

- All datasets are automatically downloaded and processed within their respective notebooks
- GPU memory requirements vary by task (Task 3 requires the most memory)
- Ensure sufficient disk space for dataset storage and model checkpoints
- Some datasets may require API keys or authentication (detailed in notebooks)