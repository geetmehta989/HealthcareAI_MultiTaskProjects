# Dataset Links and Instructions

This document contains all dataset sources and download instructions for the three projects.

---

## 📊 Task 1: ECG Arrhythmia Classification

### Dataset: MIT-BIH Arrhythmia Heartbeat Dataset

**Source**: Google Drive  
**Link**: https://drive.google.com/file/d/1xAs-CjlpuDqUT2EJUVR5cPuqTUdw2uQg/view?usp=sharing

**Format**: CSV files containing preprocessed ECG signals
- `mitbih_train.csv` - Training data
- `mitbih_test.csv` - Test data

**Download Method**:
```python
# Method 1: Using gdown
!pip install gdown
import gdown
gdown.download('https://drive.google.com/uc?id=1xAs-CjlpuDqUT2EJUVR5cPuqTUdw2uQg', 'arrhythmia.zip', quiet=False)
!unzip arrhythmia.zip

# Method 2: Mount Google Drive in Colab
from google.colab import drive
drive.mount('/content/drive')
```

**Dataset Details**:
- **Classes**: 5 types of heartbeats
  - Normal (N)
  - Supraventricular premature beat (S)
  - Premature ventricular contraction (V)
  - Fusion of ventricular and normal beat (F)
  - Unclassifiable beat (Q)
- **Features**: 187 time-series signal values per heartbeat
- **Size**: ~100K training samples, ~20K test samples
- **Preprocessing**: Already segmented and normalized

**Reference**:
- Original Source: PhysioNet MIT-BIH Arrhythmia Database
- Paper: Moody GB, Mark RG. "The impact of the MIT-BIH Arrhythmia Database"

---

## 📄 Task 2: Clinical Text Classification

### Dataset: Clinical Sentence Classification (22 Categories)

**Source**: JSON dataset (Medical notes)  
**Format**: JSON with sentence-label pairs

**Sample Structure**:
```json
{
  "data": [
    {
      "text": "Patient presents with chest pain and shortness of breath.",
      "label": "chief_complaint"
    },
    {
      "text": "Blood pressure: 140/90 mmHg, Heart rate: 88 bpm",
      "label": "vital_signs"
    }
  ]
}
```

**Categories (22 classes)**:
- Chief Complaint
- History of Present Illness
- Past Medical History
- Medications
- Allergies
- Family History
- Social History
- Review of Systems
- Physical Examination
- Vital Signs
- Laboratory Results
- Imaging Results
- Assessment
- Diagnosis
- Treatment Plan
- Procedures
- Discharge Instructions
- Follow-up
- Prognosis
- Patient Education
- Consent
- Other

**Download Method**:
```python
# If hosted on Google Drive
import gdown
gdown.download('DRIVE_LINK_HERE', 'clinical_notes.json', quiet=False)

# Alternative: Create synthetic dataset in notebook
# The notebook includes code to generate sample data
```

**Notes**:
- This is a specialized medical NLP task
- Requires Bio_ClinicalBERT for domain-specific understanding
- May have class imbalance - handle with weighted loss

---

## 📰 Task 3: Text Summarization

### Dataset: CNN/DailyMail Summarization Dataset

**Source**: Kaggle  
**Link**: https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail

**Format**: CSV with article-summary pairs
- Columns: `article`, `highlights`

**Download Method**:

#### Option 1: Kaggle API (Recommended)
```python
# 1. Get Kaggle API credentials
# - Go to https://www.kaggle.com/settings
# - Click "Create New API Token"
# - Upload kaggle.json to Colab

# 2. Set up Kaggle API
!pip install kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# 3. Download dataset
!kaggle datasets download -d gowrishankarp/newspaper-text-summarization-cnn-dailymail
!unzip newspaper-text-summarization-cnn-dailymail.zip
```

#### Option 2: Hugging Face Datasets
```python
from datasets import load_dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
```

**Dataset Details**:
- **Size**: ~300K article-summary pairs
- **Article Length**: Average 500-1000 words
- **Summary Length**: Average 50-150 words
- **Domain**: News articles (CNN and DailyMail)
- **Task**: Abstractive summarization

**Preprocessing Requirements**:
- Tokenization with LLaMA tokenizer
- Truncation to max sequence length (512-1024 tokens)
- Special tokens for generation
- DataCollator for sequence-to-sequence

**Alternative Datasets** (if needed):
- XSum: Extreme summarization (1-sentence summaries)
- PubMed: Medical/scientific paper summarization
- arXiv: Scientific paper summarization

---

## 🔑 Required API Keys and Credentials

### For Task 3 (Kaggle):
1. **Kaggle API Token**:
   - Sign in to Kaggle
   - Go to Account Settings
   - Scroll to API section
   - Click "Create New API Token"
   - Download `kaggle.json`

### For All Tasks (Optional):
1. **Hugging Face Token** (for gated models like LLaMA):
   - Sign up at https://huggingface.co/
   - Go to Settings → Access Tokens
   - Create token with read permissions
   - Use: `huggingface-cli login`

---

## 💾 Storage Requirements

- **Task 1**: ~100 MB (ECG dataset)
- **Task 2**: ~10-50 MB (JSON dataset)
- **Task 3**: ~500 MB - 1 GB (CNN/DailyMail)
- **Models**: 2-10 GB (depending on task)

**Total**: ~5-15 GB including models and checkpoints

---

## 🔄 Data Refresh and Updates

- Datasets are static for reproducibility
- Original sources may have newer versions
- Check PhysioNet, Kaggle, and Hugging Face for updates

---

## ⚠️ Important Notes

1. **Privacy**: Clinical datasets may contain sensitive information - use only for educational purposes
2. **Licensing**: Check individual dataset licenses before commercial use
3. **Ethics**: Follow medical AI ethics guidelines when working with healthcare data
4. **Validation**: Always validate model predictions before any real-world application

---

**Last Updated**: October 2025
