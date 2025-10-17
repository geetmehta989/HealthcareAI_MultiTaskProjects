# HealthcareAI MultiTask Projects

This repository contains three end-to-end AI/ML healthcare projects designed to run on Google Colab with GPU support.

## 📋 Projects Overview

### 1️⃣ Task 1: Arrhythmia Classification Using CNN
- **Objective**: Classify arrhythmias from ECG signals using a 1D CNN
- **Dataset**: MIT-BIH Arrhythmia Heartbeat Dataset
- **Techniques**: 1D Convolutional Neural Networks, Signal Processing
- **Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Notebook**: `Task1_ECG_CNN.ipynb`

### 2️⃣ Task 2: Clinical Text Classification with Bio_ClinicalBERT
- **Objective**: Fine-tune Bio_ClinicalBERT for clinical note sentence classification (22 categories)
- **Model**: `emilyalsentzer/Bio_ClinicalBERT`
- **Techniques**: Transfer Learning, BERT Fine-tuning, Text Classification
- **Metrics**: Accuracy, F1-Score (Macro/Weighted), Confusion Matrix
- **Notebook**: `Task2_BioClinicalBERT.ipynb`

### 3️⃣ Task 3: Medical Text Summarization with LLaMA
- **Objective**: Fine-tune LLaMA model for abstractive text summarization
- **Dataset**: CNN/DailyMail Summarization Dataset
- **Techniques**: Sequence-to-Sequence Learning, Parameter-Efficient Fine-Tuning (LoRA)
- **Metrics**: ROUGE (1/2/L), BLEU Score
- **Notebook**: `Task3_LLaMA_Summarization.ipynb`

## 🚀 Getting Started

### Prerequisites
- Google Colab account (free or Pro for better GPU access)
- Google Drive account (for dataset storage)
- Kaggle API credentials (for Task 3 dataset)
- Hugging Face account (optional, for model access)

### Running the Notebooks

1. **Upload to Google Colab**:
   - Upload each `.ipynb` file to Google Colab
   - Or open directly from GitHub using File → Open Notebook → GitHub

2. **Enable GPU**:
   - Runtime → Change runtime type → Hardware accelerator → GPU (T4 or better)

3. **Run All Cells**:
   - Each notebook is self-contained with all necessary installations
   - Follow the markdown instructions within each notebook

### Dataset Links
See `dataset_links.md` for all dataset sources and download instructions.

## 📦 Project Structure

```
HealthcareAI_MultiTaskProjects/
│
├── Task1_ECG_CNN.ipynb              # ECG arrhythmia classification
├── Task2_BioClinicalBERT.ipynb      # Clinical text classification
├── Task3_LLaMA_Summarization.ipynb  # Medical text summarization
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
└── dataset_links.md                  # Dataset sources and instructions
```

## 🛠️ Technologies Used

- **Deep Learning Frameworks**: TensorFlow/Keras, PyTorch
- **NLP Libraries**: Hugging Face Transformers, Tokenizers
- **Data Processing**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Evaluation**: ROUGE, BLEU, Scikit-learn metrics

## 📊 Results Summary

### Task 1: ECG Classification
- Target: Multi-class arrhythmia classification
- Expected Accuracy: >95% on test set
- Includes confusion matrix and per-class metrics

### Task 2: Bio_ClinicalBERT
- Target: 22-class clinical sentence classification
- Uses pre-trained biomedical BERT
- Handles class imbalance with weighted loss

### Task 3: LLaMA Summarization
- Target: Abstractive text summarization
- Uses LoRA for efficient fine-tuning
- Evaluated with ROUGE and BLEU metrics

## 🎓 Learning Outcomes

- End-to-end ML pipeline development
- Healthcare-specific AI applications
- Transfer learning and fine-tuning techniques
- Model evaluation and interpretation
- Working with medical/clinical datasets
- GPU-accelerated training in Colab

## 📝 Notes

- All notebooks include detailed markdown explanations
- GPU is required for reasonable training times
- Estimated runtime: 30-60 minutes per notebook on Colab GPU
- Models are saved and can be downloaded for deployment

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements.

## 📄 License

This project is for educational purposes. Please check individual dataset licenses before commercial use.

## 👨‍💻 Author

Healthcare AI Assignment - Complete Multi-Task Project Suite

---

**Last Updated**: October 2025
