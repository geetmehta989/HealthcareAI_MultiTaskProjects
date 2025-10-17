# Healthcare AI Multi-Task Projects

This repository contains three comprehensive AI/ML projects focused on healthcare applications, designed to run seamlessly in Google Colab with GPU support.

## 🏥 Project Overview

### Task 1: Arrhythmia Classification Using CNN
- **Objective**: Classify arrhythmias from ECG signals using Convolutional Neural Networks
- **Dataset**: Heartbeat Dataset from Google Drive
- **Model**: 1D CNN for time-series classification
- **Evaluation**: Accuracy, Precision, Recall, F1-score, Confusion Matrix

### Task 2: Bio_ClinicalBERT Fine-tuning
- **Objective**: Classify clinical note sentences into 22 categories
- **Dataset**: Clinical notes JSON dataset
- **Model**: Fine-tuned Bio_ClinicalBERT using Hugging Face Transformers
- **Evaluation**: Accuracy, F1-score, Confusion Matrix

### Task 3: LLaMA 3.1 Text Summarization
- **Objective**: Fine-tune LLaMA 3.1 for abstractive summarization
- **Dataset**: CNN/DailyMail Summarization dataset from Kaggle
- **Model**: Fine-tuned LLaMA 3.1 for sequence-to-sequence learning
- **Evaluation**: ROUGE and BLEU metrics

## 🚀 Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/HealthcareAI_MultiTaskProjects.git
   cd HealthcareAI_MultiTaskProjects
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run in Google Colab**:
   - Open any of the three Jupyter notebooks
   - Ensure GPU runtime is enabled
   - Run all cells sequentially

## 📁 Repository Structure

```
HealthcareAI_MultiTaskProjects/
│
├── Task1_ECG_CNN.ipynb          # ECG arrhythmia classification
├── Task2_BioClinicalBERT.ipynb  # Clinical note classification
├── Task3_LLaMA_Summarization.ipynb  # Text summarization
├── README.md                     # This file
├── requirements.txt              # Python dependencies
└── dataset_links.md             # Dataset download links and instructions
```

## 🔧 Requirements

- Python 3.8+
- Google Colab with GPU runtime (recommended)
- CUDA-compatible GPU for training
- 8GB+ RAM recommended

## 📊 Key Features

- **Self-contained notebooks**: Each notebook includes all necessary code, data loading, preprocessing, training, and evaluation
- **GPU optimization**: All models are optimized for GPU training
- **Comprehensive evaluation**: Detailed metrics, plots, and analysis for each task
- **Production-ready**: Clean, well-documented code suitable for real-world applications

## 🎯 Performance Highlights

- **Task 1**: Achieves high accuracy on ECG arrhythmia classification
- **Task 2**: Demonstrates effective fine-tuning of clinical language models
- **Task 3**: Produces high-quality abstractive summaries with competitive ROUGE scores

## 📚 References

- [Medium Tutorial on CNN for Arrhythmia Detection](https://medium.com)
- [Bio_ClinicalBERT Paper](https://arxiv.org/abs/1904.03323)
- [LLaMA 3.1 Documentation](https://huggingface.co/meta-llama)

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this project.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.