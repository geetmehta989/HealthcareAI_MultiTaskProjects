# HealthcareAI Multi-Task Projects

A comprehensive collection of three AI/ML projects focused on healthcare applications, demonstrating state-of-the-art deep learning techniques for medical data analysis.

## 🏥 Project Overview

This repository contains three complete end-to-end AI projects designed for healthcare applications:

1. **🫀 Arrhythmia Classification Using CNN** - ECG signal analysis for cardiac arrhythmia detection
2. **📋 Bio_ClinicalBERT Fine-tuning** - Clinical text classification using domain-specific BERT
3. **📄 Text Summarization with Transformer Models** - Abstractive summarization for medical literature

## 🚀 Quick Start

### Google Colab Setup
Each notebook is designed to run seamlessly in Google Colab with GPU support:

1. Open any notebook in Google Colab
2. Ensure GPU runtime is enabled (`Runtime > Change runtime type > GPU`)
3. Run all cells - dependencies will be installed automatically
4. Datasets will be downloaded and processed automatically

### Local Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/HealthcareAI_MultiTaskProjects.git
cd HealthcareAI_MultiTaskProjects

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebooks
jupyter notebook
```

## 📊 Projects Details

### Task 1: Arrhythmia Classification Using CNN 🫀

**Objective**: Classify different types of cardiac arrhythmias from ECG signals using deep learning.

**Key Features**:
- 1D Convolutional Neural Network architecture
- Real-time ECG signal preprocessing and normalization
- Multi-class classification with comprehensive evaluation
- Interactive visualizations of ECG patterns and model performance

**Dataset**: Heartbeat Dataset (Google Drive)
- **Size**: Variable ECG signal lengths
- **Classes**: Multiple arrhythmia types
- **Download**: Automated via `gdown` library

**Model Architecture**:
- 4 Convolutional blocks with batch normalization
- MaxPooling for feature reduction
- Fully connected layers with dropout regularization
- Optimized for 1D signal processing

**Results**:
- Accuracy: [Achieved during training]
- Precision/Recall/F1-Score: Comprehensive per-class analysis
- Confusion matrices with detailed error analysis
- Real-time prediction capabilities

**Clinical Applications**:
- Automated ECG monitoring systems
- Early detection of cardiac abnormalities
- Support for emergency medical decisions
- Continuous patient monitoring in ICU settings

---

### Task 2: Bio_ClinicalBERT Fine-tuning 📋

**Objective**: Fine-tune Bio_ClinicalBERT for classifying clinical notes into 22 medical specialties.

**Key Features**:
- Domain-specific BERT model (`emilyalsentzer/Bio_ClinicalBERT`)
- 22-class medical specialty classification
- Class imbalance handling with weighted loss functions
- Comprehensive text preprocessing and tokenization

**Dataset**: Clinical Text Classification
- **Categories**: 22 medical specialties (Cardiology, Neurology, Oncology, etc.)
- **Size**: Synthetic dataset with realistic clinical scenarios
- **Format**: JSON with text and label pairs

**Model Configuration**:
- Base Model: Bio_ClinicalBERT (110M parameters)
- Fine-tuning: Full model fine-tuning with class weights
- Optimization: Adam optimizer with learning rate scheduling
- Regularization: Early stopping and dropout

**Results**:
- Weighted F1-Score: [Achieved during training]
- Per-class performance analysis
- Confusion matrix with specialty-specific insights
- Model calibration and confidence analysis

**Clinical Applications**:
- Automated clinical note routing
- Medical specialty recommendation
- Quality assurance for documentation
- Research study categorization

---

### Task 3: Text Summarization with Transformer Models 📄

**Objective**: Fine-tune transformer models for abstractive text summarization using CNN/DailyMail dataset.

**Key Features**:
- BART-Large-CNN model for sequence-to-sequence learning
- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- Multiple evaluation metrics (ROUGE-1, ROUGE-2, ROUGE-L, BLEU)
- Interactive summary generation with customizable parameters

**Dataset**: CNN/DailyMail Summarization
- **Source**: Hugging Face datasets or Kaggle
- **Size**: News articles with human-written summaries
- **Task**: Abstractive text summarization

**Model Architecture**:
- Base: BART-Large-CNN (400M parameters)
- Fine-tuning: LoRA adaptation for efficiency
- Generation: Beam search with length penalty
- Optimization: Mixed precision training with gradient accumulation

**Results**:
- ROUGE-1: [Achieved during training]
- ROUGE-2: [Achieved during training]
- ROUGE-L: [Achieved during training]
- BLEU Score: [Achieved during training]

**Clinical Applications**:
- Medical literature summarization
- Patient report condensation
- Research paper abstracts
- Clinical guideline summaries

## 🛠️ Technical Stack

### Core Libraries
- **Deep Learning**: PyTorch, Transformers (Hugging Face)
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Evaluation**: Rouge-score, NLTK, Evaluate

### Model Architectures
- **CNN**: Custom 1D CNN for ECG signal processing
- **BERT**: Bio_ClinicalBERT for clinical text understanding
- **BART**: BART-Large-CNN for text summarization
- **LoRA**: Parameter-efficient fine-tuning technique

### Optimization Techniques
- **Mixed Precision Training**: FP16 for faster training
- **Gradient Accumulation**: Effective larger batch sizes
- **Learning Rate Scheduling**: Adaptive learning rate adjustment
- **Early Stopping**: Prevent overfitting with patience-based stopping

## 📈 Performance Metrics

### Task 1: ECG Classification
- **Primary**: Accuracy, F1-Score (weighted)
- **Secondary**: Precision, Recall per class
- **Visualization**: Confusion matrices, ROC curves
- **Clinical**: Sensitivity, Specificity for each arrhythmia type

### Task 2: Clinical Text Classification
- **Primary**: Weighted F1-Score, Accuracy
- **Secondary**: Per-class Precision/Recall
- **Analysis**: Confusion matrix, calibration plots
- **Interpretability**: Attention visualization, feature importance

### Task 3: Text Summarization
- **Primary**: ROUGE-1, ROUGE-2, ROUGE-L
- **Secondary**: BLEU score, summary length analysis
- **Quality**: Human evaluation metrics
- **Efficiency**: Inference time, memory usage

## 🔧 Installation and Setup

### System Requirements
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **Memory**: 8GB RAM minimum, 16GB+ recommended
- **Storage**: 5GB free space for datasets and models

### Environment Setup
```bash
# Create virtual environment
python -m venv healthcare_ai_env
source healthcare_ai_env/bin/activate  # Linux/Mac
# healthcare_ai_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# For Jupyter notebook support
pip install jupyter ipywidgets

# For GPU support (if available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Google Colab Setup
```python
# All notebooks include automatic setup
!pip install -r requirements.txt

# GPU verification
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## 📁 Repository Structure

```
HealthcareAI_MultiTaskProjects/
│
├── 📓 Task1_ECG_CNN.ipynb              # Arrhythmia classification notebook
├── 📓 Task2_BioClinicalBERT.ipynb      # Clinical text classification notebook  
├── 📓 Task3_LLaMA_Summarization.ipynb  # Text summarization notebook
│
├── 📋 requirements.txt                  # Python dependencies
├── 📄 README.md                        # This file
├── 📄 dataset_links.md                 # Dataset information and links
│
├── 📊 results/                         # Training results and plots (generated)
│   ├── task1_plots/
│   ├── task2_plots/
│   └── task3_plots/
│
├── 💾 models/                          # Saved models (generated)
│   ├── best_ecg_model.pth
│   ├── best_bio_clinical_bert/
│   └── best_summarization_model/
│
└── 📈 logs/                           # Training logs (generated)
    ├── task1_logs/
    ├── task2_logs/
    └── task3_logs/
```

## 🚀 Usage Examples

### Task 1: ECG Classification
```python
# Load trained model
model = ECG_CNN(input_size=187, n_classes=5)
model.load_state_dict(torch.load('best_ecg_model.pth'))

# Predict on new ECG signal
prediction = model(ecg_signal_tensor)
arrhythmia_type = label_encoder.inverse_transform([prediction.argmax().item()])[0]
```

### Task 2: Clinical Text Classification
```python
# Load fine-tuned model
tokenizer = AutoTokenizer.from_pretrained('./best_bio_clinical_bert')
model = AutoModelForSequenceClassification.from_pretrained('./best_bio_clinical_bert')

# Classify clinical text
inputs = tokenizer(clinical_text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
predicted_specialty = torch.argmax(outputs.logits, dim=-1)
```

### Task 3: Text Summarization
```python
# Load fine-tuned summarization model
tokenizer = AutoTokenizer.from_pretrained('./best_summarization_model')
model = AutoModelForSeq2SeqLM.from_pretrained('./best_summarization_model')

# Generate summary
inputs = tokenizer(article_text, return_tensors="pt", truncation=True, max_length=1024)
summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=4)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
```

## 📊 Results and Benchmarks

### Comparative Performance

| Task | Model | Primary Metric | Score | Training Time | GPU Memory |
|------|-------|---------------|--------|---------------|------------|
| ECG Classification | 1D CNN | F1-Score | TBD | ~30 min | 4GB |
| Clinical Text | Bio_ClinicalBERT | Weighted F1 | TBD | ~45 min | 8GB |
| Summarization | BART-Large-CNN | ROUGE-1 | TBD | ~60 min | 12GB |

### Hardware Recommendations

| Task | Minimum | Recommended | Optimal |
|------|---------|-------------|---------|
| **GPU Memory** | 4GB | 8GB | 16GB+ |
| **RAM** | 8GB | 16GB | 32GB+ |
| **Storage** | 2GB | 5GB | 10GB+ |
| **Training Time** | 2-4 hours | 1-2 hours | 30-60 min |

## 🔬 Research Applications

### Academic Research
- **Methodology**: Reproducible research with detailed documentation
- **Benchmarking**: Standardized evaluation metrics and datasets
- **Comparison**: Multiple model architectures and approaches
- **Publication**: Ready-to-use results and visualizations

### Clinical Deployment
- **Validation**: Comprehensive evaluation on medical datasets
- **Safety**: Robust error handling and uncertainty quantification
- **Scalability**: Efficient inference for real-time applications
- **Integration**: Compatible with existing healthcare systems

### Educational Use
- **Learning**: Step-by-step implementation with explanations
- **Experimentation**: Modular code for easy modification
- **Visualization**: Interactive plots and analysis tools
- **Documentation**: Comprehensive comments and markdown explanations

## 🤝 Contributing

We welcome contributions to improve and extend these healthcare AI projects!

### How to Contribute
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Contribution Areas
- **New Models**: Implement additional architectures
- **Datasets**: Add support for new medical datasets
- **Evaluation**: Develop new metrics and benchmarks
- **Optimization**: Improve training efficiency and performance
- **Documentation**: Enhance explanations and tutorials

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

### Datasets
- **ECG Data**: Heartbeat Dataset contributors
- **Clinical Text**: Bio_ClinicalBERT dataset creators
- **Summarization**: CNN/DailyMail dataset maintainers

### Models
- **Bio_ClinicalBERT**: Emily Alsentzer et al.
- **BART**: Facebook AI Research
- **Transformers**: Hugging Face team

### Libraries
- **PyTorch**: Facebook AI Research
- **Transformers**: Hugging Face
- **Scikit-learn**: Scikit-learn developers
- **Plotly**: Plotly team

## 📞 Contact

For questions, suggestions, or collaborations:

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for general questions
- **Email**: [Your contact information]

## 🔗 Related Projects

- [Medical Image Analysis](https://github.com/example/medical-imaging)
- [Clinical NLP Toolkit](https://github.com/example/clinical-nlp)
- [Healthcare ML Benchmarks](https://github.com/example/healthcare-benchmarks)

---

**⭐ If you find this project helpful, please consider giving it a star!**

**🔄 Stay updated by watching the repository for new releases and improvements.**