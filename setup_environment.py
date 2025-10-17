#!/usr/bin/env python3
"""
HealthcareAI_MultiTaskProjects - Environment Setup Script

This script sets up the environment for running the three healthcare AI projects:
1. ECG Arrhythmia Classification using CNN
2. Bio_ClinicalBERT Fine-tuning for Clinical Note Classification
3. LLaMA 3.1 Text Summarization

Usage:
    python setup_environment.py
"""

import subprocess
import sys
import os
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported. Please use Python 3.8 or higher.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_gpu_availability():
    """Check GPU availability"""
    print("🖥️ Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU available: {gpu_name} ({gpu_memory:.1f} GB)")
            return True
        else:
            print("⚠️ No GPU available. Training will be slower on CPU.")
            return False
    except ImportError:
        print("⚠️ PyTorch not installed yet. Will check after installation.")
        return False

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    # Core packages
    packages = [
        "torch>=2.0.0",
        "tensorflow>=2.12.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "accelerate>=0.20.0",
        "evaluate>=0.4.0",
        "rouge-score>=0.1.2",
        "sacrebleu>=2.3.0",
        "scikit-learn>=1.3.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "gdown>=4.7.0",
        "kaggle>=1.5.0",
        "h5py>=3.9.0",
        "tqdm>=4.65.0",
        "jupyter>=1.0.0",
        "ipywidgets>=8.0.0",
        "requests>=2.31.0",
        "urllib3>=2.0.0"
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️ Failed to install {package}. You may need to install it manually.")
    
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating project directories...")
    
    directories = [
        "data",
        "models",
        "results",
        "logs",
        "checkpoints"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def setup_kaggle():
    """Setup Kaggle API"""
    print("🔑 Setting up Kaggle API...")
    
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    
    print("📝 To use Kaggle datasets, please:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Click 'Create New API Token'")
    print("3. Download kaggle.json")
    print(f"4. Place it in {kaggle_dir}/kaggle.json")
    print("5. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
    
    return True

def verify_installation():
    """Verify that all packages are installed correctly"""
    print("🔍 Verifying installation...")
    
    required_modules = [
        "torch",
        "tensorflow",
        "transformers",
        "datasets",
        "sklearn",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "plotly",
        "gdown",
        "kaggle",
        "evaluate",
        "rouge_score",
        "sacrebleu"
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️ Failed to import: {', '.join(failed_imports)}")
        print("Please install these packages manually:")
        for module in failed_imports:
            print(f"  pip install {module}")
        return False
    
    print("✅ All required packages are installed correctly")
    return True

def main():
    """Main setup function"""
    print("🚀 HealthcareAI_MultiTaskProjects - Environment Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check GPU availability
    check_gpu_availability()
    
    # Install requirements
    if not install_requirements():
        print("⚠️ Some packages failed to install. Please check the errors above.")
    
    # Create directories
    create_directories()
    
    # Setup Kaggle
    setup_kaggle()
    
    # Verify installation
    if not verify_installation():
        print("⚠️ Installation verification failed. Please check the errors above.")
        sys.exit(1)
    
    print("\n🎉 Environment setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Open the notebooks in Google Colab or Jupyter")
    print("2. Enable GPU runtime if available")
    print("3. Run the notebooks in order:")
    print("   - Task1_ECG_CNN.ipynb")
    print("   - Task2_BioClinicalBERT.ipynb")
    print("   - Task3_LLaMA_Summarization.ipynb")
    print("\n🔗 For Google Colab:")
    print("   - Upload the notebooks to Google Colab")
    print("   - Enable GPU runtime (Runtime > Change runtime type > GPU)")
    print("   - Run all cells in each notebook")
    
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main()