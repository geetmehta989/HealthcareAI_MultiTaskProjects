#!/usr/bin/env python3
"""
Colab Setup Script for Healthcare AI Multi-Task Projects
This script sets up the environment and installs dependencies
"""

import subprocess
import sys
import os

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

def main():
    """Main setup function"""
    print("🚀 Setting up Healthcare AI Multi-Task Projects for Colab")
    print("=" * 60)
    
    # Check if we're in Colab
    try:
        import google.colab
        print("✅ Running in Google Colab")
        is_colab = True
    except ImportError:
        print("⚠️  Not running in Google Colab - some features may not work")
        is_colab = False
    
    # Install system dependencies
    if is_colab:
        commands = [
            ("apt-get update", "Updating package lists"),
            ("apt-get install -y git", "Installing git"),
        ]
        
        for command, description in commands:
            if not run_command(f"!{command}", description):
                print(f"⚠️  {description} failed, but continuing...")
    
    # Install Python packages
    print("\n📦 Installing Python packages...")
    
    # Install from requirements.txt
    if os.path.exists('requirements.txt'):
        if not run_command("pip install -r requirements.txt", "Installing packages from requirements.txt"):
            print("⚠️  Failed to install from requirements.txt, trying individual packages...")
            
            # Install key packages individually
            key_packages = [
                "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
                "transformers datasets accelerate evaluate",
                "rouge-score sacrebleu",
                "scikit-learn matplotlib seaborn plotly",
                "nltk gdown kaggle",
                "tqdm ipywidgets"
            ]
            
            for package in key_packages:
                run_command(f"pip install {package}", f"Installing {package.split()[0]}")
    else:
        print("⚠️  requirements.txt not found, installing key packages...")
        key_packages = [
            "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
            "transformers datasets accelerate evaluate",
            "rouge-score sacrebleu",
            "scikit-learn matplotlib seaborn plotly",
            "nltk gdown kaggle",
            "tqdm ipywidgets"
        ]
        
        for package in key_packages:
            run_command(f"pip install {package}", f"Installing {package.split()[0]}")
    
    # Download NLTK data
    print("\n📚 Downloading NLTK data...")
    nltk_commands = [
        "import nltk; nltk.download('punkt')",
        "import nltk; nltk.download('stopwords')",
        "import nltk; nltk.download('punkt_tab')"
    ]
    
    for command in nltk_commands:
        run_command(f"python -c \"{command}\"", f"Downloading NLTK data")
    
    # Check GPU availability
    print("\n🖥️  Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  CUDA not available - training will be slower")
    except ImportError:
        print("❌ PyTorch not installed properly")
    
    # Check key imports
    print("\n🔍 Checking key imports...")
    key_imports = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("sklearn", "Scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("nltk", "NLTK")
    ]
    
    all_imports_ok = True
    for module, name in key_imports:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name}")
            all_imports_ok = False
    
    # Final status
    print("\n" + "=" * 60)
    if all_imports_ok:
        print("🎉 Setup completed successfully!")
        print("📝 You can now run the notebooks:")
        print("   1. Task1_ECG_CNN.ipynb - ECG Arrhythmia Classification")
        print("   2. Task2_BioClinicalBERT.ipynb - Clinical Note Classification")
        print("   3. Task3_LLaMA_Summarization.ipynb - Text Summarization")
        print("\n💡 Tips:")
        print("   - Enable GPU runtime in Colab for faster training")
        print("   - Run cells sequentially for best results")
        print("   - Check the README.md for detailed instructions")
    else:
        print("❌ Setup completed with errors")
        print("   Some packages may not be installed correctly")
        print("   Please check the error messages above")
    
    return all_imports_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)