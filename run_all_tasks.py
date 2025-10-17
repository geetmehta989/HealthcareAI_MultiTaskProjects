#!/usr/bin/env python3
"""
HealthcareAI_MultiTaskProjects - Run All Tasks Script

This script runs all three healthcare AI projects in sequence:
1. ECG Arrhythmia Classification using CNN
2. Bio_ClinicalBERT Fine-tuning for Clinical Note Classification
3. LLaMA 3.1 Text Summarization

Usage:
    python run_all_tasks.py
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def run_notebook(notebook_path, task_name):
    """Run a Jupyter notebook and handle errors"""
    print(f"\n🚀 Starting {task_name}...")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Convert notebook to Python and run
        result = subprocess.run([
            "jupyter", "nbconvert", "--to", "python", "--execute", 
            "--ExecutePreprocessor.timeout=3600", notebook_path
        ], check=True, capture_output=True, text=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ {task_name} completed successfully in {duration:.2f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {task_name} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_environment():
    """Check if the environment is properly set up"""
    print("🔍 Checking environment...")
    
    # Check if notebooks exist
    notebooks = [
        "Task1_ECG_CNN.ipynb",
        "Task2_BioClinicalBERT.ipynb", 
        "Task3_LLaMA_Summarization.ipynb"
    ]
    
    missing_notebooks = []
    for notebook in notebooks:
        if not os.path.exists(notebook):
            missing_notebooks.append(notebook)
    
    if missing_notebooks:
        print(f"❌ Missing notebooks: {', '.join(missing_notebooks)}")
        return False
    
    # Check if required packages are installed
    try:
        import torch
        import transformers
        import datasets
        import sklearn
        import pandas
        import numpy
        import matplotlib
        print("✅ Required packages are available")
    except ImportError as e:
        print(f"❌ Missing required packages: {e}")
        print("Please run: python setup_environment.py")
        return False
    
    return True

def main():
    """Main function to run all tasks"""
    print("🏥 HealthcareAI_MultiTaskProjects - Run All Tasks")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check environment
    if not check_environment():
        print("❌ Environment check failed. Please fix the issues above.")
        sys.exit(1)
    
    # Define tasks
    tasks = [
        ("Task1_ECG_CNN.ipynb", "Task 1: ECG Arrhythmia Classification"),
        ("Task2_BioClinicalBERT.ipynb", "Task 2: Bio_ClinicalBERT Fine-tuning"),
        ("Task3_LLaMA_Summarization.ipynb", "Task 3: Text Summarization")
    ]
    
    # Run each task
    successful_tasks = 0
    total_tasks = len(tasks)
    
    for notebook, task_name in tasks:
        if run_notebook(notebook, task_name):
            successful_tasks += 1
        else:
            print(f"⚠️ {task_name} failed. Continuing with next task...")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    print(f"Total tasks: {total_tasks}")
    print(f"Successful: {successful_tasks}")
    print(f"Failed: {total_tasks - successful_tasks}")
    print(f"Success rate: {successful_tasks/total_tasks*100:.1f}%")
    
    if successful_tasks == total_tasks:
        print("🎉 All tasks completed successfully!")
    else:
        print("⚠️ Some tasks failed. Please check the error messages above.")
    
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()