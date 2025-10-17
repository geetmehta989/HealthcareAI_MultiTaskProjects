#!/usr/bin/env python3
"""
Test script to verify all notebooks are working correctly
This script checks for basic syntax and import issues
"""

import os
import json
import subprocess
import sys
from pathlib import Path

def test_notebook_syntax(notebook_path):
    """Test if a notebook has valid JSON syntax"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Check if it's a valid Jupyter notebook
        required_keys = ['cells', 'metadata', 'nbformat', 'nbformat_minor']
        for key in required_keys:
            if key not in notebook:
                return False, f"Missing required key: {key}"
        
        # Check if cells are properly formatted
        if not isinstance(notebook['cells'], list):
            return False, "Cells must be a list"
        
        # Check each cell
        for i, cell in enumerate(notebook['cells']):
            if 'cell_type' not in cell:
                return False, f"Cell {i} missing cell_type"
            if 'source' not in cell:
                return False, f"Cell {i} missing source"
        
        return True, "Valid notebook syntax"
    
    except json.JSONDecodeError as e:
        return False, f"JSON syntax error: {e}"
    except Exception as e:
        return False, f"Error reading notebook: {e}"

def test_imports(notebook_path):
    """Test if all imports in the notebook are valid"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        imports = []
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if 'import' in source or 'from' in source:
                    imports.append(source)
        
        # Basic import validation (this is a simplified check)
        required_imports = [
            'numpy',
            'pandas',
            'torch',
            'matplotlib',
            'sklearn'
        ]
        
        all_source = '\n'.join(imports).lower()
        missing_imports = []
        for req_import in required_imports:
            if req_import not in all_source:
                missing_imports.append(req_import)
        
        if missing_imports:
            return False, f"Missing imports: {missing_imports}"
        
        return True, "All required imports found"
    
    except Exception as e:
        return False, f"Error checking imports: {e}"

def test_notebook_structure(notebook_path):
    """Test if notebook has proper structure for the task"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Check for required sections based on notebook name
        notebook_name = Path(notebook_path).stem
        
        if 'ECG_CNN' in notebook_name:
            required_sections = [
                'Data Download',
                'Model Architecture',
                'Training',
                'Evaluation',
                'Confusion Matrix'
            ]
        elif 'BioClinicalBERT' in notebook_name:
            required_sections = [
                'Dataset Creation',
                'Model Setup',
                'Training',
                'Evaluation',
                'Classification Report'
            ]
        elif 'LLaMA_Summarization' in notebook_name:
            required_sections = [
                'Dataset Creation',
                'Model Setup',
                'Training',
                'Evaluation',
                'ROUGE',
                'BLEU'
            ]
        else:
            required_sections = []
        
        # Check if required sections exist in markdown cells
        all_text = ''
        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown':
                all_text += ' '.join(cell['source']).lower()
        
        missing_sections = []
        for section in required_sections:
            if section.lower() not in all_text:
                missing_sections.append(section)
        
        if missing_sections:
            return False, f"Missing sections: {missing_sections}"
        
        return True, "All required sections found"
    
    except Exception as e:
        return False, f"Error checking structure: {e}"

def main():
    """Main test function"""
    print("Testing Healthcare AI Multi-Task Projects")
    print("=" * 50)
    
    # List of notebooks to test
    notebooks = [
        'Task1_ECG_CNN.ipynb',
        'Task2_BioClinicalBERT.ipynb',
        'Task3_LLaMA_Summarization.ipynb'
    ]
    
    all_passed = True
    
    for notebook in notebooks:
        print(f"\nTesting {notebook}...")
        print("-" * 30)
        
        if not os.path.exists(notebook):
            print(f"❌ ERROR: {notebook} not found")
            all_passed = False
            continue
        
        # Test JSON syntax
        syntax_ok, syntax_msg = test_notebook_syntax(notebook)
        if syntax_ok:
            print(f"✅ JSON syntax: {syntax_msg}")
        else:
            print(f"❌ JSON syntax: {syntax_msg}")
            all_passed = False
        
        # Test imports
        imports_ok, imports_msg = test_imports(notebook)
        if imports_ok:
            print(f"✅ Imports: {imports_msg}")
        else:
            print(f"❌ Imports: {imports_msg}")
            all_passed = False
        
        # Test structure
        structure_ok, structure_msg = test_notebook_structure(notebook)
        if structure_ok:
            print(f"✅ Structure: {structure_msg}")
        else:
            print(f"❌ Structure: {structure_msg}")
            all_passed = False
        
        if syntax_ok and imports_ok and structure_ok:
            print(f"✅ {notebook} - All tests passed")
        else:
            print(f"❌ {notebook} - Some tests failed")
            all_passed = False
    
    # Test additional files
    print(f"\nTesting additional files...")
    print("-" * 30)
    
    additional_files = [
        'README.md',
        'requirements.txt',
        'dataset_links.md'
    ]
    
    for file in additional_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            all_passed = False
    
    # Final result
    print(f"\n{'='*50}")
    if all_passed:
        print("🎉 All tests passed! The project is ready for Colab.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)