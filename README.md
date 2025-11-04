# RoBERTa Stereotype Debiasing

Project for the **Natural Language Processing** exam - Reducing stereotypical bias in language models.

## Description

This project implements **debiasing** techniques to reduce gender, profession, race, and religion stereotypes in the **RoBERTa-base** model through fine-tuning with Masked Language Modeling on balanced datasets.

**Course**: Natural Language Processing  
**Academic Year**: 2024/2025  

## Objectives

1. Evaluate stereotypical bias in the baseline RoBERTa model
2. Apply debiasing techniques through fine-tuning
3. Compare performance before and after training
4. Analyze results by domain (Gender, Profession, Race, Religion)

## Quick Start

```bash
# Installation
pip install -r requirements.txt

# Full execution
python main.py --epochs 3 --learning-rate 2e-5 --batch-size 8

# Visualization only
python main.py --only-visualize
```

## Project Structure

```
src/
├── main.py                 # Main script
├── evaluation.py           # StereoSet evaluation
├── training.py             # Model fine-tuning
├── visualization.py        # Plot generation
├── requirements.txt        # Dependencies
├── results/               # Experiment results (JSON)
├── plots/                 # Generated plots (PNG)
├── checkpoints/           # Training checkpoints
└── debiased_model/        # Final debiased model
```

## Requirements

```txt
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
matplotlib>=3.7.0
numpy>=1.24.0
```
