# ZeroMAT: Zero-training MATerial Autonomous Analysis with Large Language Model and Retrieval-Augmented Generation


**Hyungjun Kim, Dohun Kang, Jaejun Lee, Jiwon Sun, and Seunghan Lee**

**Elif Ertekin Group, University of Illinois at Urbana-Champaign**  
**Chris Wolverton Group, Northwestern University**  
**Implementation by Jaejun Lee**

---

## Table of Contents
- [Project Overview](#project-overview)
- [Installation & Setup](#installation--setup)
- [Core Components](#core-components)
- [Performance Results](#performance-results)

---

## Project Overview

This repository demonstrates the superior performance and efficiency of TabPFN (large language model) when dealing with unseen small datasets. We leverage **Retrieval-Augmented Generation (RAG)** using OpenAI and Materials Project databases to extract domain-specific knowledge, taking advantage of TabPFN's in-context learning capabilities.

### Key Innovations

- **RAG-Enhanced Feature Selection**: AI-driven domain knowledge extraction for optimal descriptor selection
- **Intelligent Clustering**: Handles TabPFN's 10k sample limitation through chemistry-aware data segmentation  
- **Real-time Adaptability**: Dynamic feature updates based on specific materials properties
- **Efficient Training**: Significantly faster than traditional LLM fine-tuning approaches

### Key Advantages

- **Superior Performance**: TabPFN fine-tuning outperforms LLM-based fine-tuning on unseen small datasets
- **Training Efficiency**: Significantly more efficient learning process compared to traditional approaches  
- **Robust Feature Handling**: Leverages TabPFN's in-context learning to handle:
  - Variable feature lengths without retraining
  - High proportions of missing (NaN) values
  - Direct deployment with high performance expectations

---
## Installation & Setup

### Prerequisites

- Python 3.12.11
- CUDA-compatible GPU (recommended for TabPFN)
- Google Colab or Colab Pro/Pro+ (recommended due to GPU memory requirements)

### Requirements
```bash
pip install pymatgen==2025.6.14
pip install mp-api==0.45.8
pip install openai
```

### Required API Keys

1. **OpenAI API Key**
   - Get from: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
   - Edit `rag.py` line 8: `API_KEY = "your-openai-key-here"`

2. **Materials Project API Key**
   - Get from: [https://next-gen.materialsproject.org/api](https://next-gen.materialsproject.org/api)
   - Set as environment variable: `export MP_API_KEY="your-mp-key-here"`

### Installation
```bash
# Clone repository
git clone https://github.com/Ahri111/ZEROMAT.git
cd ZEROMAT
conda create -n yourenv python=3.12.11
conda activate yourenv

# Install dependencies
pip install -r requirements.txt
```

## Core Components

### 1. RAG Feature Recommender (`rag.py`)

**Purpose**: Intelligent feature selection using domain expertise from GPT-4

**Key Features**:
- Physics-informed feature recommendations
- Interactive chat interface for query refinement
- Automatic parsing and validation of recommended features
- Integration with Materials Project feature mapping

**Usage**:
```bash
python rag.py

print("Choose mode:")
print("1. Interactive chat")
print("2. Quick test")

Example Interaction:
Your question: What are the most important features for predicting bandgap in perovskites?

AI Response:
**RECOMMENDED FEATURES:**
1. formation_energy_per_atom - Thermodynamic stability indicator
2. density - Structural compactness affects electronic properties
3. band_gap - Direct electronic property correlation
4. dielectric_total - Electronic screening effects
5. bulk_modulus - Mechanical stiffness relates to bonding

These Recommended features will be saved as a name of feature_rocommendations.txt
```

### 2. Materials Project Data Fetcher (`production_mp_fetcher.py`)

**Purpose**: Production-ready data augmentation using Materials Project database

**Key Features**:
- Batch processing for large datasets (configurable batch sizes)
- Automatic material ID validation and cleaning
- Robust error handling and retry mechanisms

**Usage**:
```bash
export 
python production_mp_fetcher.py
export MP_API_KEY="your-materials-project-key"

Write all the requirements for implementing  production_mp_fetcher.py as follows:
```

### 3. Script(Colab).ipynb

**Purpose**: To evaluate the performance of our proposed mathod compared to Bert + Finetuning. Note that it is better to implement this Script at Colab rather than your own comptuter

**Quick Start**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Qa7RZCslDFe23i52A31VbJoVkDasYq_8?usp=sharing)

## Performance Results

### Comprehensive Performance Comparison

| Approach | Dataset Size | Training Time | GPU for Training | R² Score | MAE (eV) |
|----------|--------------|---------------|------------|----------|----------|
| **LLM-Prop + Fine-tuning** | Small (<10k) | 38 min | ~8 GB | 0.3881 | 0.7999 |
| **LLM-Prop + TabPFN** | Small (<10k) | 62.34 s | <1 GB | 0.5788 | 0.6559 |
| **LLM-Prop + TabPFN + RAG** | Small (<10k) | 78.32 s | <1 GB | 0.8261 | 0.3652 |
| **LLM-Prop + TabPFN + RAG** | Large (40k) | 157.40 s | <1 GB | **0.9876** | **0.0060** 

### Key Performance Highlights

- **RAG Integration Impact**: 43% improvement in R² score (0.5788 → 0.8261)
- **Training Speed**: 30x faster than LLM-Prop fine-tuning (38 min → 78.32 s)
- **Memory Efficiency**: 2-4x lower GPU memory requirements
- **Accuracy**: Best-in-class R² score of 0.8261 with RAG enhancement
- **Error Reduction**: 44% lower MAE compared to LLM-Prop + TabPFN alone

### Method-Specific Results

**LLM-Prop + Fine-tuning (Baseline)**:
- Longest training time (38 minutes)
- Poorest performance (R² = 0.3881, MAE = 0.7999)
- High computational cost

**LLM-Prop + TabPFN**:
- 37x speed improvement over fine-tuning
- Moderate performance gains (R² = 0.5788)
- Significant memory efficiency

**LLM-Prop + TabPFN + RAG (Our Method)**:
- **Best overall performance** (R² = 0.8261, MAE = 0.3652)
- Minimal additional training time (+16 seconds)
- Superior domain knowledge integration

### Clustering Performance

- **Dataset Size**: Successfully handles ~40k materials
- **Cluster Coherence**: Maintains chemical similarity within clusters
- **Scalability**: Automatic splitting for large datasets
- **Coverage**: 98%+ successful materials property prediction
