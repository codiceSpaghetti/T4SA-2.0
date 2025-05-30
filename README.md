# The Emotions of the Crowd: Learning Image Sentiment from Tweets via Cross-modal Distillation

[![arXiv](https://img.shields.io/badge/arXiv-2304.14942-b31b1b.svg)](https://arxiv.org/abs/2304.14942)  [![Website](https://img.shields.io/badge/🌐_Website-blue)](https://fabiocarrara.github.io/cross-modal-visual-sentiment-analysis/)

![Method Overview](https://fabiocarrara.github.io/cross-modal-visual-sentiment-analysis/static/images/overview.png)

## 📖 Introduction

This work introduces **T4SA 2.0**, a comprehensive dataset and cross-modal learning method for training visual models for **Sentiment Analysis in the Twitter domain**.

Our approach fine-tunes Vision-Transformer (ViT) models pre-trained on ImageNet-21k, achieving remarkable results on external benchmarks with manual annotations, even ***surpassing the current State-of-the-Art!***

### Key Features
- 🎯 **Cross-modal distillation**: Leverages both visual and textual information from social media
- 📊 **Large-scale dataset**: ~3.7M images crawled from April 1 to June 30
- 🤖 **Automated labeling**: Uses distant supervision to minimize human annotation efforts
- 🏆 **SOTA performance**: Beats current state-of-the-art on multiple benchmarks
- 🔧 **Ready-to-use models**: Pre-trained models available for immediate deployment

The cross-modal teacher-student learning technique eliminates the need for human annotators while enabling the creation of vast training sets, supporting future research in training robust visual models as parameter counts continue to grow exponentially.

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- CUDA-compatible GPU (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git config --global http.postBuffer 1048576000
   git clone --recursive https://github.com/fabiocarrara/cross-modal-visual-sentiment-analysis.git
   cd cross-modal-visual-sentiment-analysis
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Install system dependencies (Linux/Ubuntu)**
   ```bash
   chmod +x setup.sh
   sudo ./setup.sh
   ```

## 🧪 Usage

### Benchmark Evaluation

#### Test a single model on a benchmark
```bash
python3 scripts/test_benchmark.py -m <model_name> -b <benchmark_name>
```

**Available models:**
- `boosted_model` (recommended)
- `ViT_L16`, `ViT_L32`, `ViT_B16`, `ViT_B32`
- `merged_T4SA`, `bal_flat_T4SA2.0`, `bal_T4SA2.0`, `unb_T4SA2.0`
- `B-T4SA_1.0_upd_filt`, `B-T4SA_1.0_upd`, `B-T4SA_1.0`

**Available benchmarks:**
- `5agree`, `4agree`, `3agree`
- `FI_complete`, `emotion_ROI_test`, `twitter_testing_2`

#### Five-fold cross-validation
```bash
python3 scripts/5_fold_cross.py -b <benchmark_name>
```
*Returns mean accuracy, standard deviation, and saves predictions*

#### Fine-tune on FI dataset
```bash
python3 scripts/fine_tune_FI.py
```

## 🏗️ Project Structure

```
.
├── dataset/
│   ├── benchmark/
│   │   ├── EmotionROI/
│   │   │   └── images/
│   │   │       ├── anger/
│   │   │       ├── disgust/
│   │   │       ├── fear/
│   │   │       ├── joy/
│   │   │       ├── sadness/
│   │   │       └── surprise/
│   │   ├── FI/
│   │   │   ├── images/
│   │   │   │   ├── amusement/
│   │   │   │   ├── anger/
│   │   │   │   ├── awe/
│   │   │   │   ├── contentment/
│   │   │   │   ├── disgust/
│   │   │   │   ├── excitement/
│   │   │   │   ├── fear/
│   │   │   │   └── sadness/
│   │   │   ├── split_1/ ... split_5/
│   │   ├── Twitter Testing Dataset I/
│   │   └── Twitter Testing Dataset II/
│   ├── t4sa 1.0/
│   │   ├── dataset with new labels/
│   │   └── original dataset/
│   └── t4sa 2.0/
│       ├── bal_T4SA2.0/
│       ├── bal_flat_T4SA2.0/
│       ├── img/
│       ├── merged_T4SA/
│       └── unb_T4SA2.0/
├── models/
├── notebooks/
├── predictions/
├── scripts/
├── requirements.txt
├── setup.sh
└── README.md
```

## 🎯 Pre-trained Models

Our models achieve state-of-the-art performance on Twitter sentiment analysis benchmarks:

<table align="center">
  <tr>
    <th colspan="2"></th>
    <th colspan="3">Confidence Filter Threshold</th>
    <th colspan="1"></th>
    <th colspan="3">Accuracy on Twitter Dataset (%)</th>
  </tr>
  <tr>
    <th>Model</th>
    <th>Dataset</th>
    <th>Pos</th>
    <th>Neu</th>
    <th>Neg</th>
    <th>Architecture</th>
    <th>5 agree</th>
    <th>≥4 agree</th>
    <th>≥3 agree</th>
  </tr>
  <tr>
    <td><a href="https://github.com/fabiocarrara/cross-modal-visual-sentiment-analysis/releases/download/v0.1.0/B-T4SA_1.0_upd.h5">Model 3.1</a></td>
    <td>A</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>ViT-B/32</td>
    <td>82.2</td>
    <td>78.0</td>
    <td>75.5</td>
  </tr>
  <tr>
    <td><a href="https://github.com/fabiocarrara/cross-modal-visual-sentiment-analysis/releases/download/v0.1.0/B-T4SA_1.0_upd_filt.h5">Model 3.2</a></td>
    <td>A</td>
    <td>0.70</td>
    <td>0.70</td>
    <td>0.70</td>
    <td>ViT-B/32</td>
    <td>84.7</td>
    <td>79.7</td>
    <td>76.6</td>
  </tr>
  <tr>
    <td><a href="https://github.com/fabiocarrara/cross-modal-visual-sentiment-analysis/releases/download/v0.1.0/bal_flat_T4SA2.0.h5">Model 3.3</a></td>
    <td>B</td>
    <td>0.70</td>
    <td>0.70</td>
    <td>0.70</td>
    <td>ViT-B/32</td>
    <td>82.3</td>
    <td>78.7</td>
    <td>75.3</td>
  </tr>
  <tr>
    <td><a href="https://github.com/fabiocarrara/cross-modal-visual-sentiment-analysis/releases/download/v0.1.0/bal_T4SA2.0.h5">Model 3.4</a></td>
    <td>B</td>
    <td>0.90</td>
    <td>0.90</td>
    <td>0.70</td>
    <td>ViT-B/32</td>
    <td>84.4</td>
    <td>80.3</td>
    <td>77.1</td>
  </tr>
  <tr>
    <td><a href="https://github.com/fabiocarrara/cross-modal-visual-sentiment-analysis/releases/download/v0.1.0/ViT_B32.h5">Model 3.5</a></td>
    <td>A+B</td>
    <td>0.90</td>
    <td>0.90</td>
    <td>0.70</td>
    <td>ViT-B/32</td>
    <td>86.5</td>
    <td>82.6</td>
    <td>78.9</td>
  </tr>
  <tr>
    <td><a href="https://github.com/fabiocarrara/cross-modal-visual-sentiment-analysis/releases/download/v0.1.0/ViT_L32.h5">Model 3.6</a></td>
    <td>A+B</td>
    <td>0.90</td>
    <td>0.90</td>
    <td>0.70</td>
    <td>ViT-L/32</td>
    <td>85.0</td>
    <td>82.4</td>
    <td>79.4</td>
  </tr>
  <tr>
    <td><a href="https://github.com/fabiocarrara/cross-modal-visual-sentiment-analysis/releases/download/v0.1.0/ViT_B16.h5">Model 3.7</a></td>
    <td>A+B</td>
    <td>0.90</td>
    <td>0.90</td>
    <td>0.70</td>
    <td>ViT-B/16</td>
    <td>87.0</td>
    <td>83.1</td>
    <td>79.4</td>
  </tr>
  <tr>
    <td><a href="https://github.com/fabiocarrara/cross-modal-visual-sentiment-analysis/releases/download/v0.1.0/ViT_L16.h5">Model 3.8</a></td>
    <td>A+B</td>
    <td>0.90</td>
    <td>0.90</td>
    <td>0.70</td>
    <td>ViT-L/16</td>
    <td><strong>87.8</strong></td>
    <td><strong>84.8</strong></td>
    <td><strong>81.9</strong></td>
  </tr>
</table>

**Best performing model:** ViT-L/16 (Model 3.8) achieves 87.8% accuracy on the most challenging 5-agreement benchmark.

## 📊 Dataset

The T4SA 2.0 dataset will be made available soon. It contains:
- **~3.7M images** from Twitter
- **Cross-modal annotations** using distant supervision
- **Multiple confidence thresholds** for different use cases
- **Balanced and unbalanced** versions

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{serra2023emotions,
  author    = {Serra, Alessio and Carrara, Fabio and Tesconi, Maurizio and Falchi, Fabrizio},
  editor    = {Kobi Gal and Ann Now{\'{e}} and Grzegorz J. Nalepa and Roy Fairstein and Roxana Radulescu},
  title     = {The Emotions of the Crowd: Learning Image Sentiment from Tweets via Cross-Modal Distillation},
  booktitle = {{ECAI} 2023 - 26th European Conference on Artificial Intelligence, September 30 - October 4, 2023, Krak{\'{o}}w, Poland - Including 12th Conference on Prestigious Applications of Intelligent Systems ({PAIS} 2023)},
  series    = {Frontiers in Artificial Intelligence and Applications},
  volume    = {372},
  pages     = {2089--2096},
  publisher = {{IOS} Press},
  year      = {2023},
  url       = {https://doi.org/10.3233/FAIA230503},
  doi       = {10.3233/FAIA230503},
}
```

<p align="center">
  <a href="https://github.com/fabiocarrara/cross-modal-visual-sentiment-analysis/issues">Report Bug</a> •
  <a href="https://github.com/fabiocarrara/cross-modal-visual-sentiment-analysis/issues">Request Feature</a> •
  <a href="https://fabiocarrara.github.io/cross-modal-visual-sentiment-analysis/">Website</a>
</p>