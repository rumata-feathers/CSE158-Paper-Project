# CSE 158 Assignment 2: Paper Citation Prediction and Recommendation

## Project Overview

This project implements a citation prediction and paper recommendation system using the OpenAlex academic dataset. The goal is to predict which papers will cite each other and recommend relevant papers to researchers based on their interests and past citations.

### Dataset: OpenAlex
- **Source**: https://huggingface.co/datasets/sumuks/openalex
- **Size**: 240M+ papers, 2.1B citations
- **Features**: Paper metadata including titles, abstracts, authors, venues, publication dates, citations, and references

## Assignment 2 Requirements

This project addresses all five core components of CSE 158 Assignment 2:

### 1. Dataset Description and Exploratory Analysis
- Number of papers, authors, and citations
- Distribution of citations per paper
- Temporal trends in publications and citations
- Field and venue distributions
- Network analysis of citation graphs

### 2. Predictive Tasks

#### Task 1: Citation Prediction (Classification)
- **Goal**: Predict whether paper A will cite paper B
- **Features**: Temporal features, author overlap, venue similarity, textual similarity, citation context
- **Evaluation**: Accuracy, Precision, Recall, F1-score, AUC-ROC

#### Task 2: Paper Recommendation (Ranking)
- **Goal**: Recommend top-K relevant papers to a researcher
- **Features**: Research interests, citation history, co-authorship networks, topic modeling
- **Evaluation**: Precision@K, Recall@K, NDCG, MRR

### 3. Models

#### Baseline Models:
- Random baseline
- Popularity-based recommendation
- Jaccard similarity (author/venue overlap)

#### Advanced Models:
- Logistic Regression with engineered features
- Collaborative Filtering (matrix factorization)
- Graph-based methods (PageRank, Node2Vec)
- Text-based models (TF-IDF, embeddings)
- Gradient Boosting (XGBoost/LightGBM)

### 4. Related Literature
- Citation recommendation systems
- Link prediction in citation networks
- Collaborative filtering for academic papers
- Graph neural networks for citation analysis
- Temporal dynamics in scientific collaboration

### 5. Results and Analysis
- Model comparison and performance metrics
- Feature importance analysis
- Error analysis and failure cases
- Insights into citation patterns
- Conclusions and future work

## Repository Structure

```
CSE158-Paper-Project/
├── README.md                 # This file
├── data/                     # Data directory (gitignored)
│   ├── raw/                 # Raw OpenAlex data
│   ├── processed/           # Processed features
│   └── splits/              # Train/validation/test splits
├── notebooks/               # Jupyter notebooks
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_advanced_models.ipynb
│   └── 05_results_analysis.ipynb
├── src/                     # Source code
│   ├── data_loader.py      # OpenAlex data loading
│   ├── preprocessing.py    # Data cleaning and preprocessing
│   ├── features.py         # Feature engineering
│   ├── models.py           # Model implementations
│   ├── evaluation.py       # Evaluation metrics
│   └── utils.py            # Utility functions
├── scripts/                 # Standalone scripts
│   ├── download_data.sh    # Script to download OpenAlex subset
│   └── train_model.py      # Training script
├── results/                 # Results and figures
│   ├── figures/            # Plots and visualizations
│   └── metrics/            # Performance metrics
├── report/                  # Final report
│   └── assignment2_report.pdf
├── requirements.txt         # Python dependencies
└── .gitignore              # Git ignore file
```

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/rumata-feathers/CSE158-Paper-Project.git
cd CSE158-Paper-Project
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download data
```bash
# Option 1: Use HuggingFace datasets library
python -c "from datasets import load_dataset; ds = load_dataset('sumuks/openalex', split='train[:100000]'); ds.save_to_disk('data/raw/openalex_sample')"

# Option 2: Download manually from HuggingFace
# Visit: https://huggingface.co/datasets/sumuks/openalex
```

## Usage

### Exploratory Data Analysis
```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

### Feature Engineering
```bash
python src/features.py --input data/raw/openalex_sample --output data/processed/features
```

### Train Models
```bash
python scripts/train_model.py --model logistic_regression --config configs/lr_config.json
```

### Evaluate Models
```bash
python src/evaluation.py --model_path models/lr_model.pkl --test_data data/splits/test.pkl
```

## Key Features Implemented

### Citation Prediction Features:
1. **Temporal Features**: Time difference between papers, publication recency
2. **Author Features**: Author overlap, co-authorship history, author productivity
3. **Venue Features**: Venue similarity, venue prestige
4. **Textual Features**: Title/abstract similarity (cosine similarity, TF-IDF)
5. **Graph Features**: Common citations, citation count, PageRank scores
6. **Topic Features**: Topic overlap (LDA/NMF)

### Recommendation Features:
1. **User Profile**: Research interests, citation patterns
2. **Collaborative Filtering**: Similar users' preferences
3. **Content-Based**: Paper similarity to user's past papers
4. **Hybrid Approaches**: Combining multiple signals

## Evaluation Metrics

### Classification (Citation Prediction):
- Accuracy
- Precision, Recall, F1-Score
- AUC-ROC
- Confusion Matrix

### Ranking (Paper Recommendation):
- Precision@K (K=5, 10, 20)
- Recall@K
- NDCG (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)

## Expected Results

Based on literature and similar projects:
- Citation prediction accuracy: 75-85%
- Recommendation Precision@10: 15-25%
- Significant improvement over baselines with engineered features
- Graph-based and text-based features expected to be most predictive

## Timeline

- [x] Week 1: Dataset selection and initial exploration
- [x] Week 2: Exploratory data analysis
- [ ] Week 3: Feature engineering and baseline models
- [ ] Week 4: Advanced models and hyperparameter tuning
- [ ] Week 5: Results analysis and report writing
- [ ] Week 6: Final report and presentation

## Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter
- datasets (HuggingFace)
- networkx
- scipy
- xgboost
- lightgbm

See `requirements.txt` for complete list with versions.

## References

1. OpenAlex Dataset: https://openalex.org/
2. HuggingFace Dataset: https://huggingface.co/datasets/sumuks/openalex
3. CSE 158 Course Materials: https://cseweb.ucsd.edu/classes/fa25/cse258-a/
4. Related papers on citation prediction and recommendation (to be added)

## Contributors

- Your Name (rumata-feathers)

## License

MIT License - See LICENSE file for details

## Acknowledgments

- CSE 158/258: Web Mining and Recommender Systems, UC San Diego
- Professor Julian McAuley
- OpenAlex team for providing the dataset
