Uzbek Embedding Evaluation Platform
1. Overview
This repository presents a unified experimental platform for evaluating word embedding models for the Uzbek language.
The platform is designed to provide reproducible, fair, and extensible intrinsic evaluations of both static and contextual embedding models under identical experimental conditions.
The primary objective of this work is to compare the semantic representation quality and computational characteristics of different embedding paradigms when applied to a morphologically rich and low-resource language such as Uzbek.
The platform has been developed to support scientific experimentation, particularly for Scopus- and OAK-level research publications.
2. Supported Models
The current implementation supports two major classes of embedding models:
FastText-UZ
Static word embeddings based on subword information, suitable for morphologically rich languages.
Multilingual BERT (mBERT)
Contextualized embeddings based on Transformer architecture, capable of capturing semantic relations beyond surface-level similarity.
Both models are accessed through a unified interface, enabling direct and unbiased comparison.
3. Evaluation Tasks
The platform focuses on intrinsic evaluation tasks, which assess the internal quality of embedding spaces independently of downstream applications.
Implemented tasks include:
Word Similarity Evaluation
Measuring cosine similarity between word pairs and comparing model scores against human-annotated similarity judgments.
Word Analogy Evaluation
Evaluating relational reasoning using analogy questions of the form:
A is to B as C is to ?
Computational Performance Analysis
Measuring execution time and memory usage for embedding extraction.
4. Datasets
The evaluation is conducted using custom-curated Uzbek-language datasets, including:
Small-scale word similarity pairs annotated for semantic relatedness
Analogy question sets adapted to Uzbek lexical and morphological properties
Due to the limited availability of standardized Uzbek benchmarks, datasets are intentionally kept transparent and reproducible.
5. Experimental Setup
All experiments are executed using a modular Python-based backend architecture.
Key characteristics:
Identical preprocessing and evaluation pipelines for all models
Dynamic model loading through a centralized manager
Separation of model logic, evaluation logic, and visualization
This design ensures fair comparison and simplifies the integration of additional models in future work.
6. Evaluation Metrics
The following metrics are employed:
Cosine Similarity
Used as the primary semantic proximity measure between embedding vectors.
Spearman Rank Correlation
Used to quantify agreement between model-generated similarity scores and human annotations.
Accuracy (Analogy Task)
Percentage of correctly answered analogy questions.
Runtime Latency
Time required to compute embeddings.
Memory Footprint
Peak memory usage during model execution.
7. Visualization
Evaluation results are visualized using Matplotlib to facilitate qualitative comparison.
Implemented plots include:
Scatter plots comparing similarity scores across models
Performance comparison graphs highlighting efficiency trade-offs
All figures are automatically generated and saved to the results/ directory for direct inclusion in academic manuscripts.
8. Reproducibility
Reproducibility is a core design principle of this platform.
Modular codebase
Deterministic evaluation pipeline
Logged experiment configurations
Clear separation of data, models, and results
The platform enables researchers to re-run experiments under identical conditions and verify reported findings.

9. Repository Structure
uz_embidding/
│
├── models/        # Embedding model wrappers (FastText, BERT)
├── utils/         # Tokenization, similarity, evaluation, visualization
├── api/           # Experiment control interfaces
├── results/       # Generated plots and evaluation outputs
├── logs/          # Execution and debugging logs
└── README.md      # Project documentation
10. How to Run Experiments
Clone the repository:
git clone https://github.com/Abdilatif1909/uz_embidding.git
cd uz_embidding
Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
Install dependencies:
pip install -r requirements.txt
Run evaluation scripts:
python main.py
Generated results and figures will be saved automatically.
11. Citation
If you use this platform in academic work, please cite it as:
Meyliyev, A. R. (2025).
A Unified Evaluation Framework for Uzbek Word Embeddings.
GitHub repository: https://github.com/Abdilatif1909/uz_embidding
12. Future Work
Planned extensions include:
Morphology-aware tokenization for Uzbek
Sentence- and document-level embedding evaluation
Fine-tuning contextual models on Uzbek corpora
Expansion of benchmark datasets

