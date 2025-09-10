# Content Recommendation System

A comprehensive content recommendation system that combines content-based filtering, collaborative filtering, and hybrid approaches to recommend posts, videos, or articles to users.

## Features

- **Multiple Recommendation Approaches**:
  - Content-based filtering using Sentence-BERT embeddings and FAISS
  - Collaborative filtering using ALS and Neural Collaborative Filtering
  - Hybrid re-ranking system combining multiple approaches

- **Advanced Preprocessing**:
  - Sentence-BERT embeddings for semantic content understanding
  - Category encoding and normalization
  - Simulated user interaction data

- **Comprehensive Evaluation**:
  - Precision@K, Recall@K, NDCG@K metrics
  - A/B testing framework

- **Production-Ready Deployment**:
  - FastAPI backend with RESTful APIs
  - Streamlit frontend for interactive recommendations
  - Docker containerization

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Generate sample data:
```bash
python src/data/generate_data.py
```

3. Train models:
```bash
python src/models/train_models.py
```

4. Start the API server:
```bash
uvicorn src.api.main:app --reload
```

5. Launch the Streamlit interface:
```bash
streamlit run streamlit_app/main.py
```

## Project Structure

```
├── src/
│   ├── data/           # Data generation and loading
│   ├── preprocessing/  # Feature engineering and embeddings
│   ├── models/         # Recommendation models
│   ├── evaluation/     # Evaluation metrics
│   └── api/           # FastAPI backend
├── streamlit_app/     # Streamlit frontend
├── config/           # Configuration files
├── tests/           # Test files
└── data/           # Data storage
```

## API Endpoints

- `GET /recommendations/{user_id}` - Get recommendations for a user
- `POST /feedback` - Submit user feedback
- `GET /health` - Health check
- `GET /metrics` - Get recommendation metrics

## License

MIT License