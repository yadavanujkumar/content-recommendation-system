"""
Content-based recommendation system using FAISS for similarity search.
"""

import numpy as np
import pandas as pd
import faiss
from typing import List, Tuple, Optional
import pickle
import os
import yaml
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """Content-based recommender using FAISS for efficient similarity search."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize content-based recommender."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.faiss_config = self.config['models']['faiss']
        self.paths_config = self.config['paths']
        
        self.index = None
        self.item_embeddings = None
        self.user_profiles = None
        self.items_df = None
        
        os.makedirs(self.paths_config['models_dir'], exist_ok=True)
    
    def build_index(self, item_embeddings: np.ndarray):
        """Build FAISS index for item embeddings."""
        print("Building FAISS index...")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(item_embeddings)
        
        # Create index
        dimension = item_embeddings.shape[1]
        
        if self.faiss_config['index_type'] == "IndexFlatIP":
            # Inner Product index for cosine similarity (after normalization)
            self.index = faiss.IndexFlatIP(dimension)
        else:
            # Default to flat L2 index
            self.index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        self.index.add(item_embeddings.astype('float32'))
        self.item_embeddings = item_embeddings
        
        print(f"Index built with {self.index.ntotal} items")
    
    def set_user_profiles(self, user_profiles: np.ndarray):
        """Set user profiles for personalized recommendations."""
        self.user_profiles = user_profiles
    
    def set_items_data(self, items_df: pd.DataFrame):
        """Set items dataframe for metadata."""
        self.items_df = items_df
    
    def get_similar_items(self, item_id: int, k: int = 10) -> List[Tuple[int, float]]:
        """Get k most similar items to a given item."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Get item embedding
        item_embedding = self.item_embeddings[item_id:item_id+1].astype('float32')
        
        # Search for similar items
        scores, indices = self.index.search(item_embedding, k + 1)  # +1 to exclude the item itself
        
        # Remove the item itself from results
        similar_items = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != item_id:  # Exclude the item itself
                similar_items.append((int(idx), float(score)))
        
        return similar_items[:k]
    
    def get_user_recommendations(self, user_id: int, k: int = 10, 
                               exclude_seen: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """Get personalized recommendations for a user."""
        if self.user_profiles is None:
            raise ValueError("User profiles not set. Call set_user_profiles first.")
        
        # Get user profile
        user_profile = self.user_profiles[user_id:user_id+1].astype('float32')
        
        # Normalize user profile
        faiss.normalize_L2(user_profile)
        
        # Search for relevant items
        scores, indices = self.index.search(user_profile, k * 2)  # Get more to filter out seen items
        
        # Filter out seen items
        recommendations = []
        exclude_seen = exclude_seen or []
        
        for score, idx in zip(scores[0], indices[0]):
            if int(idx) not in exclude_seen and len(recommendations) < k:
                recommendations.append((int(idx), float(score)))
        
        return recommendations
    
    def get_category_based_recommendations(self, user_id: int, category: str, k: int = 10) -> List[Tuple[int, float]]:
        """Get recommendations within a specific category."""
        if self.items_df is None:
            raise ValueError("Items data not set. Call set_items_data first.")
        
        # Filter items by category
        category_items = self.items_df[self.items_df['category'] == category]
        category_indices = category_items.index.tolist()
        
        if not category_indices:
            return []
        
        # Get user profile
        user_profile = self.user_profiles[user_id:user_id+1].astype('float32')
        faiss.normalize_L2(user_profile)
        
        # Calculate similarities for category items only
        category_embeddings = self.item_embeddings[category_indices].astype('float32')
        similarities = np.dot(user_profile, category_embeddings.T)[0]
        
        # Get top-k items
        top_indices = np.argsort(similarities)[::-1][:k]
        recommendations = [(category_indices[i], similarities[i]) for i in top_indices]
        
        return recommendations
    
    def get_diverse_recommendations(self, user_id: int, k: int = 10, 
                                  diversity_weight: float = 0.3) -> List[Tuple[int, float]]:
        """Get diverse recommendations using Maximum Marginal Relevance."""
        if self.user_profiles is None or self.items_df is None:
            raise ValueError("User profiles and items data must be set.")
        
        # Get initial candidate set
        candidates = self.get_user_recommendations(user_id, k * 3)  # Get more candidates
        
        if not candidates:
            return []
        
        # Initialize with the top item
        selected = [candidates[0]]
        remaining = candidates[1:]
        
        # Select diverse items using MMR
        while len(selected) < k and remaining:
            mmr_scores = []
            
            for item_id, relevance_score in remaining:
                # Calculate diversity (minimum similarity to selected items)
                diversity_score = 1.0
                for selected_id, _ in selected:
                    sim = cosine_similarity(
                        self.item_embeddings[item_id:item_id+1],
                        self.item_embeddings[selected_id:selected_id+1]
                    )[0][0]
                    diversity_score = min(diversity_score, 1 - sim)
                
                # MMR score
                mmr_score = (1 - diversity_weight) * relevance_score + diversity_weight * diversity_score
                mmr_scores.append((item_id, mmr_score))
            
            # Select item with highest MMR score
            best_item = max(mmr_scores, key=lambda x: x[1])
            selected.append((best_item[0], best_item[1]))
            
            # Remove selected item from remaining
            remaining = [(item_id, score) for item_id, score in remaining if item_id != best_item[0]]
        
        return selected
    
    def get_trending_recommendations(self, k: int = 10, time_window_days: int = 7) -> List[Tuple[int, float]]:
        """Get trending items based on recent popularity."""
        if self.items_df is None:
            raise ValueError("Items data not set.")
        
        # For demonstration, use recency as a proxy for trending
        # In a real system, this would use actual interaction data
        items_with_recency = self.items_df.copy()
        items_with_recency['created_date'] = pd.to_datetime(items_with_recency['created_date'])
        items_with_recency['days_old'] = (pd.Timestamp.now() - items_with_recency['created_date']).dt.days
        
        # Score based on recency (newer items get higher scores)
        max_days = items_with_recency['days_old'].max()
        items_with_recency['trending_score'] = 1 - (items_with_recency['days_old'] / max_days)
        
        # Get top trending items
        trending_items = items_with_recency.nlargest(k, 'trending_score')
        
        recommendations = [(int(idx), score) for idx, score in 
                          zip(trending_items.index, trending_items['trending_score'])]
        
        return recommendations
    
    def save_model(self):
        """Save the FAISS index and model components."""
        print("Saving content-based model...")
        
        # Save FAISS index
        faiss.write_index(self.index, f"{self.paths_config['models_dir']}/content_faiss.index")
        
        # Save other components
        with open(f"{self.paths_config['models_dir']}/content_model.pkl", 'wb') as f:
            pickle.dump({
                'item_embeddings': self.item_embeddings,
                'user_profiles': self.user_profiles,
                'config': self.faiss_config
            }, f)
        
        print(f"Model saved to {self.paths_config['models_dir']}/")
    
    def load_model(self):
        """Load the FAISS index and model components."""
        print("Loading content-based model...")
        
        # Load FAISS index
        self.index = faiss.read_index(f"{self.paths_config['models_dir']}/content_faiss.index")
        
        # Load other components
        with open(f"{self.paths_config['models_dir']}/content_model.pkl", 'rb') as f:
            data = pickle.load(f)
            self.item_embeddings = data['item_embeddings']
            self.user_profiles = data['user_profiles']
        
        print("Model loaded successfully")
    
    def explain_recommendation(self, user_id: int, item_id: int) -> dict:
        """Explain why an item was recommended to a user."""
        if self.user_profiles is None or self.items_df is None:
            return {"error": "Model not properly initialized"}
        
        # Calculate similarity
        user_profile = self.user_profiles[user_id:user_id+1]
        item_embedding = self.item_embeddings[item_id:item_id+1]
        similarity = cosine_similarity(user_profile, item_embedding)[0][0]
        
        # Get item details
        item_info = self.items_df.iloc[item_id]
        
        explanation = {
            "similarity_score": similarity,
            "item_title": item_info['title'],
            "item_category": item_info['category'],
            "item_type": item_info['content_type'],
            "reason": f"This {item_info['content_type']} about {item_info['category']} "
                     f"matches your interests with a similarity score of {similarity:.3f}"
        }
        
        return explanation


def main():
    """Main function to build and test content-based recommender."""
    from src.data.data_loader import DataLoader
    from src.preprocessing.feature_engineering import ContentPreprocessor
    
    # Load data
    loader = DataLoader()
    users_df, items_df, interactions_df = loader.load_raw_data()
    
    # Load preprocessed features
    preprocessor = ContentPreprocessor()
    content_embeddings, item_features, user_profiles, _ = preprocessor.load_features()
    
    # Build content-based recommender
    recommender = ContentBasedRecommender()
    recommender.build_index(content_embeddings)
    recommender.set_user_profiles(user_profiles)
    recommender.set_items_data(items_df)
    
    # Test recommendations
    user_id = 0
    recommendations = recommender.get_user_recommendations(user_id, k=5)
    
    print(f"Recommendations for user {user_id}:")
    for item_id, score in recommendations:
        item_info = items_df.iloc[item_id]
        print(f"- {item_info['title']} (Score: {score:.3f})")
    
    # Save model
    recommender.save_model()


if __name__ == "__main__":
    main()