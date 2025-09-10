"""
Hybrid recommendation system combining content-based and collaborative filtering approaches.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import yaml
from src.recommendation.content_based import ContentBasedRecommender
from src.recommendation.collaborative_filtering import ALSRecommender, NeuralCFRecommender


class HybridRecommender:
    """Hybrid recommender combining multiple approaches with re-ranking."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize hybrid recommender."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.paths_config = self.config['paths']
        
        # Initialize component recommenders
        self.content_recommender = ContentBasedRecommender(config_path)
        self.als_recommender = ALSRecommender(config_path)
        self.ncf_recommender = NeuralCFRecommender(config_path)
        
        # Weights for different approaches
        self.weights = {
            'content': 0.4,
            'als': 0.3,
            'ncf': 0.3
        }
        
        self.items_df = None
        self.users_df = None
        self.interactions_df = None
    
    def set_data(self, users_df: pd.DataFrame, items_df: pd.DataFrame, interactions_df: pd.DataFrame):
        """Set data for the recommender."""
        self.users_df = users_df
        self.items_df = items_df
        self.interactions_df = interactions_df
        
        # Set data for component recommenders
        self.content_recommender.set_items_data(items_df)
    
    def load_models(self):
        """Load all component models."""
        print("Loading component models...")
        
        try:
            self.content_recommender.load_model()
        except Exception as e:
            print(f"Warning: Could not load content model: {e}")
        
        try:
            self.als_recommender.load_model()
        except Exception as e:
            print(f"Warning: Could not load ALS model: {e}")
        
        try:
            self.ncf_recommender.load_model()
        except Exception as e:
            print(f"Warning: Could not load Neural CF model: {e}")
    
    def get_recommendations_from_all_models(self, user_id: int, k: int = 20) -> Dict[str, List[Tuple[int, float]]]:
        """Get recommendations from all component models."""
        recommendations = {}
        
        # Get user's seen items
        seen_items = self.interactions_df[self.interactions_df['user_id'] == user_id]['item_id'].tolist()
        
        # Content-based recommendations
        try:
            content_recs = self.content_recommender.get_user_recommendations(
                user_id, k=k, exclude_seen=seen_items
            )
            recommendations['content'] = content_recs
        except Exception as e:
            print(f"Content-based recommendations failed: {e}")
            recommendations['content'] = []
        
        # ALS recommendations
        try:
            als_recs = self.als_recommender.get_user_recommendations(
                user_id, k=k, filter_already_liked=True
            )
            recommendations['als'] = als_recs
        except Exception as e:
            print(f"ALS recommendations failed: {e}")
            recommendations['als'] = []
        
        # Neural CF recommendations
        try:
            ncf_recs = self.ncf_recommender.get_user_recommendations(
                user_id, k=k, exclude_seen=seen_items
            )
            recommendations['ncf'] = ncf_recs
        except Exception as e:
            print(f"Neural CF recommendations failed: {e}")
            recommendations['ncf'] = []
        
        return recommendations
    
    def weighted_fusion(self, recommendations: Dict[str, List[Tuple[int, float]]], k: int = 10) -> List[Tuple[int, float]]:
        """Combine recommendations using weighted fusion."""
        item_scores = {}
        
        # Aggregate scores from all models
        for model_name, recs in recommendations.items():
            weight = self.weights.get(model_name, 0)
            
            for item_id, score in recs:
                if item_id not in item_scores:
                    item_scores[item_id] = 0
                item_scores[item_id] += weight * score
        
        # Sort by combined score
        combined_recs = [(item_id, score) for item_id, score in item_scores.items()]
        combined_recs.sort(key=lambda x: x[1], reverse=True)
        
        return combined_recs[:k]
    
    def rank_fusion(self, recommendations: Dict[str, List[Tuple[int, float]]], k: int = 10) -> List[Tuple[int, float]]:
        """Combine recommendations using rank fusion (Borda count)."""
        item_ranks = {}
        
        # Get ranks from each model
        for model_name, recs in recommendations.items():
            weight = self.weights.get(model_name, 0)
            
            for rank, (item_id, score) in enumerate(recs):
                if item_id not in item_ranks:
                    item_ranks[item_id] = 0
                # Lower rank is better, so we subtract from max_rank
                item_ranks[item_id] += weight * (len(recs) - rank)
        
        # Sort by combined rank
        combined_recs = [(item_id, rank) for item_id, rank in item_ranks.items()]
        combined_recs.sort(key=lambda x: x[1], reverse=True)
        
        return combined_recs[:k]
    
    def diversity_reranking(self, recommendations: List[Tuple[int, float]], 
                          diversity_weight: float = 0.3, k: int = 10) -> List[Tuple[int, float]]:
        """Re-rank recommendations to increase diversity."""
        if not recommendations or not self.items_df is not None:
            return recommendations
        
        # Initialize with top recommendation
        selected = [recommendations[0]]
        remaining = recommendations[1:]
        
        while len(selected) < k and remaining:
            mmr_scores = []
            
            for item_id, relevance_score in remaining:
                # Calculate diversity based on category and content type
                diversity_score = 1.0
                item_category = self.items_df.iloc[item_id]['category']
                item_type = self.items_df.iloc[item_id]['content_type']
                
                for selected_id, _ in selected:
                    selected_category = self.items_df.iloc[selected_id]['category']
                    selected_type = self.items_df.iloc[selected_id]['content_type']
                    
                    # Penalize same category and type
                    if item_category == selected_category:
                        diversity_score *= 0.7
                    if item_type == selected_type:
                        diversity_score *= 0.8
                
                # MMR score
                mmr_score = (1 - diversity_weight) * relevance_score + diversity_weight * diversity_score
                mmr_scores.append((item_id, mmr_score))
            
            # Select best item
            best_item = max(mmr_scores, key=lambda x: x[1])
            selected.append(best_item)
            
            # Remove from remaining
            remaining = [(item_id, score) for item_id, score in remaining if item_id != best_item[0]]
        
        return selected
    
    def get_hybrid_recommendations(self, user_id: int, k: int = 10, 
                                 fusion_method: str = 'weighted',
                                 diversity_weight: float = 0.2) -> List[Tuple[int, float]]:
        """Get hybrid recommendations combining all approaches."""
        # Get recommendations from all models
        all_recommendations = self.get_recommendations_from_all_models(user_id, k=k*2)
        
        # Combine using specified fusion method
        if fusion_method == 'weighted':
            combined_recs = self.weighted_fusion(all_recommendations, k=k*2)
        elif fusion_method == 'rank':
            combined_recs = self.rank_fusion(all_recommendations, k=k*2)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Apply diversity re-ranking
        if diversity_weight > 0:
            final_recs = self.diversity_reranking(combined_recs, diversity_weight, k)
        else:
            final_recs = combined_recs[:k]
        
        return final_recs
    
    def get_cold_start_recommendations(self, user_id: int, k: int = 10) -> List[Tuple[int, float]]:
        """Get recommendations for cold start users (no interaction history)."""
        # For cold start, rely more on content-based and trending items
        recommendations = {}
        
        # Get trending items
        try:
            trending_recs = self.content_recommender.get_trending_recommendations(k=k)
            recommendations['trending'] = trending_recs
        except Exception as e:
            print(f"Trending recommendations failed: {e}")
            recommendations['trending'] = []
        
        # Get popular items by category
        try:
            # Get most popular items overall (highest average rating)
            popular_items = self.interactions_df.groupby('item_id')['rating'].agg(['mean', 'count']).reset_index()
            popular_items = popular_items[popular_items['count'] >= 5]  # At least 5 ratings
            popular_items = popular_items.nlargest(k, 'mean')
            
            popular_recs = [(int(item_id), score) for item_id, score in 
                           zip(popular_items['item_id'], popular_items['mean'])]
            recommendations['popular'] = popular_recs
        except Exception as e:
            print(f"Popular recommendations failed: {e}")
            recommendations['popular'] = []
        
        # Combine with equal weights
        cold_start_weights = {'trending': 0.6, 'popular': 0.4}
        
        item_scores = {}
        for model_name, recs in recommendations.items():
            weight = cold_start_weights.get(model_name, 0)
            
            for item_id, score in recs:
                if item_id not in item_scores:
                    item_scores[item_id] = 0
                item_scores[item_id] += weight * score
        
        combined_recs = [(item_id, score) for item_id, score in item_scores.items()]
        combined_recs.sort(key=lambda x: x[1], reverse=True)
        
        return combined_recs[:k]
    
    def get_category_recommendations(self, user_id: int, category: str, k: int = 10) -> List[Tuple[int, float]]:
        """Get recommendations within a specific category."""
        # Filter items by category
        category_items = self.items_df[self.items_df['category'] == category]
        category_item_ids = category_items.index.tolist()
        
        if not category_item_ids:
            return []
        
        # Get hybrid recommendations and filter by category
        all_recs = self.get_hybrid_recommendations(user_id, k=k*3)
        category_recs = [(item_id, score) for item_id, score in all_recs 
                        if item_id in category_item_ids]
        
        return category_recs[:k]
    
    def explain_recommendation(self, user_id: int, item_id: int) -> Dict:
        """Explain why an item was recommended."""
        explanation = {
            'item_id': item_id,
            'user_id': user_id,
            'models_used': [],
            'scores': {},
            'item_info': {}
        }
        
        # Get item information
        if self.items_df is not None:
            item_info = self.items_df.iloc[item_id]
            explanation['item_info'] = {
                'title': item_info['title'],
                'category': item_info['category'],
                'content_type': item_info['content_type']
            }
        
        # Get explanations from each model
        try:
            content_explanation = self.content_recommender.explain_recommendation(user_id, item_id)
            explanation['models_used'].append('content-based')
            explanation['scores']['content'] = content_explanation.get('similarity_score', 0)
        except:
            pass
        
        # Note: ALS and Neural CF explanations would require additional implementation
        
        return explanation
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update model weights for fusion."""
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in new_weights.items()}
        else:
            raise ValueError("Total weight must be greater than 0")


def main():
    """Main function to test hybrid recommender."""
    from src.data.data_loader import DataLoader
    from src.preprocessing.feature_engineering import ContentPreprocessor
    
    # Load data
    loader = DataLoader()
    users_df, items_df, interactions_df = loader.load_raw_data()
    
    # Initialize hybrid recommender
    hybrid_recommender = HybridRecommender()
    hybrid_recommender.set_data(users_df, items_df, interactions_df)
    hybrid_recommender.load_models()
    
    # Test recommendations
    user_id = 0
    
    print(f"Hybrid recommendations for user {user_id}:")
    recommendations = hybrid_recommender.get_hybrid_recommendations(user_id, k=5)
    
    for item_id, score in recommendations:
        item_info = items_df.iloc[item_id]
        print(f"- {item_info['title']} (Score: {score:.3f})")
    
    # Test cold start recommendations
    print(f"\nCold start recommendations:")
    cold_start_recs = hybrid_recommender.get_cold_start_recommendations(user_id, k=5)
    
    for item_id, score in cold_start_recs:
        item_info = items_df.iloc[item_id]
        print(f"- {item_info['title']} (Score: {score:.3f})")


if __name__ == "__main__":
    main()