"""
Evaluation metrics for recommendation systems.
Implements Precision@K, Recall@K, NDCG@K, and other relevant metrics.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set
import yaml
from sklearn.metrics import roc_auc_score, mean_squared_error
import math


class RecommendationEvaluator:
    """Evaluate recommendation system performance."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize evaluator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.eval_config = self.config['evaluation']
        self.k_values = self.eval_config['k_values']
    
    def precision_at_k(self, recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
        """Calculate Precision@K."""
        if k <= 0:
            return 0.0
        
        recommended_k = recommended_items[:k]
        if not recommended_k:
            return 0.0
        
        relevant_recommended = sum(1 for item in recommended_k if item in relevant_items)
        return relevant_recommended / len(recommended_k)
    
    def recall_at_k(self, recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
        """Calculate Recall@K."""
        if not relevant_items or k <= 0:
            return 0.0
        
        recommended_k = recommended_items[:k]
        relevant_recommended = sum(1 for item in recommended_k if item in relevant_items)
        return relevant_recommended / len(relevant_items)
    
    def f1_at_k(self, recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
        """Calculate F1@K."""
        precision = self.precision_at_k(recommended_items, relevant_items, k)
        recall = self.recall_at_k(recommended_items, relevant_items, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def ndcg_at_k(self, recommended_items: List[int], relevant_items: Dict[int, float], k: int) -> float:
        """Calculate NDCG@K."""
        if k <= 0:
            return 0.0
        
        recommended_k = recommended_items[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(recommended_k):
            relevance = relevant_items.get(item, 0.0)
            if relevance > 0:
                dcg += relevance / math.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG
        sorted_relevances = sorted(relevant_items.values(), reverse=True)[:k]
        idcg = 0.0
        for i, relevance in enumerate(sorted_relevances):
            if relevance > 0:
                idcg += relevance / math.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def map_at_k(self, recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
        """Calculate Mean Average Precision@K."""
        if not relevant_items or k <= 0:
            return 0.0
        
        recommended_k = recommended_items[:k]
        
        score = 0.0
        num_hits = 0
        
        for i, item in enumerate(recommended_k):
            if item in relevant_items:
                num_hits += 1
                score += num_hits / (i + 1)
        
        return score / len(relevant_items)
    
    def mrr_at_k(self, recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
        """Calculate Mean Reciprocal Rank@K."""
        if not relevant_items or k <= 0:
            return 0.0
        
        recommended_k = recommended_items[:k]
        
        for i, item in enumerate(recommended_k):
            if item in relevant_items:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def hit_rate_at_k(self, recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
        """Calculate Hit Rate@K (binary relevance)."""
        if not relevant_items or k <= 0:
            return 0.0
        
        recommended_k = recommended_items[:k]
        return 1.0 if any(item in relevant_items for item in recommended_k) else 0.0
    
    def coverage(self, all_recommendations: List[List[int]], total_items: int) -> float:
        """Calculate item coverage - fraction of items that appear in recommendations."""
        recommended_items = set()
        for recs in all_recommendations:
            recommended_items.update(recs)
        
        return len(recommended_items) / total_items if total_items > 0 else 0.0
    
    def diversity(self, recommendations: List[int], item_features: np.ndarray) -> float:
        """Calculate intra-list diversity based on item features."""
        if len(recommendations) < 2:
            return 0.0
        
        similarities = []
        for i in range(len(recommendations)):
            for j in range(i + 1, len(recommendations)):
                item1_features = item_features[recommendations[i]]
                item2_features = item_features[recommendations[j]]
                
                # Cosine similarity
                norm1 = np.linalg.norm(item1_features)
                norm2 = np.linalg.norm(item2_features)
                if norm1 > 0 and norm2 > 0:
                    similarity = np.dot(item1_features, item2_features) / (norm1 * norm2)
                    similarities.append(similarity)
        
        return 1 - np.mean(similarities) if similarities else 0.0
    
    def novelty(self, recommendations: List[int], item_popularity: Dict[int, float]) -> float:
        """Calculate novelty based on item popularity."""
        if not recommendations:
            return 0.0
        
        novelty_scores = []
        for item in recommendations:
            popularity = item_popularity.get(item, 0.0)
            # Novelty is inversely related to popularity
            novelty_scores.append(-math.log2(popularity + 1e-10))
        
        return np.mean(novelty_scores)
    
    def evaluate_user(self, user_id: int, recommended_items: List[int], 
                     test_interactions: pd.DataFrame) -> Dict[str, Dict[int, float]]:
        """Evaluate recommendations for a single user."""
        # Get relevant items for this user from test set
        user_test = test_interactions[test_interactions['user_id'] == user_id]
        
        # For binary relevance metrics
        relevant_items_binary = set(user_test['item_id'].tolist())
        
        # For graded relevance metrics (using ratings)
        relevant_items_graded = dict(zip(user_test['item_id'], user_test['rating']))
        
        results = {}
        
        # Calculate metrics for each k
        for k in self.k_values:
            results[k] = {
                'precision': self.precision_at_k(recommended_items, relevant_items_binary, k),
                'recall': self.recall_at_k(recommended_items, relevant_items_binary, k),
                'f1': self.f1_at_k(recommended_items, relevant_items_binary, k),
                'ndcg': self.ndcg_at_k(recommended_items, relevant_items_graded, k),
                'map': self.map_at_k(recommended_items, relevant_items_binary, k),
                'mrr': self.mrr_at_k(recommended_items, relevant_items_binary, k),
                'hit_rate': self.hit_rate_at_k(recommended_items, relevant_items_binary, k)
            }
        
        return results
    
    def evaluate_system(self, all_recommendations: Dict[int, List[int]], 
                       test_interactions: pd.DataFrame,
                       item_features: Optional[np.ndarray] = None,
                       item_popularity: Optional[Dict[int, float]] = None) -> Dict:
        """Evaluate the entire recommendation system."""
        user_results = {}
        
        # Evaluate each user
        for user_id, recommendations in all_recommendations.items():
            user_results[user_id] = self.evaluate_user(user_id, recommendations, test_interactions)
        
        # Aggregate results across users
        aggregated_results = {}
        
        for k in self.k_values:
            aggregated_results[k] = {}
            
            metrics = ['precision', 'recall', 'f1', 'ndcg', 'map', 'mrr', 'hit_rate']
            
            for metric in metrics:
                values = [user_results[user_id][k][metric] for user_id in user_results 
                         if user_id in user_results and k in user_results[user_id]]
                
                if values:
                    aggregated_results[k][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'median': np.median(values)
                    }
                else:
                    aggregated_results[k][metric] = {'mean': 0.0, 'std': 0.0, 'median': 0.0}
        
        # Calculate system-level metrics
        system_metrics = {}
        
        # Coverage
        if test_interactions is not None:
            total_items = test_interactions['item_id'].nunique()
            all_rec_lists = list(all_recommendations.values())
            system_metrics['coverage'] = self.coverage(all_rec_lists, total_items)
        
        # Average diversity
        if item_features is not None:
            diversity_scores = []
            for recommendations in all_recommendations.values():
                if len(recommendations) > 1:
                    diversity_scores.append(self.diversity(recommendations, item_features))
            system_metrics['diversity'] = np.mean(diversity_scores) if diversity_scores else 0.0
        
        # Average novelty
        if item_popularity is not None:
            novelty_scores = []
            for recommendations in all_recommendations.values():
                if recommendations:
                    novelty_scores.append(self.novelty(recommendations, item_popularity))
            system_metrics['novelty'] = np.mean(novelty_scores) if novelty_scores else 0.0
        
        return {
            'user_results': user_results,
            'aggregated_results': aggregated_results,
            'system_metrics': system_metrics
        }
    
    def compare_models(self, model_results: Dict[str, Dict]) -> pd.DataFrame:
        """Compare multiple models' performance."""
        comparison_data = []
        
        for model_name, results in model_results.items():
            for k in self.k_values:
                row = {'model': model_name, 'k': k}
                
                if k in results['aggregated_results']:
                    for metric in ['precision', 'recall', 'f1', 'ndcg', 'map', 'mrr', 'hit_rate']:
                        row[f'{metric}_mean'] = results['aggregated_results'][k][metric]['mean']
                        row[f'{metric}_std'] = results['aggregated_results'][k][metric]['std']
                
                # Add system metrics
                for metric, value in results['system_metrics'].items():
                    row[metric] = value
                
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def print_evaluation_summary(self, results: Dict, model_name: str = "Model"):
        """Print a summary of evaluation results."""
        print(f"\n{model_name} Evaluation Results")
        print("=" * 50)
        
        for k in self.k_values:
            if k in results['aggregated_results']:
                print(f"\nMetrics@{k}:")
                k_results = results['aggregated_results'][k]
                
                for metric in ['precision', 'recall', 'f1', 'ndcg', 'map', 'mrr', 'hit_rate']:
                    mean_val = k_results[metric]['mean']
                    std_val = k_results[metric]['std']
                    print(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
        
        print(f"\nSystem Metrics:")
        for metric, value in results['system_metrics'].items():
            print(f"  {metric.capitalize()}: {value:.4f}")


def main():
    """Main function to demonstrate evaluation."""
    # This would typically be called from the training/evaluation pipeline
    
    # Example usage (would be replaced with actual model outputs)
    test_recommendations = {
        0: [1, 5, 10, 15, 20],  # User 0's recommendations
        1: [2, 6, 11, 16, 21],  # User 1's recommendations
    }
    
    # Mock test interactions
    test_data = pd.DataFrame({
        'user_id': [0, 0, 1, 1],
        'item_id': [1, 3, 2, 8],
        'rating': [4.5, 3.0, 5.0, 4.0]
    })
    
    evaluator = RecommendationEvaluator()
    results = evaluator.evaluate_system(test_recommendations, test_data)
    evaluator.print_evaluation_summary(results, "Example Model")


if __name__ == "__main__":
    main()