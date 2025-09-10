"""
Collaborative filtering recommendation system using ALS and Neural Collaborative Filtering.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import TruncatedSVD
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import train_test_split as implicit_train_test_split
import scipy.sparse as sp
from typing import List, Tuple, Optional
import pickle
import os
import yaml
from tqdm import tqdm


class ALSRecommender:
    """Alternating Least Squares collaborative filtering recommender."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize ALS recommender."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.als_config = self.config['models']['collaborative']['als']
        self.paths_config = self.config['paths']
        
        self.model = AlternatingLeastSquares(
            factors=self.als_config['factors'],
            iterations=self.als_config['iterations'],
            regularization=self.als_config['regularization'],
            random_state=42
        )
        
        self.user_item_matrix = None
        self.item_user_matrix = None
        
    def prepare_data(self, interactions_df: pd.DataFrame) -> sp.csr_matrix:
        """Prepare interaction data for ALS."""
        # Create user-item matrix
        user_item_df = interactions_df.pivot_table(
            index='user_id',
            columns='item_id', 
            values='rating',
            fill_value=0
        )
        
        # Convert to sparse matrix
        self.user_item_matrix = sp.csr_matrix(user_item_df.values)
        self.item_user_matrix = self.user_item_matrix.T.tocsr()
        
        return self.user_item_matrix
    
    def train(self, interactions_df: pd.DataFrame):
        """Train the ALS model."""
        print("Training ALS model...")
        
        # Prepare data
        user_item_matrix = self.prepare_data(interactions_df)
        
        # Train model
        self.model.fit(user_item_matrix, show_progress=True)
        
        print("ALS training completed")
    
    def get_user_recommendations(self, user_id: int, k: int = 10, 
                               filter_already_liked: bool = True) -> List[Tuple[int, float]]:
        """Get recommendations for a user."""
        if self.user_item_matrix is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get recommendations
        item_ids, scores = self.model.recommend(
            user_id, 
            self.user_item_matrix[user_id],
            N=k,
            filter_already_liked_items=filter_already_liked
        )
        
        return list(zip(item_ids.tolist(), scores.tolist()))
    
    def get_similar_items(self, item_id: int, k: int = 10) -> List[Tuple[int, float]]:
        """Get similar items."""
        if self.item_user_matrix is None:
            raise ValueError("Model not trained. Call train() first.")
        
        item_ids, scores = self.model.similar_items(item_id, N=k)
        return list(zip(item_ids.tolist(), scores.tolist()))
    
    def get_similar_users(self, user_id: int, k: int = 10) -> List[Tuple[int, float]]:
        """Get similar users."""
        if self.user_item_matrix is None:
            raise ValueError("Model not trained. Call train() first.")
        
        user_ids, scores = self.model.similar_users(user_id, N=k)
        return list(zip(user_ids.tolist(), scores.tolist()))
    
    def save_model(self):
        """Save the ALS model."""
        model_path = f"{self.paths_config['models_dir']}/als_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'user_item_matrix': self.user_item_matrix,
                'item_user_matrix': self.item_user_matrix
            }, f)
        print(f"ALS model saved to {model_path}")
    
    def load_model(self):
        """Load the ALS model."""
        model_path = f"{self.paths_config['models_dir']}/als_model.pkl"
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.user_item_matrix = data['user_item_matrix']
            self.item_user_matrix = data['item_user_matrix']
        print(f"ALS model loaded from {model_path}")


class NeuralCF(nn.Module):
    """Neural Collaborative Filtering model."""
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, 
                 hidden_layers: List[int] = [128, 64, 32], dropout: float = 0.2):
        """Initialize Neural CF model."""
        super(NeuralCF, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        layers = []
        input_dim = embedding_dim * 2
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Concatenate embeddings
        x = torch.cat([user_emb, item_emb], dim=1)
        
        # Pass through MLP
        output = self.mlp(x)
        
        return output.squeeze()


class NeuralCFRecommender:
    """Neural Collaborative Filtering recommender."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize Neural CF recommender."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.ncf_config = self.config['models']['collaborative']['neural_cf']
        self.paths_config = self.config['paths']
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_users = None
        self.num_items = None
        
    def prepare_data(self, interactions_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare data for Neural CF training."""
        # Get user and item counts
        self.num_users = interactions_df['user_id'].nunique()
        self.num_items = interactions_df['item_id'].nunique()
        
        # Normalize ratings to [0, 1]
        min_rating = interactions_df['rating'].min()
        max_rating = interactions_df['rating'].max()
        interactions_df = interactions_df.copy()
        interactions_df['rating_normalized'] = (interactions_df['rating'] - min_rating) / (max_rating - min_rating)
        
        # Convert to tensors
        user_ids = torch.LongTensor(interactions_df['user_id'].values)
        item_ids = torch.LongTensor(interactions_df['item_id'].values)
        ratings = torch.FloatTensor(interactions_df['rating_normalized'].values)
        
        return user_ids, item_ids, ratings
    
    def train(self, interactions_df: pd.DataFrame):
        """Train the Neural CF model."""
        print("Training Neural CF model...")
        
        # Prepare data
        user_ids, item_ids, ratings = self.prepare_data(interactions_df)
        
        # Initialize model
        self.model = NeuralCF(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.ncf_config['embedding_dim'],
            hidden_layers=self.ncf_config['hidden_layers'],
            dropout=self.ncf_config['dropout']
        ).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(user_ids, item_ids, ratings)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.ncf_config['batch_size'], 
            shuffle=True
        )
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.ncf_config['learning_rate'])
        
        # Training loop
        self.model.train()
        for epoch in range(self.ncf_config['epochs']):
            total_loss = 0
            
            for batch_users, batch_items, batch_ratings in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                batch_users = batch_users.to(self.device)
                batch_items = batch_items.to(self.device)
                batch_ratings = batch_ratings.to(self.device)
                
                # Forward pass
                predictions = self.model(batch_users, batch_items)
                loss = criterion(predictions, batch_ratings)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.ncf_config['epochs']}, Loss: {avg_loss:.4f}")
        
        print("Neural CF training completed")
    
    def predict(self, user_id: int, item_ids: List[int]) -> List[float]:
        """Predict ratings for user-item pairs."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.eval()
        
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id] * len(item_ids)).to(self.device)
            item_tensor = torch.LongTensor(item_ids).to(self.device)
            
            predictions = self.model(user_tensor, item_tensor)
            return predictions.cpu().numpy().tolist()
    
    def get_user_recommendations(self, user_id: int, k: int = 10, 
                               exclude_seen: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """Get recommendations for a user."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        exclude_seen = exclude_seen or []
        
        # Get all item IDs
        all_item_ids = list(range(self.num_items))
        candidate_items = [item_id for item_id in all_item_ids if item_id not in exclude_seen]
        
        # Predict ratings
        predictions = self.predict(user_id, candidate_items)
        
        # Get top-k recommendations
        item_scores = list(zip(candidate_items, predictions))
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        return item_scores[:k]
    
    def save_model(self):
        """Save the Neural CF model."""
        model_path = f"{self.paths_config['models_dir']}/neural_cf_model.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_users': self.num_users,
            'num_items': self.num_items,
            'config': self.ncf_config
        }, model_path)
        print(f"Neural CF model saved to {model_path}")
    
    def load_model(self):
        """Load the Neural CF model."""
        model_path = f"{self.paths_config['models_dir']}/neural_cf_model.pth"
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.num_users = checkpoint['num_users']
        self.num_items = checkpoint['num_items']
        
        self.model = NeuralCF(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=checkpoint['config']['embedding_dim'],
            hidden_layers=checkpoint['config']['hidden_layers'],
            dropout=checkpoint['config']['dropout']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Neural CF model loaded from {model_path}")


def main():
    """Main function to train collaborative filtering models."""
    from src.data.data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    users_df, items_df, interactions_df = loader.load_raw_data()
    train_df, test_df = loader.split_data(interactions_df)
    
    # Train ALS model
    print("Training ALS model...")
    als_recommender = ALSRecommender()
    als_recommender.train(train_df)
    als_recommender.save_model()
    
    # Test ALS recommendations
    user_id = 0
    recommendations = als_recommender.get_user_recommendations(user_id, k=5)
    print(f"\nALS recommendations for user {user_id}:")
    for item_id, score in recommendations:
        item_info = items_df.iloc[item_id]
        print(f"- {item_info['title']} (Score: {score:.3f})")
    
    # Train Neural CF model
    print("\nTraining Neural CF model...")
    ncf_recommender = NeuralCFRecommender()
    ncf_recommender.train(train_df)
    ncf_recommender.save_model()
    
    # Test Neural CF recommendations
    recommendations = ncf_recommender.get_user_recommendations(user_id, k=5)
    print(f"\nNeural CF recommendations for user {user_id}:")
    for item_id, score in recommendations:
        item_info = items_df.iloc[item_id]
        print(f"- {item_info['title']} (Score: {score:.3f})")


if __name__ == "__main__":
    main()