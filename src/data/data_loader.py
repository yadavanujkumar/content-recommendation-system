"""
Data loading utilities for the content recommendation system.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import os


class DataLoader:
    """Load and prepare data for recommendation models."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """Initialize data loader."""
        self.data_dir = data_dir
    
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load raw data from CSV files."""
        users_df = pd.read_csv(os.path.join(self.data_dir, "users.csv"))
        items_df = pd.read_csv(os.path.join(self.data_dir, "items.csv"))
        interactions_df = pd.read_csv(os.path.join(self.data_dir, "interactions.csv"))
        
        # Convert string lists back to actual lists for users preferred_categories
        users_df['preferred_categories'] = users_df['preferred_categories'].apply(eval)
        items_df['tags'] = items_df['tags'].apply(eval)
        
        return users_df, items_df, interactions_df
    
    def create_interaction_matrix(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Create user-item interaction matrix."""
        # Use rating as the interaction strength
        interaction_matrix = interactions_df.pivot_table(
            index='user_id',
            columns='item_id',
            values='rating',
            fill_value=0,
            aggfunc='mean'  # Average rating if multiple interactions
        )
        
        return interaction_matrix
    
    def split_data(self, interactions_df: pd.DataFrame, test_ratio: float = 0.2, 
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split interactions into train and test sets."""
        # Sort by timestamp for temporal split
        interactions_df = interactions_df.sort_values('timestamp')
        
        # Split by user to ensure each user has both train and test data
        train_data = []
        test_data = []
        
        np.random.seed(random_state)
        
        for user_id in interactions_df['user_id'].unique():
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]
            
            if len(user_interactions) < 2:
                # If user has only one interaction, put it in training
                train_data.append(user_interactions)
                continue
            
            # Split user's interactions
            n_test = max(1, int(len(user_interactions) * test_ratio))
            test_interactions = user_interactions.tail(n_test)
            train_interactions = user_interactions.head(len(user_interactions) - n_test)
            
            train_data.append(train_interactions)
            test_data.append(test_interactions)
        
        train_df = pd.concat(train_data, ignore_index=True)
        test_df = pd.concat(test_data, ignore_index=True)
        
        return train_df, test_df
    
    def get_user_item_features(self, users_df: pd.DataFrame, items_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare user and item features for modeling."""
        # User features
        user_features = users_df.copy()
        
        # Encode categorical variables
        user_features = pd.get_dummies(user_features, columns=['gender', 'location', 'activity_level'])
        
        # Item features
        item_features = items_df.copy()
        
        # Handle missing values
        item_features['duration_minutes'] = item_features['duration_minutes'].fillna(0)
        item_features['word_count'] = item_features['word_count'].fillna(0)
        
        # Encode categorical variables
        item_features = pd.get_dummies(item_features, columns=['category', 'content_type'])
        
        # Convert date to features
        item_features['created_date'] = pd.to_datetime(item_features['created_date'])
        item_features['days_since_creation'] = (pd.Timestamp.now() - item_features['created_date']).dt.days
        item_features = item_features.drop('created_date', axis=1)
        
        return user_features, item_features