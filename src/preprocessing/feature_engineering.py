"""
Preprocessing module for content recommendation system.
Handles text embeddings, feature engineering, and normalization.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from typing import Tuple, List, Optional
import yaml


class ContentPreprocessor:
    """Preprocess content for recommendation system."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize preprocessor with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['models']['sentence_bert']
        self.paths_config = self.config['paths']
        
        # Initialize models
        self.sentence_model = SentenceTransformer(self.model_config['model_name'])
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        os.makedirs(self.paths_config['processed_data_dir'], exist_ok=True)
    
    def create_content_embeddings(self, items_df: pd.DataFrame) -> np.ndarray:
        """Create Sentence-BERT embeddings for content."""
        print("Creating content embeddings...")
        
        # Combine title and description for richer embeddings
        content_texts = []
        for _, item in items_df.iterrows():
            text = f"{item['title']} {item['description']}"
            if isinstance(item['tags'], list):
                text += " " + " ".join(item['tags'])
            content_texts.append(text)
        
        # Generate embeddings
        embeddings = self.sentence_model.encode(
            content_texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_tensor=False
        )
        
        return embeddings
    
    def create_categorical_features(self, items_df: pd.DataFrame) -> np.ndarray:
        """Create encoded categorical features."""
        print("Creating categorical features...")
        
        categorical_features = []
        
        # Encode categories
        for col in ['category', 'content_type', 'author']:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                encoded = self.label_encoders[col].fit_transform(items_df[col])
            else:
                encoded = self.label_encoders[col].transform(items_df[col])
            categorical_features.append(encoded.reshape(-1, 1))
        
        # Combine all categorical features
        categorical_matrix = np.hstack(categorical_features)
        
        return categorical_matrix
    
    def create_numerical_features(self, items_df: pd.DataFrame) -> np.ndarray:
        """Create and normalize numerical features."""
        print("Creating numerical features...")
        
        # Prepare numerical features
        numerical_cols = ['duration_minutes', 'word_count']
        numerical_data = items_df[numerical_cols].copy()
        
        # Fill missing values
        numerical_data['duration_minutes'] = numerical_data['duration_minutes'].fillna(0)
        numerical_data['word_count'] = numerical_data['word_count'].fillna(0)
        
        # Add derived features
        numerical_data['content_length_category'] = pd.cut(
            numerical_data['word_count'], 
            bins=[0, 500, 1500, 5000, float('inf')], 
            labels=['short', 'medium', 'long', 'very_long']
        )
        
        # Encode the new categorical feature
        if 'content_length_category' not in self.label_encoders:
            self.label_encoders['content_length_category'] = LabelEncoder()
        
        numerical_data['content_length_encoded'] = self.label_encoders['content_length_category'].fit_transform(
            numerical_data['content_length_category']
        )
        
        # Select final numerical features
        final_numerical = numerical_data[['duration_minutes', 'word_count', 'content_length_encoded']]
        
        # Normalize features
        normalized_features = self.scaler.fit_transform(final_numerical)
        
        return normalized_features
    
    def create_tag_features(self, items_df: pd.DataFrame) -> np.ndarray:
        """Create TF-IDF features from tags."""
        print("Creating tag features...")
        
        # Convert tags to text
        tag_texts = []
        for _, item in items_df.iterrows():
            if isinstance(item['tags'], list):
                tag_text = " ".join(item['tags'])
            else:
                tag_text = ""
            tag_texts.append(tag_text)
        
        # Create TF-IDF features
        tag_features = self.tfidf_vectorizer.fit_transform(tag_texts).toarray()
        
        return tag_features
    
    def create_user_profiles(self, users_df: pd.DataFrame, items_df: pd.DataFrame, 
                           interactions_df: pd.DataFrame, item_embeddings: np.ndarray) -> np.ndarray:
        """Create user profiles based on interaction history."""
        print("Creating user profiles...")
        
        user_profiles = []
        embedding_dim = item_embeddings.shape[1]
        
        for user_id in users_df['user_id']:
            # Get user's interactions
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]
            
            if len(user_interactions) == 0:
                # No interactions - use zero profile
                profile = np.zeros(embedding_dim)
            else:
                # Weight by rating and recency
                weights = user_interactions['rating'].values
                item_indices = user_interactions['item_id'].values
                
                # Get embeddings for interacted items
                interacted_embeddings = item_embeddings[item_indices]
                
                # Create weighted average profile
                profile = np.average(interacted_embeddings, axis=0, weights=weights)
            
            user_profiles.append(profile)
        
        return np.array(user_profiles)
    
    def process_all_features(self, users_df: pd.DataFrame, items_df: pd.DataFrame, 
                           interactions_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Process all features for the recommendation system."""
        print("Processing all features...")
        
        # Create item embeddings
        content_embeddings = self.create_content_embeddings(items_df)
        
        # Create categorical features
        categorical_features = self.create_categorical_features(items_df)
        
        # Create numerical features
        numerical_features = self.create_numerical_features(items_df)
        
        # Create tag features
        tag_features = self.create_tag_features(items_df)
        
        # Combine all item features
        item_features = np.hstack([
            content_embeddings,
            categorical_features,
            numerical_features,
            tag_features
        ])
        
        # Create user profiles
        user_profiles = self.create_user_profiles(users_df, items_df, interactions_df, content_embeddings)
        
        return content_embeddings, item_features, user_profiles, categorical_features
    
    def save_features(self, content_embeddings: np.ndarray, item_features: np.ndarray, 
                     user_profiles: np.ndarray, categorical_features: np.ndarray):
        """Save processed features to disk."""
        print("Saving processed features...")
        
        # Save embeddings and features
        np.save(f"{self.paths_config['processed_data_dir']}/content_embeddings.npy", content_embeddings)
        np.save(f"{self.paths_config['processed_data_dir']}/item_features.npy", item_features)
        np.save(f"{self.paths_config['processed_data_dir']}/user_profiles.npy", user_profiles)
        np.save(f"{self.paths_config['processed_data_dir']}/categorical_features.npy", categorical_features)
        
        # Save preprocessors
        with open(f"{self.paths_config['processed_data_dir']}/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(f"{self.paths_config['processed_data_dir']}/label_encoders.pkl", 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        with open(f"{self.paths_config['processed_data_dir']}/tfidf_vectorizer.pkl", 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        print(f"Features saved to {self.paths_config['processed_data_dir']}/")
    
    def load_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load processed features from disk."""
        content_embeddings = np.load(f"{self.paths_config['processed_data_dir']}/content_embeddings.npy")
        item_features = np.load(f"{self.paths_config['processed_data_dir']}/item_features.npy")
        user_profiles = np.load(f"{self.paths_config['processed_data_dir']}/user_profiles.npy")
        categorical_features = np.load(f"{self.paths_config['processed_data_dir']}/categorical_features.npy")
        
        # Load preprocessors
        with open(f"{self.paths_config['processed_data_dir']}/scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(f"{self.paths_config['processed_data_dir']}/label_encoders.pkl", 'rb') as f:
            self.label_encoders = pickle.load(f)
        
        with open(f"{self.paths_config['processed_data_dir']}/tfidf_vectorizer.pkl", 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        
        return content_embeddings, item_features, user_profiles, categorical_features


def main():
    """Main function to preprocess all data."""
    from src.data.data_loader import DataLoader
    
    # Load raw data
    loader = DataLoader()
    users_df, items_df, interactions_df = loader.load_raw_data()
    
    # Process features
    preprocessor = ContentPreprocessor()
    content_embeddings, item_features, user_profiles, categorical_features = preprocessor.process_all_features(
        users_df, items_df, interactions_df
    )
    
    # Save features
    preprocessor.save_features(content_embeddings, item_features, user_profiles, categorical_features)
    
    print("Preprocessing completed!")
    print(f"Content embeddings shape: {content_embeddings.shape}")
    print(f"Item features shape: {item_features.shape}")
    print(f"User profiles shape: {user_profiles.shape}")


if __name__ == "__main__":
    main()