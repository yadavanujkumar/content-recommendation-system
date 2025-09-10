"""
Data generation module for content recommendation system.
Generates synthetic users, items, and interactions for testing and development.
"""

import pandas as pd
import numpy as np
import random
from typing import List, Dict, Tuple
import yaml
import os
from datetime import datetime, timedelta


class DataGenerator:
    """Generate synthetic data for content recommendation system."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.paths_config = self.config['paths']
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        random.seed(42)
    
    def generate_users(self) -> pd.DataFrame:
        """Generate user data with demographics and preferences."""
        users = []
        
        for user_id in range(self.data_config['num_users']):
            user = {
                'user_id': user_id,
                'age': np.random.randint(18, 65),
                'gender': np.random.choice(['M', 'F', 'O'], p=[0.45, 0.45, 0.1]),
                'location': np.random.choice(['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'IN', 'BR']),
                'preferred_categories': np.random.choice(
                    self.data_config['categories'], 
                    size=np.random.randint(1, 4), 
                    replace=False
                ).tolist(),
                'activity_level': np.random.choice(['low', 'medium', 'high'], p=[0.2, 0.5, 0.3])
            }
            users.append(user)
        
        return pd.DataFrame(users)
    
    def generate_items(self) -> pd.DataFrame:
        """Generate item/content data with metadata."""
        items = []
        
        # Sample titles and descriptions for different categories
        content_templates = {
            'technology': {
                'titles': ['AI Revolution', 'Machine Learning Basics', 'Cloud Computing', 'Cybersecurity'],
                'keywords': ['artificial intelligence', 'programming', 'software', 'innovation']
            },
            'sports': {
                'titles': ['Championship Game', 'Player Statistics', 'Team Analysis', 'Sports News'],
                'keywords': ['football', 'basketball', 'soccer', 'olympics']
            },
            'entertainment': {
                'titles': ['Movie Review', 'Celebrity News', 'Music Album', 'TV Show'],
                'keywords': ['hollywood', 'music', 'cinema', 'celebrity']
            },
            'science': {
                'titles': ['Space Discovery', 'Medical Breakthrough', 'Research Study', 'Scientific Method'],
                'keywords': ['research', 'discovery', 'experiment', 'innovation']
            },
            'politics': {
                'titles': ['Election Update', 'Policy Analysis', 'Government News', 'Political Commentary'],
                'keywords': ['government', 'policy', 'election', 'democracy']
            },
            'health': {
                'titles': ['Fitness Tips', 'Nutrition Guide', 'Mental Health', 'Medical Advice'],
                'keywords': ['wellness', 'exercise', 'nutrition', 'health']
            },
            'travel': {
                'titles': ['Destination Guide', 'Travel Tips', 'Adventure Story', 'Cultural Experience'],
                'keywords': ['tourism', 'culture', 'adventure', 'destination']
            },
            'food': {
                'titles': ['Recipe Collection', 'Restaurant Review', 'Cooking Tips', 'Food Culture'],
                'keywords': ['cooking', 'recipe', 'cuisine', 'restaurant']
            }
        }
        
        for item_id in range(self.data_config['num_items']):
            category = np.random.choice(self.data_config['categories'])
            content_type = np.random.choice(self.data_config['content_types'])
            
            # Generate title and description
            base_title = np.random.choice(content_templates[category]['titles'])
            title = f"{base_title} - {category.title()} {content_type.title()}"
            
            keywords = content_templates[category]['keywords']
            description = f"Explore {np.random.choice(keywords)} in this comprehensive {content_type} about {category}. " \
                         f"Learn about {np.random.choice(keywords)} and discover {np.random.choice(keywords)}."
            
            item = {
                'item_id': item_id,
                'title': title,
                'description': description,
                'category': category,
                'content_type': content_type,
                'created_date': self._random_date(),
                'duration_minutes': np.random.randint(1, 120) if content_type == 'video' else None,
                'word_count': np.random.randint(100, 5000) if content_type == 'article' else None,
                'author': f"Author_{np.random.randint(1, 100)}",
                'tags': random.sample(keywords, k=min(3, len(keywords)))
            }
            items.append(item)
        
        return pd.DataFrame(items)
    
    def generate_interactions(self, users_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
        """Generate user-item interactions with realistic patterns."""
        interactions = []
        
        # Create user preferences based on demographics
        user_preferences = self._create_user_preferences(users_df, items_df)
        
        for _ in range(self.data_config['num_interactions']):
            user_id = np.random.randint(0, len(users_df))
            user = users_df.iloc[user_id]
            
            # Select item based on user preferences
            item_id = self._select_item_for_user(user, items_df, user_preferences[user_id])
            
            # Generate interaction details
            interaction_type = np.random.choice(['view', 'like', 'share', 'comment'], p=[0.7, 0.2, 0.05, 0.05])
            rating = self._generate_rating(user, items_df.iloc[item_id], interaction_type)
            
            interaction = {
                'user_id': user_id,
                'item_id': item_id,
                'interaction_type': interaction_type,
                'rating': rating,
                'timestamp': self._random_timestamp(),
                'session_id': f"session_{np.random.randint(1, 10000)}"
            }
            interactions.append(interaction)
        
        return pd.DataFrame(interactions)
    
    def _create_user_preferences(self, users_df: pd.DataFrame, items_df: pd.DataFrame) -> Dict:
        """Create user preference weights for different categories."""
        preferences = {}
        
        for idx, user in users_df.iterrows():
            user_prefs = {}
            
            # Higher weights for preferred categories
            for category in self.data_config['categories']:
                if category in user['preferred_categories']:
                    user_prefs[category] = np.random.uniform(0.7, 1.0)
                else:
                    user_prefs[category] = np.random.uniform(0.1, 0.4)
            
            preferences[user['user_id']] = user_prefs
        
        return preferences
    
    def _select_item_for_user(self, user: pd.Series, items_df: pd.DataFrame, preferences: Dict) -> int:
        """Select an item for a user based on preferences."""
        # Calculate item weights based on user preferences
        weights = []
        for _, item in items_df.iterrows():
            weight = preferences.get(item['category'], 0.1)
            
            # Boost weight for content type preferences
            if user['activity_level'] == 'high' and item['content_type'] == 'video':
                weight *= 1.2
            elif user['activity_level'] == 'low' and item['content_type'] == 'post':
                weight *= 1.1
                
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        return np.random.choice(len(items_df), p=weights)
    
    def _generate_rating(self, user: pd.Series, item: pd.Series, interaction_type: str) -> float:
        """Generate a rating based on user-item compatibility."""
        base_rating = 3.0
        
        # Adjust based on category preference
        if item['category'] in user['preferred_categories']:
            base_rating += np.random.uniform(0.5, 1.5)
        
        # Adjust based on interaction type
        if interaction_type == 'like':
            base_rating += np.random.uniform(0.5, 1.0)
        elif interaction_type == 'share':
            base_rating += np.random.uniform(1.0, 1.5)
        elif interaction_type == 'comment':
            base_rating += np.random.uniform(0.5, 1.0)
        
        # Add some noise
        base_rating += np.random.normal(0, 0.3)
        
        return max(1.0, min(5.0, base_rating))
    
    def _random_date(self) -> str:
        """Generate a random date in the past year."""
        start_date = datetime.now() - timedelta(days=365)
        random_days = np.random.randint(0, 365)
        random_date = start_date + timedelta(days=random_days)
        return random_date.strftime('%Y-%m-%d')
    
    def _random_timestamp(self) -> str:
        """Generate a random timestamp in the past month."""
        start_time = datetime.now() - timedelta(days=30)
        random_seconds = np.random.randint(0, 30 * 24 * 3600)
        random_timestamp = start_time + timedelta(seconds=random_seconds)
        return random_timestamp.isoformat()
    
    def generate_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate all data (users, items, interactions)."""
        print("Generating users...")
        users_df = self.generate_users()
        
        print("Generating items...")
        items_df = self.generate_items()
        
        print("Generating interactions...")
        interactions_df = self.generate_interactions(users_df, items_df)
        
        return users_df, items_df, interactions_df
    
    def save_data(self, users_df: pd.DataFrame, items_df: pd.DataFrame, interactions_df: pd.DataFrame):
        """Save generated data to files."""
        os.makedirs(self.paths_config['raw_data_dir'], exist_ok=True)
        
        users_df.to_csv(f"{self.paths_config['raw_data_dir']}/users.csv", index=False)
        items_df.to_csv(f"{self.paths_config['raw_data_dir']}/items.csv", index=False)
        interactions_df.to_csv(f"{self.paths_config['raw_data_dir']}/interactions.csv", index=False)
        
        print(f"Data saved to {self.paths_config['raw_data_dir']}/")
        print(f"Users: {len(users_df)}")
        print(f"Items: {len(items_df)}")
        print(f"Interactions: {len(interactions_df)}")


def main():
    """Main function to generate and save data."""
    generator = DataGenerator()
    users_df, items_df, interactions_df = generator.generate_all_data()
    generator.save_data(users_df, items_df, interactions_df)


if __name__ == "__main__":
    main()