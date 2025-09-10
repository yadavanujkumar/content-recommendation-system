"""
Simple data generation using only built-in Python libraries.
"""

import json
import random
import csv
from datetime import datetime, timedelta
import os


def generate_simple_data():
    """Generate simple test data using built-in libraries."""
    
    # Configuration
    num_users = 100
    num_items = 500
    num_interactions = 2000
    categories = ["technology", "sports", "entertainment", "science", "politics", "health", "travel", "food"]
    content_types = ["article", "video", "post"]
    
    # Set random seed
    random.seed(42)
    
    # Create directories
    os.makedirs("data/raw", exist_ok=True)
    
    # Generate users
    users = []
    for user_id in range(num_users):
        user = {
            'user_id': user_id,
            'age': random.randint(18, 65),
            'gender': random.choice(['M', 'F', 'O']),
            'location': random.choice(['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'IN', 'BR']),
            'preferred_categories': str(random.sample(categories, random.randint(1, 3))),
            'activity_level': random.choice(['low', 'medium', 'high'])
        }
        users.append(user)
    
    # Generate items
    items = []
    for item_id in range(num_items):
        category = random.choice(categories)
        content_type = random.choice(content_types)
        
        item = {
            'item_id': item_id,
            'title': f"Sample {content_type} about {category} #{item_id}",
            'description': f"This is a {content_type} about {category}. Lorem ipsum dolor sit amet.",
            'category': category,
            'content_type': content_type,
            'created_date': (datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d'),
            'duration_minutes': random.randint(1, 120) if content_type == 'video' else None,
            'word_count': random.randint(100, 5000) if content_type == 'article' else None,
            'author': f"Author_{random.randint(1, 50)}",
            'tags': str([f"tag_{random.randint(1, 20)}" for _ in range(random.randint(1, 3))])
        }
        items.append(item)
    
    # Generate interactions
    interactions = []
    for _ in range(num_interactions):
        user_id = random.randint(0, num_users - 1)
        item_id = random.randint(0, num_items - 1)
        
        interaction = {
            'user_id': user_id,
            'item_id': item_id,
            'interaction_type': random.choice(['view', 'like', 'share', 'comment']),
            'rating': round(random.uniform(1.0, 5.0), 2),
            'timestamp': (datetime.now() - timedelta(minutes=random.randint(1, 43200))).isoformat(),
            'session_id': f"session_{random.randint(1, 1000)}"
        }
        interactions.append(interaction)
    
    # Save to CSV files
    with open("data/raw/users.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=users[0].keys())
        writer.writeheader()
        writer.writerows(users)
    
    with open("data/raw/items.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=items[0].keys())
        writer.writeheader()
        writer.writerows(items)
    
    with open("data/raw/interactions.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=interactions[0].keys())
        writer.writeheader()
        writer.writerows(interactions)
    
    print("Sample data generated successfully!")
    print(f"Users: {len(users)}")
    print(f"Items: {len(items)}")
    print(f"Interactions: {len(interactions)}")


if __name__ == "__main__":
    generate_simple_data()