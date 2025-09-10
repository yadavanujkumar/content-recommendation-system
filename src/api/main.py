"""
FastAPI backend for content recommendation system.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import csv
import json
import random
from datetime import datetime
import os

# Initialize FastAPI app
app = FastAPI(
    title="Content Recommendation System API",
    description="API for content recommendation system with multiple recommendation strategies",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class RecommendationRequest(BaseModel):
    user_id: int
    k: int = 10
    method: str = "hybrid"  # content, collaborative, hybrid
    category: Optional[str] = None

class FeedbackRequest(BaseModel):
    user_id: int
    item_id: int
    rating: float
    interaction_type: str = "rating"

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[Dict[str, Any]]
    method: str
    timestamp: str

# Global variables for data storage
users_data = []
items_data = []
interactions_data = []

def load_data():
    """Load data from CSV files."""
    global users_data, items_data, interactions_data
    
    try:
        # Load users
        with open("data/raw/users.csv", 'r') as f:
            users_data = list(csv.DictReader(f))
        
        # Load items
        with open("data/raw/items.csv", 'r') as f:
            items_data = list(csv.DictReader(f))
        
        # Load interactions
        with open("data/raw/interactions.csv", 'r') as f:
            interactions_data = list(csv.DictReader(f))
        
        # Convert string representations back to proper types
        for user in users_data:
            user['user_id'] = int(user['user_id'])
            user['age'] = int(user['age'])
            user['preferred_categories'] = eval(user['preferred_categories'])
        
        for item in items_data:
            item['item_id'] = int(item['item_id'])
            try:
                item['duration_minutes'] = int(item['duration_minutes']) if item['duration_minutes'] and item['duration_minutes'] != 'None' else None
                item['word_count'] = int(item['word_count']) if item['word_count'] and item['word_count'] != 'None' else None
            except ValueError:
                item['duration_minutes'] = None
                item['word_count'] = None
            item['tags'] = eval(item['tags'])
        
        for interaction in interactions_data:
            interaction['user_id'] = int(interaction['user_id'])
            interaction['item_id'] = int(interaction['item_id'])
            interaction['rating'] = float(interaction['rating'])
        
        print(f"Loaded {len(users_data)} users, {len(items_data)} items, {len(interactions_data)} interactions")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        # Create empty data if files don't exist
        users_data = []
        items_data = []
        interactions_data = []

def get_user_interactions(user_id: int) -> List[Dict]:
    """Get all interactions for a user."""
    return [i for i in interactions_data if i['user_id'] == user_id]

def get_item_by_id(item_id: int) -> Optional[Dict]:
    """Get item by ID."""
    for item in items_data:
        if item['item_id'] == item_id:
            return item
    return None

def get_user_by_id(user_id: int) -> Optional[Dict]:
    """Get user by ID."""
    for user in users_data:
        if user['user_id'] == user_id:
            return user
    return None

def content_based_recommendations(user_id: int, k: int = 10, category: Optional[str] = None) -> List[Dict]:
    """Simple content-based recommendations."""
    user = get_user_by_id(user_id)
    if not user:
        return []
    
    user_interactions = get_user_interactions(user_id)
    seen_items = {i['item_id'] for i in user_interactions}
    
    # Get user's preferred categories
    preferred_categories = user.get('preferred_categories', [])
    
    # Score items based on category preference
    item_scores = []
    for item in items_data:
        if item['item_id'] in seen_items:
            continue
        
        if category and item['category'] != category:
            continue
        
        score = 0.5  # Base score
        
        # Boost score for preferred categories
        if item['category'] in preferred_categories:
            score += 0.4
        
        # Add some randomness
        score += random.uniform(0, 0.1)
        
        item_scores.append((item, score))
    
    # Sort by score and return top k
    item_scores.sort(key=lambda x: x[1], reverse=True)
    
    recommendations = []
    for item, score in item_scores[:k]:
        rec = item.copy()
        rec['score'] = round(score, 3)
        rec['reason'] = f"Recommended based on your interest in {item['category']}"
        recommendations.append(rec)
    
    return recommendations

def collaborative_recommendations(user_id: int, k: int = 10) -> List[Dict]:
    """Simple collaborative filtering recommendations."""
    user_interactions = get_user_interactions(user_id)
    if not user_interactions:
        return popular_recommendations(k)
    
    seen_items = {i['item_id'] for i in user_interactions}
    user_ratings = {i['item_id']: i['rating'] for i in user_interactions}
    
    # Find similar users (users who rated similar items highly)
    similar_users = []
    for other_user_id in range(len(users_data)):
        if other_user_id == user_id:
            continue
        
        other_interactions = get_user_interactions(other_user_id)
        other_ratings = {i['item_id']: i['rating'] for i in other_interactions}
        
        # Calculate similarity based on common items
        common_items = set(user_ratings.keys()) & set(other_ratings.keys())
        if len(common_items) < 2:
            continue
        
        similarity = 0
        for item_id in common_items:
            similarity += abs(user_ratings[item_id] - other_ratings[item_id])
        
        similarity = 1 / (1 + similarity / len(common_items))  # Convert to similarity
        similar_users.append((other_user_id, similarity))
    
    # Sort by similarity
    similar_users.sort(key=lambda x: x[1], reverse=True)
    
    # Get recommendations from similar users
    item_scores = {}
    for other_user_id, similarity in similar_users[:10]:  # Top 10 similar users
        other_interactions = get_user_interactions(other_user_id)
        for interaction in other_interactions:
            item_id = interaction['item_id']
            if item_id in seen_items:
                continue
            
            if item_id not in item_scores:
                item_scores[item_id] = 0
            
            item_scores[item_id] += similarity * interaction['rating']
    
    # Sort and get top recommendations
    sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
    
    recommendations = []
    for item_id, score in sorted_items[:k]:
        item = get_item_by_id(item_id)
        if item:
            rec = item.copy()
            rec['score'] = round(score, 3)
            rec['reason'] = "Recommended based on users with similar preferences"
            recommendations.append(rec)
    
    return recommendations

def popular_recommendations(k: int = 10) -> List[Dict]:
    """Get popular items based on ratings."""
    item_ratings = {}
    item_counts = {}
    
    for interaction in interactions_data:
        item_id = interaction['item_id']
        rating = interaction['rating']
        
        if item_id not in item_ratings:
            item_ratings[item_id] = 0
            item_counts[item_id] = 0
        
        item_ratings[item_id] += rating
        item_counts[item_id] += 1
    
    # Calculate average ratings
    item_avg_ratings = {}
    for item_id in item_ratings:
        if item_counts[item_id] >= 3:  # At least 3 ratings
            item_avg_ratings[item_id] = item_ratings[item_id] / item_counts[item_id]
    
    # Sort by average rating
    sorted_items = sorted(item_avg_ratings.items(), key=lambda x: x[1], reverse=True)
    
    recommendations = []
    for item_id, avg_rating in sorted_items[:k]:
        item = get_item_by_id(item_id)
        if item:
            rec = item.copy()
            rec['score'] = round(avg_rating, 3)
            rec['reason'] = f"Popular item with {item_counts[item_id]} ratings"
            recommendations.append(rec)
    
    return recommendations

def hybrid_recommendations(user_id: int, k: int = 10, category: Optional[str] = None) -> List[Dict]:
    """Hybrid recommendations combining content-based and collaborative."""
    content_recs = content_based_recommendations(user_id, k=k*2, category=category)
    collab_recs = collaborative_recommendations(user_id, k=k*2)
    
    # Combine recommendations with weights
    combined_scores = {}
    
    # Content-based weight: 0.6
    for rec in content_recs:
        item_id = rec['item_id']
        combined_scores[item_id] = 0.6 * rec['score']
    
    # Collaborative weight: 0.4
    for rec in collab_recs:
        item_id = rec['item_id']
        if item_id in combined_scores:
            combined_scores[item_id] += 0.4 * rec['score']
        else:
            combined_scores[item_id] = 0.4 * rec['score']
    
    # Sort by combined score
    sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    recommendations = []
    for item_id, score in sorted_items[:k]:
        item = get_item_by_id(item_id)
        if item:
            rec = item.copy()
            rec['score'] = round(score, 3)
            rec['reason'] = "Hybrid recommendation combining multiple approaches"
            recommendations.append(rec)
    
    return recommendations

# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Load data on startup."""
    load_data()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Content Recommendation System API",
        "version": "1.0.0",
        "endpoints": [
            "/recommendations/{user_id}",
            "/feedback",
            "/health",
            "/stats"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_loaded": {
            "users": len(users_data),
            "items": len(items_data),
            "interactions": len(interactions_data)
        }
    }

@app.get("/recommendations/{user_id}")
async def get_recommendations(
    user_id: int,
    k: int = Query(10, ge=1, le=50),
    method: str = Query("hybrid", regex="^(content|collaborative|hybrid|popular)$"),
    category: Optional[str] = Query(None)
) -> RecommendationResponse:
    """Get recommendations for a user."""
    
    if user_id < 0 or user_id >= len(users_data):
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get recommendations based on method
    if method == "content":
        recommendations = content_based_recommendations(user_id, k, category)
    elif method == "collaborative":
        recommendations = collaborative_recommendations(user_id, k)
    elif method == "hybrid":
        recommendations = hybrid_recommendations(user_id, k, category)
    elif method == "popular":
        recommendations = popular_recommendations(k)
    else:
        raise HTTPException(status_code=400, detail="Invalid recommendation method")
    
    return RecommendationResponse(
        user_id=user_id,
        recommendations=recommendations,
        method=method,
        timestamp=datetime.now().isoformat()
    )

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback."""
    global interactions_data
    
    # Validate user and item exist
    if feedback.user_id < 0 or feedback.user_id >= len(users_data):
        raise HTTPException(status_code=404, detail="User not found")
    
    if not get_item_by_id(feedback.item_id):
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Add new interaction
    new_interaction = {
        'user_id': feedback.user_id,
        'item_id': feedback.item_id,
        'interaction_type': feedback.interaction_type,
        'rating': feedback.rating,
        'timestamp': datetime.now().isoformat(),
        'session_id': f"api_session_{random.randint(1000, 9999)}"
    }
    
    interactions_data.append(new_interaction)
    
    # Save to file (in a real system, you'd use a database)
    try:
        with open("data/raw/interactions.csv", 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=new_interaction.keys())
            writer.writerow(new_interaction)
    except Exception as e:
        print(f"Error saving feedback: {e}")
    
    return {
        "message": "Feedback submitted successfully",
        "interaction_id": len(interactions_data) - 1,
        "timestamp": new_interaction['timestamp']
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    # Calculate basic statistics
    total_users = len(users_data)
    total_items = len(items_data)
    total_interactions = len(interactions_data)
    
    # Category distribution
    category_counts = {}
    for item in items_data:
        category = item['category']
        category_counts[category] = category_counts.get(category, 0) + 1
    
    # Content type distribution
    content_type_counts = {}
    for item in items_data:
        content_type = item['content_type']
        content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1
    
    # Average rating
    if interactions_data:
        avg_rating = sum(float(i['rating']) for i in interactions_data) / len(interactions_data)
    else:
        avg_rating = 0
    
    return {
        "total_users": total_users,
        "total_items": total_items,
        "total_interactions": total_interactions,
        "average_rating": round(avg_rating, 2),
        "category_distribution": category_counts,
        "content_type_distribution": content_type_counts
    }

@app.get("/user/{user_id}")
async def get_user_profile(user_id: int):
    """Get user profile and interaction history."""
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_interactions = get_user_interactions(user_id)
    
    # Calculate user statistics
    if user_interactions:
        avg_rating = sum(float(i['rating']) for i in user_interactions) / len(user_interactions)
        favorite_categories = {}
        for interaction in user_interactions:
            item = get_item_by_id(interaction['item_id'])
            if item:
                category = item['category']
                favorite_categories[category] = favorite_categories.get(category, 0) + 1
        
        top_category = max(favorite_categories.items(), key=lambda x: x[1])[0] if favorite_categories else None
    else:
        avg_rating = 0
        top_category = None
    
    return {
        "user": user,
        "interaction_count": len(user_interactions),
        "average_rating": round(avg_rating, 2),
        "top_category": top_category,
        "recent_interactions": user_interactions[-10:]  # Last 10 interactions
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)