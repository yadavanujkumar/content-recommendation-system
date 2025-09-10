"""
Demonstration script showing the full capabilities of the Content Recommendation System
"""

import requests
import json
import time

API_BASE = "http://localhost:8000"

def test_api():
    """Test all API endpoints"""
    print("=== Content Recommendation System Demo ===\n")
    
    # 1. Health check
    print("1. Health Check:")
    response = requests.get(f"{API_BASE}/health")
    if response.status_code == 200:
        health = response.json()
        print(f"   ✅ API is healthy")
        print(f"   📊 Data: {health['data_loaded']['users']} users, {health['data_loaded']['items']} items, {health['data_loaded']['interactions']} interactions")
    else:
        print("   ❌ API is not responding")
        return
    
    # 2. System statistics
    print("\n2. System Statistics:")
    response = requests.get(f"{API_BASE}/stats")
    stats = response.json()
    print(f"   📈 Average rating: {stats['average_rating']}/5.0")
    print(f"   📚 Categories: {list(stats['category_distribution'].keys())}")
    print(f"   📄 Content types: {list(stats['content_type_distribution'].keys())}")
    
    # 3. User profile
    user_id = 5
    print(f"\n3. User Profile (User {user_id}):")
    response = requests.get(f"{API_BASE}/user/{user_id}")
    user_data = response.json()
    user = user_data['user']
    print(f"   👤 Age: {user['age']}, Gender: {user['gender']}, Location: {user['location']}")
    print(f"   ❤️ Prefers: {user['preferred_categories']}")
    print(f"   📊 {user_data['interaction_count']} interactions, avg rating: {user_data['average_rating']:.2f}")
    
    # 4. Test different recommendation methods
    methods = ["hybrid", "content", "collaborative", "popular"]
    
    print(f"\n4. Recommendation Methods for User {user_id}:")
    
    for method in methods:
        print(f"\n   📋 {method.upper()} Method:")
        response = requests.get(f"{API_BASE}/recommendations/{user_id}?k=3&method={method}")
        recs = response.json()
        
        if recs['recommendations']:
            for i, rec in enumerate(recs['recommendations'], 1):
                print(f"      {i}. {rec['title']}")
                print(f"         Category: {rec['category']} | Score: {rec['score']:.3f}")
        else:
            print("      No recommendations found")
    
    # 5. Category-specific recommendations
    print(f"\n5. Sports Recommendations for User {user_id}:")
    response = requests.get(f"{API_BASE}/recommendations/{user_id}?k=3&method=hybrid&category=sports")
    recs = response.json()
    
    for i, rec in enumerate(recs['recommendations'], 1):
        print(f"   {i}. {rec['title']} (Score: {rec['score']:.3f})")
    
    # 6. Submit feedback
    print(f"\n6. Submitting Feedback:")
    feedback_data = {
        "user_id": user_id,
        "item_id": 100,
        "rating": 4.8,
        "interaction_type": "like"
    }
    
    response = requests.post(f"{API_BASE}/feedback", json=feedback_data)
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ Feedback submitted: {result['message']}")
    else:
        print("   ❌ Feedback submission failed")
    
    # 7. Performance demonstration
    print(f"\n7. Performance Test:")
    start_time = time.time()
    
    # Get recommendations for multiple users
    for test_user_id in range(0, 10):
        response = requests.get(f"{API_BASE}/recommendations/{test_user_id}?k=5&method=hybrid")
    
    end_time = time.time()
    print(f"   ⚡ Generated recommendations for 10 users in {end_time - start_time:.2f} seconds")
    print(f"   📈 Average response time: {(end_time - start_time) / 10 * 1000:.1f}ms per user")
    
    print("\n=== Demo Complete ===")
    print("\nThe Content Recommendation System demonstrates:")
    print("✅ Multiple recommendation algorithms (Content-based, Collaborative, Hybrid)")
    print("✅ Real-time personalized recommendations")
    print("✅ Category filtering and user preferences")
    print("✅ User feedback collection and processing")
    print("✅ RESTful API with comprehensive endpoints")
    print("✅ Web-based user interface")
    print("✅ Scalable architecture with fast response times")

if __name__ == "__main__":
    try:
        test_api()
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Please ensure the API server is running on port 8000")
        print("Start with: python src/api/main.py")