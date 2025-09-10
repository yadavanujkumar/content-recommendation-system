"""
Streamlit frontend for content recommendation system.
"""

import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import time

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Content Recommendation System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

def make_api_request(endpoint, method="GET", data=None):
    """Make API request with error handling."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Please make sure the API server is running on http://localhost:8000")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ API request timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def display_recommendation(rec, index):
    """Display a single recommendation."""
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**{rec['title']}**")
            st.write(f"Category: {rec['category']} | Type: {rec['content_type']}")
            st.write(f"📝 {rec['description']}")
            if 'reason' in rec:
                st.write(f"💡 {rec['reason']}")
        
        with col2:
            st.metric("Score", f"{rec.get('score', 0):.3f}")
            
            # Rating buttons
            st.write("Rate this:")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if st.button("👍", key=f"like_{index}"):
                    submit_feedback(rec['item_id'], 5.0, "like")
            with col_b:
                if st.button("👎", key=f"dislike_{index}"):
                    submit_feedback(rec['item_id'], 2.0, "dislike")
            with col_c:
                if st.button("❤️", key=f"love_{index}"):
                    submit_feedback(rec['item_id'], 5.0, "love")

def submit_feedback(item_id, rating, interaction_type):
    """Submit user feedback."""
    user_id = st.session_state.get('selected_user_id', 0)
    
    feedback_data = {
        "user_id": user_id,
        "item_id": item_id,
        "rating": rating,
        "interaction_type": interaction_type
    }
    
    result = make_api_request("/feedback", method="POST", data=feedback_data)
    if result:
        st.success(f"Feedback submitted! ({interaction_type})")
        time.sleep(1)
        st.experimental_rerun()

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🎯 Content Recommendation System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Check API health
    health = make_api_request("/health")
    if health:
        st.sidebar.success("✅ API Connected")
        st.sidebar.json(health['data_loaded'])
    else:
        st.sidebar.error("❌ API Disconnected")
        st.stop()
    
    # User selection
    st.sidebar.subheader("User Selection")
    user_id = st.sidebar.number_input("User ID", min_value=0, max_value=99, value=0)
    st.session_state['selected_user_id'] = user_id
    
    # Get user profile
    user_profile = make_api_request(f"/user/{user_id}")
    if user_profile:
        st.sidebar.subheader("User Profile")
        st.sidebar.write(f"**Age:** {user_profile['user']['age']}")
        st.sidebar.write(f"**Gender:** {user_profile['user']['gender']}")
        st.sidebar.write(f"**Location:** {user_profile['user']['location']}")
        st.sidebar.write(f"**Activity Level:** {user_profile['user']['activity_level']}")
        st.sidebar.write(f"**Interactions:** {user_profile['interaction_count']}")
        if user_profile['top_category']:
            st.sidebar.write(f"**Favorite Category:** {user_profile['top_category']}")
    
    # Recommendation settings
    st.sidebar.subheader("Recommendation Settings")
    method = st.sidebar.selectbox(
        "Method",
        ["hybrid", "content", "collaborative", "popular"],
        help="Choose recommendation algorithm"
    )
    
    k = st.sidebar.slider("Number of recommendations", 1, 20, 10)
    
    category = st.sidebar.selectbox(
        "Filter by category",
        ["None", "technology", "sports", "entertainment", "science", "politics", "health", "travel", "food"]
    )
    category = None if category == "None" else category
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🎯 Recommendations", "📊 Statistics", "👤 User History"])
    
    with tab1:
        st.header("Personalized Recommendations")
        
        if st.button("🔄 Get New Recommendations", type="primary"):
            with st.spinner("Getting recommendations..."):
                endpoint = f"/recommendations/{user_id}?k={k}&method={method}"
                if category:
                    endpoint += f"&category={category}"
                
                recommendations = make_api_request(endpoint)
                
                if recommendations:
                    st.session_state['recommendations'] = recommendations
        
        # Display recommendations
        if 'recommendations' in st.session_state:
            recs = st.session_state['recommendations']
            
            st.subheader(f"📝 {len(recs['recommendations'])} recommendations using {recs['method']} method")
            
            if recs['recommendations']:
                for i, rec in enumerate(recs['recommendations']):
                    with st.expander(f"{i+1}. {rec['title']}", expanded=(i < 3)):
                        display_recommendation(rec, i)
            else:
                st.warning("No recommendations found. Try adjusting your filters.")
    
    with tab2:
        st.header("System Statistics")
        
        stats = make_api_request("/stats")
        if stats:
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Users", stats['total_users'])
            with col2:
                st.metric("Total Items", stats['total_items'])
            with col3:
                st.metric("Total Interactions", stats['total_interactions'])
            with col4:
                st.metric("Average Rating", stats['average_rating'])
            
            # Category distribution
            st.subheader("Content Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**By Category**")
                category_df = pd.DataFrame(
                    list(stats['category_distribution'].items()),
                    columns=['Category', 'Count']
                )
                st.bar_chart(category_df.set_index('Category'))
            
            with col2:
                st.write("**By Content Type**")
                type_df = pd.DataFrame(
                    list(stats['content_type_distribution'].items()),
                    columns=['Type', 'Count']
                )
                st.bar_chart(type_df.set_index('Type'))
    
    with tab3:
        st.header("User Interaction History")
        
        if user_profile and user_profile['recent_interactions']:
            st.subheader(f"Recent interactions for User {user_id}")
            
            interactions_df = pd.DataFrame(user_profile['recent_interactions'])
            interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
            interactions_df = interactions_df.sort_values('timestamp', ascending=False)
            
            for _, interaction in interactions_df.head(10).iterrows():
                with st.expander(f"{interaction['interaction_type'].title()} - Rating: {interaction['rating']}"):
                    # Get item details
                    item_endpoint = f"/recommendations/{user_id}?k=1"  # We'll filter for this item
                    
                    st.write(f"**Item ID:** {interaction['item_id']}")
                    st.write(f"**Rating:** {interaction['rating']}/5.0")
                    st.write(f"**Type:** {interaction['interaction_type']}")
                    st.write(f"**Timestamp:** {interaction['timestamp']}")
                    st.write(f"**Session:** {interaction['session_id']}")
        else:
            st.info("No interaction history available for this user.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit and FastAPI | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

if __name__ == "__main__":
    main()