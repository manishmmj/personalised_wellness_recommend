pip install flask

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained models and scaler
scaler = joblib.load("scaler.joblib")
kmeans = joblib.load("kmeans_model.joblib")
df = pd.read_excel("/mnt/data/User_Wellness_Data.xlsx")

# Load recommendations mapping
cluster_recommendations = {
    0: "Meditation and relaxation activities",
    1: "Increase physical activity and hydration",
    2: "Focus on improving sleep hygiene",
    3: "Balanced diet and mindfulness practices",
    4: "Regular exercise and stress management techniques"
}

def recommend_activities(user_profile, num_recommendations=3):
    """
    Recommend activities based on user's input profile.

    Args:
        user_profile (dict): User's input features.
        num_recommendations (int): Number of recommendations to provide.

    Returns:
        list: Recommended activities.
    """
    # Normalize user input
    user_features = np.array([user_profile[feature] for feature in numerical_features]).reshape(1, -1)
    user_features_scaled = scaler.transform(user_features)

    # Predict user's cluster
    user_cluster = kmeans.predict(user_features_scaled)[0]

    # Find similar users in the same cluster
    similar_users = df[df['KMeans_Cluster'] == user_cluster]

    # Get top recommendations
    recommended_activities = similar_users['Recommendations'].value_counts().head(num_recommendations).index.tolist()
    return recommended_activities

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    API endpoint to get activity recommendations.

    Request JSON Example:
    {
        "Sleep_Duration_Hours": 6,
        "Physical_Activity_Minutes": 30,
        "Heart_Rate_bpm": 75,
        "Step_Count": 5000,
        "Body_Temperature_C": 36.5
    }

    Response JSON Example:
    {
        "recommendations": ["Meditation", "Improve Sleep Hygiene", "Increase Hydration"]
    }
    """
    try:
        # Parse user input
        user_profile = request.json

        # Validate input
        for feature in numerical_features:
            if feature not in user_profile:
                return jsonify({"error": f"Missing feature: {feature}"}), 400

        # Get recommendations
        recommendations = recommend_activities(user_profile)

        return jsonify({"recommendations": recommendations})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/clusters', methods=['GET'])
def get_clusters():
    """
    API endpoint to get details about clusters.

    Response JSON Example:
    {
        "clusters": {
            "0": "Meditation and relaxation activities",
            "1": "Increase physical activity and hydration",
            ...
        }
    }
    """
    return jsonify({"clusters": cluster_recommendations})

@app.route('/visualize', methods=['GET'])
def visualize_clusters():
    """
    API endpoint to generate cluster visualization (if applicable).

    Note: Actual implementation would depend on the chosen visualization technique
    and may require returning static files or dynamically generated plots.
    """
    return jsonify({"message": "Visualization endpoint under construction."})

if __name__ == '__main__':
    app.run(debug=True)
