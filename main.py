import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.metrics.pairwise import cosine_similarity

# Expanded Data
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Hank', 'Isla', 'Jack', 'Kara', 'Leo'],
    'age': [25, 30, 27, 22, 28, 35, 29, 31, 26, 24, 33, 32],
    'gender': ['Female', 'Male', 'Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
    'preferred_gender': ['Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'hobbies': [['music', 'sports', 'travel'], 
                ['reading', 'sports', 'cooking'], 
                ['music', 'gaming', 'travel'],
                ['sports', 'travel', 'art'], 
                ['music', 'sports', 'cooking'],
                ['cooking', 'gaming', 'travel'], 
                ['sports', 'art', 'reading'], 
                ['music', 'travel', 'gaming'], 
                ['reading', 'travel', 'music'], 
                ['sports', 'music', 'travel'], 
                ['cooking', 'reading', 'art'], 
                ['music', 'travel', 'gaming']],
    'location': ['New York', 'Los Angeles', 'Chicago', 'New York', 'Chicago', 'San Francisco', 'New York', 'Los Angeles', 'Chicago', 'San Francisco', 'New York', 'Los Angeles']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Function to collect user input
def get_user_input():
    print("Enter your details:")
    name = input("Name: ")
    age = int(input("Age: "))
    gender = input("Gender (Male/Female/Other): ")
    preferred_gender = input("Preferred Gender to date (Male/Female/Other): ")
    age_min = int(input("Minimum age to date: "))
    age_max = int(input("Maximum age to date: "))
    hobbies = input("Top 3 Hobbies (comma-separated): ").split(", ")
    location = input("Location: ")
    
    # Return data as dictionary
    return {
        'name': name,
        'age': age,
        'gender': gender,
        'preferred_gender': preferred_gender,
        'age_range': (age_min, age_max),
        'hobbies': hobbies,
        'location': location
    }

# Function to preprocess user input and database
def preprocess_data(df, user_data):
    # Preprocessing categorical columns using OneHotEncoding
    encoder = OneHotEncoder(sparse_output=False)
    encoded_gender = encoder.fit_transform(df[['gender', 'preferred_gender', 'location']])
    
    # Encode hobbies using LabelEncoder
    hobby_encoder = LabelEncoder()
    flat_hobbies = [item for sublist in df['hobbies'] for item in sublist]
    hobby_encoder.fit(flat_hobbies)
    
    df['hobbies_encoded'] = df['hobbies'].apply(lambda h: hobby_encoder.transform(h))
    
    # Prepare final feature set
    df['features'] = df.apply(lambda row: np.concatenate((
        [row['age']],  # Age as a single numerical feature
        encoded_gender[row.name],  # One-hot encoded gender, preferred_gender, and location
        row['hobbies_encoded']  # Encoded hobbies
    )), axis=1)
    
    # Process user input similarly
    user_gender_encoded = encoder.transform([[user_data['gender'], user_data['preferred_gender'], user_data['location']]])
    user_hobbies_encoded = hobby_encoder.transform(user_data['hobbies'])
    user_features = np.concatenate((
        [user_data['age']], 
        user_gender_encoded[0], 
        user_hobbies_encoded
    ))
    
    return df, user_features

# RNN Model for Matching
def build_rnn_model(input_shape):
    model = Sequential()
    model.add(SimpleRNN(64, input_shape=(input_shape, 1), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

# Training the RNN on sequential data
def train_rnn(df, user_features):
    # Preparing the data for RNN
    features = np.array(df['features'].tolist())
    features = features.reshape((features.shape[0], features.shape[1], 1))  # Reshape for RNN input
    
    # Build and train RNN
    model = build_rnn_model(features.shape[1])
    model.fit(features, features, epochs=200, verbose=0)  # Increased epochs to 200 for better performance
    
    # Predict user match scores
    user_features = user_features.reshape((1, len(user_features), 1))
    predictions = model.predict(user_features)
    return predictions

# Find matches based on RNN predictions and cosine similarity
def find_matches(df, user_features, user_data):
    features = np.array(df['features'].tolist())
    similarity_scores = cosine_similarity([user_features], features)[0]
    df['similarity'] = similarity_scores
    
    # Filter based on preferred gender
    matches = df[df['gender'] == user_data['preferred_gender']]
    
    # Filter based on age range
    matches = matches[matches['age'].between(user_data['age_range'][0], user_data['age_range'][1])]
    
    return matches.sort_values(by='similarity', ascending=False)

# Visualize the matching results
def visualize_matches(matches):
    # Age distribution of matches
    plt.figure(figsize=(10, 5))
    sns.histplot(matches['age'], kde=True)
    plt.title("Age Distribution of Potential Matches")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.show()
    
    # Hobbies distribution of matches
    hobbies = [hobby for sublist in matches['hobbies'] for hobby in sublist]
    plt.figure(figsize=(10, 5))
    sns.countplot(x=hobbies)
    plt.title("Hobbies Distribution of Potential Matches")
    plt.xlabel("Hobbies")
    plt.ylabel("Count")
    plt.show()
    
# Main flow
if __name__ == "__main__":
    # Get user input
    user_data = get_user_input()

    # Preprocess data and user input
    df, user_features = preprocess_data(df, user_data)

    # Train RNN and get matches
    predictions = train_rnn(df, user_features)

    # Find matches based on predictions and gender preference
    matches = find_matches(df, user_features, user_data)

    # Show the matching results
    print(f"Top matches for {user_data['name']}:")
    print(matches[['name', 'age', 'hobbies', 'location', 'similarity']])

    # Visualize matches
    visualize_matches(matches)
