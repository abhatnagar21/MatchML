import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd  # Importing pandas for correlation matrix

# Class to hold user data
class User:
    def __init__(self, name, age, gender, preferred_gender, hobbies, favorite_pet, location, min_age, max_age):
        self.name = name
        self.age = age
        self.gender = gender
        self.preferred_gender = preferred_gender
        self.hobbies = hobbies
        self.favorite_pet = favorite_pet
        self.location = location
        self.min_age = min_age
        self.max_age = max_age

# Function to compute cosine similarity between two vectors (hobbies)
def cosine_similarity(a, b):
    dot_product = sum(a[i] * b[i] for i in range(len(a)))
    norm_a = math.sqrt(sum(a[i] * a[i] for i in range(len(a))))
    norm_b = math.sqrt(sum(b[i] * b[i] for i in range(len(b))))
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0  # Prevent division by zero
    
    return dot_product / (norm_a * norm_b)

# Function to encode hobbies (turn them into numerical vectors)
def encode_hobbies(hobbies, all_hobbies):
    hobby_vector = [0] * len(all_hobbies)
    for hobby in hobbies:
        if hobby in all_hobbies:
            hobby_vector[all_hobbies.index(hobby)] = 1
    return hobby_vector

# Function to prepare data for machine learning
def prepare_match_data(user, other_user, all_hobbies):
    user_hobby_vector = encode_hobbies(user.hobbies, all_hobbies)
    other_user_hobby_vector = encode_hobbies(other_user.hobbies, all_hobbies)
    hobby_similarity = cosine_similarity(user_hobby_vector, other_user_hobby_vector)
    
    age_diff = abs(user.age - other_user.age)
    location_match = 1 if user.location == other_user.location else 0
    pet_match = 1 if user.favorite_pet == other_user.favorite_pet else 0
    
    return [hobby_similarity, age_diff, location_match, pet_match]

# Function to generate a random dataset for training
def generate_training_data(users, all_hobbies):
    X_train = []
    y_train = []
    
    # Simulate matching data
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            user1 = users[i]
            user2 = users[j]
            # Only consider users with compatible gender preferences
            if (user1.preferred_gender == user2.gender and user2.preferred_gender == user1.gender):
                match_features = prepare_match_data(user1, user2, all_hobbies)
                X_train.append(match_features)
                # Simulate a random outcome (1 = match, 0 = no match)
                outcome = np.random.choice([0, 1])
                y_train.append(outcome)
    
    return np.array(X_train), np.array(y_train)

# Function to train a logistic regression model
def train_match_model(users, all_hobbies):
    X_train, y_train = generate_training_data(users, all_hobbies)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, X_train, y_train  # Return the model and training data

# Function to predict match using the trained model
def predict_match(user, potential_matches, model, all_hobbies):
    X_test = [prepare_match_data(user, match, all_hobbies) for match in potential_matches]
    return model.predict(X_test), model.predict_proba(X_test)[:, 1]  # Probabilities for being a match

# Function to visualize similarity scores
def visualize_scores(users, scores):
    plt.bar([user.name for user in users], scores)
    plt.xlabel("Users")
    plt.ylabel("Match Score")
    plt.title("Match Scores for Potential Matches")
    plt.show()

# Function to display matches and their scores
def display_matches(matches, scores):
    if not matches:
        print("\nNo matches found.")
        return
    
    print("\nPotential matches found:")
    for i, match in enumerate(matches):
        print(f"Name: {match.name}")
        print(f"Age: {match.age}")
        print(f"Location: {match.location}")
        print(f"Match Score: {scores[i]:.2f}")
        print("-----------------------")

# Function to calculate and display the confusion matrix
def display_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Match', 'Match'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

def main():
    # Sample data of users
    users = [
        User("Neha", 26, "Female", "Male", ["fitness", "cooking", "art"], "Dog", "Delhi", 24, 30),
        User("Karan", 33, "Male", "Female", ["sports", "music", "gaming"], "Cat", "Chennai", 30, 38),
        User("Sanjana", 29, "Female", "Male", ["travel", "reading", "movies"], "Dog", "Mumbai", 25, 34),
        User("Rahul", 25, "Male", "Female", ["gaming", "fitness", "sports"], "Dog", "Noida", 22, 28),
        User("Tina", 31, "Female", "Male", ["yoga", "art", "travel"], "Cat", "Bangalore", 28, 36),
        User("Mohit", 24, "Male", "Female", ["music", "fitness", "cooking"], "Dog", "Delhi", 21, 30),
        User("Aditi", 27, "Female", "Male", ["dance", "cooking", "travel"], "Dog", "Chennai", 24, 32),
        User("Rohit", 29, "Male", "Female", ["movies", "fitness", "reading"], "Cat", "Hyderabad", 26, 34),
        User("Sita", 22, "Female", "Male", ["sports", "music", "art"], "Dog", "Mumbai", 20, 27),
        User("Aarav", 28, "Male", "Female", ["cooking", "travel", "fitness"], "Cat", "Delhi", 25, 33),
        User("Pooja", 30, "Female", "Male", ["fitness", "art", "sports"], "Dog", "Bangalore", 27, 35),
        User("Ravi", 35, "Male", "Female", ["reading", "gaming", "travel"], "Cat", "Delhi", 31, 40),
        User("Anisha", 25, "Female", "Male", ["travel", "fitness", "music"], "Dog", "Noida", 22, 30),
        User("Gaurav", 26, "Male", "Female", ["cooking", "movies", "sports"], "Cat", "Mumbai", 23, 32),
        User("Simran", 32, "Female", "Male", ["yoga", "art", "fitness"], "Dog", "Hyderabad", 29, 36),
        User("Vivek", 29, "Male", "Female", ["fitness", "sports", "cooking"], "Cat", "Chennai", 25, 34),
        User("Jaya", 24, "Female", "Male", ["music", "travel", "reading"], "Dog", "Bangalore", 21, 29),
        User("Siddharth", 27, "Male", "Female", ["movies", "art", "yoga"], "Cat", "Delhi", 24, 32),
        User("Neeraj", 30, "Male", "Female", ["fitness", "gaming", "travel"], "Dog", "Mumbai", 27, 35),
        User("Lata", 28, "Female", "Male", ["sports", "cooking", "reading"], "Cat", "Hyderabad", 25, 34),
        User("Amit", 26, "Male", "Female", ["travel", "music", "yoga"], "Dog", "Delhi", 23, 31),
    ]
    
    # Define all hobbies
    all_hobbies = list(set(hobby for user in users for hobby in user.hobbies))
    
    # Train the logistic regression model
    model, X_train, y_train = train_match_model(users, all_hobbies)

    # Calculate accuracy
    accuracy = model.score(X_train, y_train)
    print(f"\nModel Accuracy: {accuracy:.2f}")

    # Calculate and print the correlation matrix
    df = pd.DataFrame(X_train, columns=['Hobby Similarity', 'Age Difference', 'Location Match', 'Pet Match'])
    correlation_matrix = df.corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    # Get the current user input
    current_user = User("Arun", 21, "Male", "Female", ["music", "yoga", "gaming"], "Dog", "Delhi", 20, 25)

    # Get potential matches
    potential_matches = [user for user in users if user != current_user]

    # Predict matches
    match_predictions, match_probabilities = predict_match(current_user, potential_matches, model, all_hobbies)

    # Display matches with their scores
    display_matches(potential_matches, match_probabilities)

    # Display the confusion matrix
    display_confusion_matrix(y_train, model.predict(X_train))  # Compare true labels with predictions on training data

if __name__ == "__main__":
    main()
