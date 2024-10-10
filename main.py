import math
import matplotlib.pyplot as plt

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

# Function to compute a match score based on multiple criteria
def compute_match_score(user, other_user, all_hobbies):
    # Cosine similarity for hobbies
    user_hobby_vector = encode_hobbies(user.hobbies, all_hobbies)
    other_user_hobby_vector = encode_hobbies(other_user.hobbies, all_hobbies)
    hobby_similarity = cosine_similarity(user_hobby_vector, other_user_hobby_vector)
    
    # Age difference score (smaller difference = better match)
    age_diff = abs(user.age - other_user.age)
    if other_user.age < user.min_age or other_user.age > user.max_age:
        age_score = 0
    else:
        age_score = max(0, 1 - (age_diff / 10))  # Max age difference of 10 years
    
    # Location score (same city = higher score)
    location_score = 1 if user.location == other_user.location else 0.5
    
    # Pet preference score (same pet = higher score)
    pet_score = 1 if user.favorite_pet == other_user.favorite_pet else 0.5
    
    # Total score is a weighted sum of all criteria
    total_score = (0.4 * hobby_similarity) + (0.3 * age_score) + (0.2 * location_score) + (0.1 * pet_score)
    
    return total_score

# Function to visualize similarity scores
def visualize_scores(users, scores):
    plt.bar([user.name for user in users], scores)
    plt.xlabel("Users")
    plt.ylabel("Match Score")
    plt.title("Match Scores for Potential Matches")
    plt.show()

# Function to get user input
def get_user_input():
    print("Enter your details:")
    name = input("Name: ")
    age = int(input("Age: "))
    gender = input("Gender (Male/Female/Other): ")
    preferred_gender = input("Preferred Gender to date (Male/Female/Other): ")
    min_age = int(input("Minimum Age to date: "))
    max_age = int(input("Maximum Age to date: "))
    hobbies_str = input("Top 3 Hobbies (comma-separated): ")
    hobbies = [hobby.strip() for hobby in hobbies_str.split(",")]
    favorite_pet = input("Favorite pet (Dog/Cat): ")
    location = input("Location: ")

    return User(name, age, gender, preferred_gender, hobbies, favorite_pet, location, min_age, max_age)

# Function to separate users by gender
def separate_by_gender(users):
    male_users = [user for user in users if user.gender == "Male"]
    female_users = [user for user in users if user.gender == "Female"]
    
    return male_users, female_users

# Function to find and rank matches based on a score from the preferred gender
def find_matches(user, male_users, female_users, all_hobbies):
    # Choose the dataset based on the user's preferred gender
    if user.preferred_gender == "Male":
        potential_matches = male_users
    elif user.preferred_gender == "Female":
        potential_matches = female_users
    else:
        # For simplicity, we assume "Other" matches with both male and female users
        potential_matches = male_users + female_users

    matches = []
    scores = []

    for other_user in potential_matches:
        # Check age range preference
        if other_user.age < user.min_age or other_user.age > user.max_age:
            continue

        # Compute the match score based on multiple criteria
        score = compute_match_score(user, other_user, all_hobbies)
        
        if score > 0.5:  # Threshold to filter out low matches
            matches.append(other_user)
            scores.append(score)

    # Sort matches based on score
    sorted_matches = [match for _, match in sorted(zip(scores, matches), reverse=True)]
    sorted_scores = sorted(scores, reverse=True)
    
    return sorted_matches, sorted_scores

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

def main():
    # Sample data of users
    users = [
        User("Alisha", 25, "Female", "Male", ["music", "sports", "travel"], "Dog", "Noida", 22, 30),
        User("Bunny", 30, "Male", "Female", ["reading", "sports", "cooking"], "Cat", "Delhi", 25, 35),
        User("Chetna", 27, "Male", "Female", ["music", "gaming", "travel"], "Dog", "Chennai", 24, 32),
        User("Dia", 22, "Female", "Male", ["sports", "travel", "art"], "Dog", "Noida", 20, 28),
        User("Simran", 28, "Female", "Male", ["music", "sports", "cooking"], "Cat", "Chennai", 25, 32),
        User("Arjun", 26, "Male", "Female", ["gaming", "fitness", "music"], "Dog", "Bangalore", 23, 30),
        User("Sara", 24, "Female", "Male", ["reading", "yoga", "travel"], "Cat", "Hyderabad", 22, 28),
        User("Rahul", 29, "Male", "Female", ["movies", "fitness", "cooking"], "Dog", "Mumbai", 25, 34),
        User("Priya", 27, "Female", "Male", ["dancing", "sports", "travel"], "Dog", "Mumbai", 24, 32),
        User("Vikram", 31, "Male", "Female", ["reading", "movies", "fitness"], "Cat", "Delhi", 27, 35),
        User("Kavya", 26, "Female", "Male", ["photography", "music", "cooking"], "Dog", "Bangalore", 24, 30),
        User("Karan", 28, "Male", "Female", ["gaming", "movies", "sports"], "Dog", "Kolkata", 25, 33),
        User("Aisha", 25, "Female", "Male", ["travel", "dancing", "fitness"], "Cat", "Ahmedabad", 22, 30),
        User("Rohit", 27, "Male", "Female", ["photography", "yoga", "movies"], "Dog", "Pune", 24, 32),
        User("Neha", 23, "Female", "Male", ["sports", "art", "yoga"], "Cat", "Pune", 20, 26),
        User("Aman", 29, "Male", "Female", ["sports", "music", "gaming"], "Dog", "Jaipur", 25, 33),
        User("Riya", 24, "Female", "Male", ["movies", "yoga", "dancing"], "Cat", "Kolkata", 22, 27),
        User("Yash", 28, "Male", "Female", ["fitness", "travel", "photography"], "Dog", "Hyderabad", 24, 32),
        User("Isha", 26, "Female", "Male", ["art", "photography", "music"], "Cat", "Delhi", 23, 29),
        User("Manish", 27, "Male", "Female", ["gaming", "movies", "fitness"], "Dog", "Bangalore", 24, 32),
        User("Meera", 28, "Female", "Male", ["yoga", "fitness", "cooking"], "Dog", "Noida", 25, 32),
        User("Nikhil", 30, "Male", "Female", ["sports", "movies", "reading"], "Cat", "Chennai", 26, 34),
        User("Tanvi", 24, "Female", "Male", ["dancing", "sports", "music"], "Dog", "Jaipur", 21, 27),
        User("Siddharth", 29, "Male", "Female", ["music", "travel", "gaming"], "Dog", "Delhi", 25, 33),
    ]

    # Define all possible hobbies
    all_hobbies = ["music", "sports", "travel", "reading", "gaming", "cooking", "art"]

    # Separate users by gender
    male_users, female_users = separate_by_gender(users)

    # Get the current user input
    current_user = get_user_input()

    # Find and rank matches
    matches, scores = find_matches(current_user, male_users, female_users, all_hobbies)

    # Visualize match scores
    visualize_scores(matches, scores)

    # Display matches and their scores
    display_matches(matches, scores)

if __name__ == "__main__":
    main()
