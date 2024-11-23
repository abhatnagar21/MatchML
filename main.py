import math  # importing math for maths operations
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # for evaluating model performance
import numpy as np  
import pandas as pd  

# hold user data
class User:
    def __init__(self, name, age, gender, preferredgender, hobbies, favoritepet, location, minage, maxage):
        # initialize user attributes
        self.name = name
        self.age = age
        self.gender = gender
        self.preferredgender = preferredgender
        self.hobbies = hobbies
        self.favoritepet = favoritepet
        self.location = location
        self.minage = minage
        self.maxage = maxage

# function to compute cosine similarity between 2 vectors
def cosinesimilarity(a, b):
    # calculate the dot product of two vectors
    dotproduct = sum(a[i] * b[i] for i in range(len(a)))
    # calculate the normal of each vector
    norma = math.sqrt(sum(a[i] * a[i] for i in range(len(a))))
    normb = math.sqrt(sum(b[i] * b[i] for i in range(len(b))))
    # if normal is 0
    if norma == 0.0 or normb == 0.0:
        return 0.0
    # return cosine similarity
    return dotproduct / (norma * normb)

# function to encode hobbies
def encodehobbies(hobbies, allhobbies):
    # create a zero vector of size equal to the total number of hobbies
    hobbyvector = [0] * len(allhobbies)
    for hobby in hobbies:
        # mark 1 if the hobby exists in the list of all hobbies
        if hobby in allhobbies:
            hobbyvector[allhobbies.index(hobby)] = 1
    return hobbyvector

# data preprocessing
def preparematchdata(user, otheruser, allhobbies):
    # encode hobbies into vectors
    userhobbyvector = encodehobbies(user.hobbies, allhobbies)
    otheruserhobbyvector = encodehobbies(otheruser.hobbies, allhobbies)
    # compute similarity between hobby vectors
    hobbysimilarity = cosinesimilarity(userhobbyvector, otheruserhobbyvector)
    # calculate absolute age difference
    agediff = abs(user.age - otheruser.age)
    # check if locations match (1 match, else 0)
    locationmatch = 1 if user.location == otheruser.location else 0
    # check if favorite pets match (1 for match, else 0)
    petmatch = 1 if user.favoritepet == otheruser.favoritepet else 0
    # return feature vector
    return [hobbysimilarity, agediff, locationmatch, petmatch]

# function to generate a random dataset
def generatetrainingdata(users, allhobbies):
    Xtrain = []  # features for training
    ytrain = []  # labels for training
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            user1 = users[i]
            user2 = users[j]
            # only consider users with compatible gender preferences
            if (user1.preferredgender == user2.gender and user2.preferredgender == user1.gender):
                matchfeatures = preparematchdata(user1, user2, allhobbies)
                Xtrain.append(matchfeatures)
                # randomly generate a match outcome (1 = match, 0 = no match)
                outcome = np.random.choice([0, 1])
                ytrain.append(outcome)
    # return training data as numpy arrays
    return np.array(Xtrain), np.array(ytrain)

# function to train a logistic regression model
def trainmatchmodel(users, allhobbies):
    # generate training data
    Xtrain, ytrain = generatetrainingdata(users, allhobbies)
    # create logistic regression model
    model = LogisticRegression()
    # train the model on the training data
    model.fit(Xtrain, ytrain)
    # return the trained model and training data
    return model, Xtrain, ytrain

# function to predict match
def predictmatch(user, potentialmatches, model, allhobbies):
    # prepare test data from potential matches
    Xtest = [preparematchdata(user, match, allhobbies) for match in potentialmatches]
    # predict match (1 or 0) and probability for being a match
    return model.predict(Xtest), model.predict_proba(Xtest)[:, 1]

# function to visualize similarity scores
def visualizescores(users, scores):
    plt.bar([user.name for user in users], scores)  # create a bar chart
    plt.xlabel("Users")  # label for x-axis
    plt.ylabel("Match Score")  # label for y-axis
    plt.title("Match Scores for Potential Matches")  # title of the chart
    plt.show()

# function to display matches and their scores
def displaymatches(matches, scores):
    if not matches:
        print("\nNo matches found.")
        return
    print("\nPotential matches found:")
    for i, match in enumerate(matches):
        # display user details and match score
        print(f"Name: {match.name}")
        print(f"Age: {match.age}")
        print(f"Location: {match.location}")
        print(f"Match Score: {scores[i]:.2f}")
        print("-----------------------")

# function to calculate and display the confusion matrix
def displayconfusion_matrix(ytrue, ypred):
    cm = confusion_matrix(ytrue, ypred)  # compute confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Match', 'Match'])
    disp.plot(cmap=plt.cm.Blues)  # display confusion matrix as a plot
    plt.title("Confusion Matrix")  # title of the plot
    plt.show()

# main function to execute the program
def main():
    # sample data of users
    users = [
        User("Neha", 26, "Female", "Male", ["fitness", "cooking", "art"], "Dog", "Delhi", 24, 30),
        User("Amit", 26, "Male", "Female", ["travel", "music", "yoga"], "Dog", "Delhi", 23, 31),
    ]
    # define all hobbies in the dataset
    allhobbies = list(set(hobby for user in users for hobby in user.hobbies))
    # train the logistic regression model
    model, Xtrain, ytrain = trainmatchmodel(users, allhobbies)
    # calculate accuracy of the model
    accuracy = model.score(Xtrain, ytrain)
    print(f"\nModel Accuracy: {accuracy:.2f}")
    # calculate and print the correlation matrix
    df = pd.DataFrame(Xtrain, columns=['Hobby Similarity', 'Age Difference', 'Location Match', 'Pet Match'])
    correlationmatrix = df.corr()
    print("\nCorrelation Matrix:")
    print(correlationmatrix)
    # define the current user
    currentuser = User("Arun", 21, "Male", "Female", ["music", "yoga", "gaming"], "Dog", "Delhi", 20, 25)
    # get potential matches excluding the current user
    potentialmatches = [user for user in users if user != currentuser]
    # predict matches and scores
    matchpredictions, matchprobabilities = predictmatch(currentuser, potentialmatches, model, allhobbies)
    # display matches with their scores
    displaymatches(potentialmatches, matchprobabilities)
    # display the confusion matrix for training data
    displayconfusion_matrix(ytrain, model.predict(Xtrain))

# execute the program
if __name__ == "__main__":
    main()
