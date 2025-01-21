import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
class User:
    def __init__(self,name,age,gender,preferredgender,hobbies,favoritepet,location,minage,maxage):
        self.name=name
        self.age=age
        self.gender=gender
        self.preferredgender=preferredgender
        self.hobbies=hobbies 
        self.favoritepet=favoritepet
        self.location=location
        self.minage=minage
        self.maxage=maxage
def cosinesimilarity(a, b):
    dotproduct=sum(a[i]*b[i] for i in range(len(a)))
    norma=math.sqrt(sum(a[i]*a[i] for i in range(len(a))))
    normb=math.sqrt(sum(b[i]*b[i] for i in range(len(b))))
    if norma==0.0 or normb==0.0:
        return 0.0
    return dotproduct/(norma*normb)
def encodehobbies(hobbies,allhobbies):
    hobbyvector=[0]*len(allhobbies)
    for hobby in hobbies:
        if hobby in allhobbies:
            hobbyvector[allhobbies.index(hobby)] = 1
    return hobbyvector

# data preprocessing
def preparematchdata(user,otheruser,allhobbies):
    #encode hobbies into vectors
    userhobbyvector=encodehobbies(user.hobbies,allhobbies)
    otheruserhobbyvector=encodehobbies(otheruser.hobbies,allhobbies)
    #compute similarity between hobby vectors
    hobbysimilarity=cosinesimilarity(userhobbyvector,otheruserhobbyvector) 
    #calculate absolute age difference
    agediff = abs(user.age - otheruser.age)
    #check if locations match (1 match, else 0)
    locationmatch = 1 if user.location == otheruser.location else 0
    #check if favorite pets match (1 for match, else 0)
    petmatch=1 if user.favoritepet == otheruser.favoritepet else 0
    #return feature vector
    return [hobbysimilarity, agediff, locationmatch, petmatch]


# Function to generate a random dataset
def generatetrainingdata(users, allhobbies):
    Xtrain=[]
    ytrain=[]
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            user1=users[i]
            user2=users[j]
            if (user1.preferredgender == user2.gender and user2.preferredgender == user1.gender):
                matchfeatures=preparematchdata(user1,user2,allhobbies)
                Xtrain.append(matchfeatures)
                outcome = np.random.choice([0, 1])
                ytrain.append(outcome)
    return np.array(Xtrain),np.array(ytrain)

#logistic regression model
def trainmatchmodel(users,allhobbies):
    Xtrain,ytrain=generatetrainingdata(users,allhobbies)
    model=LogisticRegression()
    model.fit(Xtrain,ytrain)
    return model,Xtrain,ytrain

#function to predict match
def predictmatch(user,potentialmatches,model,allhobbies):
    Xtest=[preparematchdata(user,match,allhobbies) for match in potentialmatches]
    return model.predict(Xtest), model.predict_proba(Xtest)[:, 1]#wd array column match and no match selecting probability of match

def visualizescores(users,scores):
    plt.bar([user.name for user in users], scores)
    plt.xlabel("Users")
    plt.ylabel("Match Score")
    plt.title("Match Scores for Potential Matches")
    plt.show()


def displaymatches(matches,scores):
    if not matches:
        print("\nNo matches found.")
        return
    print("\nPotential matches found:")
    for i, match in enumerate(matches):
        print(f"Name:{match.name}")
        print(f"Age:{match.age}")
        print(f"Location:{match.location}")
        print(f"Match Score:{scores[i]:.2f}")
        print("-----------------------")

#function to calculate and display confusion matrix
def displayconfusion_matrix(ytrue, ypred):
    cm = confusion_matrix(ytrue, ypred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Match','Match'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

# Main function
def main():
    # Sample data of users (could be replaced with dynamic input if needed)
    users = [
        User("Neha", 26, "Female", "Male", ["fitness", "cooking", "art"], "Dog", "Delhi", 24, 30),
User("Amit", 26, "Male", "Female", ["travel", "music", "yoga"], "Dog", "Delhi", 23, 31),
User("Kriti", 28, "Female", "Male", ["fashion", "cooking", "fitness"], "Cat", "Mumbai", 25, 33),
User("Nikhil", 32, "Male", "Female", ["sports", "reading", "movies"], "Dog", "Bangalore", 30, 36),
User("Maya", 27, "Female", "Male", ["baking", "yoga", "painting"], "None", "Chennai", 25, 31),
User("Gaurav", 30, "Male", "Female", ["gaming", "travel", "photography"], "Cat", "Pune", 28, 34),
User("Meera", 29, "Female", "Male", ["fitness", "hiking", "travel"], "Dog", "Mumbai", 27, 32),
User("Ayesha", 24, "Female", "Male", ["dancing", "cooking", "photography"], "None", "Delhi", 22, 27),
User("Raj", 28, "Male", "Female", ["gaming", "traveling", "sports"], "Dog", "Kolkata", 26, 32),
User("Sanya", 26, "Female", "Male", ["fitness", "baking", "music"], "Dog", "Bangalore", 24, 29),
User("Varun", 30, "Male", "Female", ["movies", "fitness", "yoga"], "Cat", "Hyderabad", 28, 33),
User("Rohit", 27, "Male", "Female", ["photography", "travel", "fitness"], "Dog", "Delhi", 25, 30),
User("Nisha", 24, "Female", "Male", ["reading", "traveling", "cooking"], "Cat", "Mumbai", 22, 28),
User("Siddharth", 29, "Male", "Female", ["fitness", "hiking", "cooking"], "Dog", "Chennai", 26, 32),
User("Deepika", 28, "Female", "Male", ["dancing", "painting", "music"], "Cat", "Pune", 26, 30),
User("Shiv", 30, "Male", "Female", ["gaming", "sports", "cooking"], "None", "Delhi", 28, 34),
User("Priyanka", 27, "Female", "Male", ["traveling", "fitness", "yoga"], "Dog", "Bangalore", 25, 31),
User("Manish", 28, "Male", "Female", ["sports", "reading", "cooking"], "Cat", "Hyderabad", 26, 32),
User("Aarti", 29, "Female", "Male", ["photography", "fitness", "cooking"], "Dog", "Chennai", 27, 33),
User("Kunal", 32, "Male", "Female", ["travel", "yoga", "reading"], "None", "Pune", 30, 35),
User("Shweta", 27, "Female", "Male", ["painting", "fitness", "dancing"], "Cat", "Delhi", 25, 31),
User("Suresh", 30, "Male", "Female", ["fitness", "hiking", "cooking"], "Dog", "Mumbai", 28, 33),
User("Ritika", 26, "Female", "Male", ["yoga", "fitness", "baking"], "Dog", "Bangalore", 24, 30),
User("Vishal", 31, "Male", "Female", ["traveling", "movies", "sports"], "Cat", "Delhi", 29, 34),
User("Neeraj", 28, "Male", "Female", ["fitness", "photography", "gaming"], "None", "Kolkata", 26, 32),
User("Sanjana", 25, "Female", "Male", ["traveling", "reading", "fitness"], "Dog", "Chennai", 23, 28),
User("Ravi", 27, "Male", "Female", ["movies", "sports", "reading"], "Cat", "Hyderabad", 25, 30),
User("Ishita", 26, "Female", "Male", ["dancing", "cooking", "music"], "Dog", "Pune", 24, 29),
User("Vikram", 30, "Male", "Female", ["gaming", "fitness", "photography"], "Cat", "Bangalore", 28, 34),
User("Akash", 27, "Male", "Female", ["sports", "reading", "fitness"], "None", "Delhi", 25, 30),
User("Anjali", 29, "Female", "Male", ["traveling", "yoga", "baking"], "Dog", "Mumbai", 27, 33),
User("Manju", 31, "Female", "Male", ["cooking", "travel", "fitness"], "Cat", "Chennai", 29, 35),
User("Tarun", 32, "Male", "Female", ["movies", "traveling", "fitness"], "None", "Pune", 30, 36),
User("Riya", 28, "Female", "Male", ["reading", "traveling", "photography"], "Cat", "Mumbai", 26, 32),
User("Rahul", 30, "Male", "Female", ["gaming", "fitness", "movies"], "Dog", "Bangalore", 28, 35),
User("Simran", 25, "Female", "Male", ["dancing", "music", "art"], "None", "Pune", 22, 28),
User("Karan", 27, "Male", "Female", ["cooking", "hiking", "gardening"], "Dog", "Delhi", 25, 30),
User("Priya", 29, "Female", "Male", ["yoga", "movies", "painting"], "Cat", "Hyderabad", 27, 34),
User("Arjun", 26, "Male", "Female", ["traveling", "fitness", "photography"], "None", "Kolkata", 24, 30),
User("Sneha", 24, "Female", "Male", ["reading", "yoga", "baking"], "Dog", "Mumbai", 22, 27),
User("Vikram", 31, "Male", "Female", ["sports", "movies", "music"], "None", "Chennai", 29, 35),
User("Ananya", 27, "Female", "Male", ["fitness", "travel", "dancing"], "Cat", "Delhi", 25, 31),
User("Abhishek", 29, "Male", "Female", ["yoga", "cooking", "gardening"], "Dog", "Bangalore", 27, 33),
        User("Riya", 25, "Female", "Male", ["reading", "writing", "baking"], "Cat", "Mumbai", 23, 28),
    User("Kunal", 30, "Male", "Female", ["running", "travel", "cycling"], "Dog", "Pune", 28, 34),
    User("Sneha", 26, "Female", "Male", ["painting", "hiking", "gardening"], "Rabbit", "Hyderabad", 24, 30),
    User("Arjun", 28, "Male", "Female", ["swimming", "fitness", "travel"], "Parrot", "Kolkata", 25, 31),
    User("Nisha", 24, "Female", "Male", ["photography", "travel", "reading"], "Fish", "Chennai", 22, 26),
    User("Rahul", 31, "Male", "Female", ["yoga", "writing", "cooking"], "Dog", "Delhi", 29, 35),
    User("Priya", 29, "Female", "Male", ["dancing", "painting", "cycling"], "Cat", "Jaipur", 27, 32),
    User("Rohit", 27, "Male", "Female", ["cricket", "hiking", "baking"], "Dog", "Lucknow", 25, 30),
    User("Ankita", 28, "Female", "Male", ["fitness", "travel", "gardening"], "Rabbit", "Ahmedabad", 26, 30),
    User("Aditya", 32, "Male", "Female", ["writing", "cooking", "photography"], "Fish", "Surat", 30, 36),
    User("Neha", 23, "Female", "Male", ["travel", "dancing", "painting"], "Parrot", "Bangalore", 21, 26),
    User("Vikas", 29, "Male", "Female", ["gardening", "fitness", "hiking"], "Cat", "Chandigarh", 27, 31),
    User("Sanya", 25, "Female", "Male", ["baking", "swimming", "yoga"], "Dog", "Kochi", 23, 27),
    User("Harsh", 30, "Male", "Female", ["photography", "cycling", "travel"], "Rabbit", "Mumbai", 28, 33),
    User("Isha", 26, "Female", "Male", ["reading", "writing", "baking"], "Fish", "Delhi", 24, 29),
    User("Kabir", 27, "Male", "Female", ["cricket", "gardening", "dancing"], "Parrot", "Hyderabad", 25, 30),
    User("Maya", 24, "Female", "Male", ["hiking", "swimming", "travel"], "Dog", "Pune", 22, 27),
    User("Nikhil", 28, "Male", "Female", ["yoga", "cooking", "painting"], "Cat", "Bangalore", 26, 31),
    User("Pooja", 29, "Female", "Male", ["cycling", "dancing", "fitness"], "Rabbit", "Chennai", 27, 33),
    User("Amit", 31, "Male", "Female", ["writing", "travel", "baking"], "Fish", "Kolkata", 29, 34),

    ]
    name=input("Enter your name: ")
    age=int(input("Enter your age: "))
    gender=input("Enter your gender (Male/Female): ")
    preferredgender=input("Enter your preferred gender (Male/Female): ")
    hobbies=input("Enter your hobbies (comma separated): ").split(",")
    favoritepet=input("Enter your favorite pet: ")
    location=input("Enter your location: ")
    minage=int(input("Enter the minimum age of a preferred match: "))
    maxage=int(input("Enter the maximum age of a preferred match: "))
    currentuser=User(name,age,gender,preferredgender,hobbies,favoritepet,location,minage,maxage)
    
    allhobbies=list(set(hobby for user in users for hobby in user.hobbies))
    
    
    model, Xtrain, ytrain = trainmatchmodel(users, allhobbies)
    accuracy = model.score(Xtrain, ytrain)
    print(f"\nModel Accuracy:{accuracy:.2f}")
    
    #get potential matches excluding the current user and filter based on preferred gender
    potentialmatches=[user for user in users if user != currentuser and user.gender==currentuser.preferredgender]
    
    #predict matches and scores
    matchpredictions,matchprobabilities=predictmatch(currentuser,potentialmatches,model,allhobbies)
    
    #display matches with their scores
    displaymatches(potentialmatches,matchprobabilities)
    
    #display the confusion matrix for training data
    displayconfusion_matrix(ytrain, model.predict(Xtrain))

if __name__ == "__main__":
    main()
