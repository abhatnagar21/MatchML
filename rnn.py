import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.metrics.pairwise import cosine_similarity
data = {
    'name': [
        'Soham','Divya','Harsh','Simran','Abhishek','Naina','Vivek','Isha',
        'Tanish','Radhika','Gaurav','Aarohi','Kartik','Ira','Ayush','Sakshi',
        'Kunal','Ananya','Ved','Tara','Ritik','Esha','Arnav','Aaliya'
    ],
    'age': [24,28,29,23,26,30,25,22,27,21,31,34,32,25,33,29,28,23,30,27,26,24,31,22],
    'gender': [
        'Male','Female','Male','Female','Male','Female','Male','Female',
        'Male','Female','Male','Female','Male','Female','Male','Female',
        'Male','Female','Male','Female','Male','Female','Male','Female'
    ],
    'preferredgender': [
        'Female','Male','Female','Male','Female','Male','Female','Male',
        'Female','Male','Female','Male','Female','Male','Female','Male',
        'Female','Male','Female','Male','Female','Male','Female','Male'
    ],
    'hobbies':[
        ['cricket','photography','movies'], 
        ['yoga','dance','travel'], 
        ['gaming','football','music'], 
        ['cooking','sketching','art'], 
        ['fitness','travel','reading'], 
        ['painting','yoga','movies'], 
        ['music','cricket', 'gaming'], 
        ['running', 'art', 'fitness'], 
        ['photography','football','music'], 
        ['travel','reading','art'], 
        ['movies','gaming','cricket'], 
        ['yoga','art','travel'], 
        ['cycling','running','fitness'],
        ['painting','music','sketching'],
        ['cricket','travel','photography'],
        ['dance','cooking','art'],
        ['yoga','fitness','reading'],
        ['running','sketching','travel'],
        ['football','music','gaming'],
        ['art','movies','yoga'],
        ['dance','photography','reading'],
        ['cooking','cycling','art'],
        ['sketching','fitness','yoga'],
        ['music','reading','movies']
    ],
    'location': [
        'Lucknow','Chennai','Delhi','Mumbai','Bangalore','Kolkata','Pune','Hyderabad', 
        'Ahmedabad','Surat','Jaipur','Nagpur','Indore','Patna','Bhopal','Vadodara', 
        'Coimbatore','Chandigarh','Thiruvananthapuram','Delhi','Mumbai','Bangalore','Chennai','Kolkata'
    ]
}

df = pd.DataFrame(data)
#function to collect user input
def getuserinput():
    #Collect details of the user to find suitable matches.
    print("Enter your details:")
    name = input("Name: ")
    age = int(input("Age: "))
    gender = input("Gender (Male/Female/Other): ")
    preferredgender = input("Preferred Gender to date (Male/Female/Other): ")
    agemin = int(input("Minimum age to date: "))
    agemax = int(input("Maximum age to date: "))
    hobbies = input("Top 3 Hobbies (comma-separated): ").split(", ")
    location = input("Location: ")
    
    # Return user data as a dictionary for further processing.
    return {
        'name': name,
        'age': age,
        'gender': gender,
        'preferredgender': preferredgender,
        'agerange': (agemin, agemax),
        'hobbies': hobbies,
        'location': location
    }
#function to preprocess data
def preprocessdata(df,userdata):
    #one-hot encode categorical columns 
    encoder = OneHotEncoder(sparse_output=False)
    encodedgender = encoder.fit_transform(df[['gender', 'preferredgender', 'location']])#convert to binary
    #encode hobbies into numerical form using labelencoder
    hobbyencoder = LabelEncoder()
    flathobbies = [item for sublist in df['hobbies'] for item in sublist]#consolidate all hobbies and give them numbers
    hobbyencoder.fit(flathobbies)
    #encode hobbies for each person in the dataset
    df['hobbiesencoded'] = df['hobbies'].apply(lambda h: hobbyencoder.transform(h))
    #combine all features into a single array for each person
    df['features']=df.apply(lambda row: np.concatenate((
        [row['age']], #include age
        encodedgender[row.name],#include encoded gender, preferred gender, and location
        row['hobbiesencoded'] #include hobbies
    )), axis=1)
    #process user input
    usergenderencoded=encoder.transform([[userdata['gender'],userdata['preferredgender'],userdata['location']]])
    userhobbiesencoded =hobbyencoder.transform(userdata['hobbies'])
    userfeatures=np.concatenate((
        [userdata['age']], 
        usergenderencoded[0], 
        userhobbiesencoded
    ))
    
    #return preprocessed dataset and user features
    return df,userfeatures

#function to build an RNN model
def buildrnnmodel(inputshape):
    #create a simple RNN model to learn relationships
    model=Sequential()
    model.add(SimpleRNN(64, input_shape=(inputshape, 1),activation='relu'))#RNN layer with 64 units Uses the ReLU (Rectified Linear Unit) activation function to introduce non-linearity, allowing the model to learn complex patterns in the data.
    model.add(Dense(32,activation='relu'))#dense layer for feature extraction
    model.add(Dense(1,activation='linear'))#output layer for predictions
    model.compile(optimizer='adam', loss='mse')#compile model with Adam optimizer and MSE loss
    return model
#train the RNN on the dataset
def trainrnn(df, userfeatures):
    #convert features to a 3D shape suitable for RNN input.
    features=np.array(df['features'].tolist())
    features=features.reshape((features.shape[0],features.shape[1], 1))
    #build and train the RNN model.
    model=buildrnnmodel(features.shape[1])
    model.fit(features,features,epochs=200,verbose=0)#train for 200 epochs
    #predict match scores for the user.
    userfeatures=userfeatures.reshape((1,len(userfeatures),1))
    predictions=model.predict(userfeatures)
    return predictions
#function to find matches based on similarity scores
def findmatches(df,userfeatures,userdata):
    #compute cosine similarity between user features and dataset features
    features=np.array(df['features'].tolist())
    similarityscores=cosine_similarity([userfeatures],features)[0]
    df['similarity']=similarityscores#add similarity scores to the ataFrame  
    #filter matches by preferred gendr
    matches=df[df['gender']==userdata['preferredgender']]
    #further filter matches by age rang
    matches = matches[matches['age'].between(userdata['agerange'][0], userdata['agerange'][1])]
    # Sort matches by similarity scores in descending order.
    return matches.sort_values(by='similarity', ascending=False)
def visualizematches(matches):
    plt.figure(figsize=(10, 5))
    sns.histplot(matches['age'], kde=True)
    plt.title("Age Distribution of Potential Matches")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.show()
    #visualize the hobbies distribution of matches
    hobbies = [hobby for sublist in matches['hobbies'] for hobby in sublist]
    plt.figure(figsize=(10, 5))
    sns.countplot(x=hobbies)
    plt.title("Hobbies Distribution of Potential Matches")
    plt.xlabel("Hobbies")
    plt.ylabel("Count")
    plt.show()
# Main flow
if __name__ == "__main__":
    #collect user input
    userdata=getuserinput()
    #preprocess the dataset and user input.
    df,userfeatures=preprocessdata(df,userdata)
    #train the RNN model and predict similarity scores for the use
    predictions=trainrnn(df,userfeatures)
    #find matches based on similarity scores and other filters
    matches=findmatches(df,userfeatures,userdata)
    
    print("Top Matches:")
    print(matches[['name', 'age', 'gender', 'hobbies', 'location']])
    visualize_matches(matches)
