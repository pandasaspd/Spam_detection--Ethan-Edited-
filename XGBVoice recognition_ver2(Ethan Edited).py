ㅟ### Importing required modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import xgboost as xgb


### Importing SMSSpam dataset
f=open('/Users/HongSukhyun/Desktop/Python/미리짜놓은 코드:소스/데이터/SMSSpamCollectiondata.csv', "rt", encoding="utf=8")
read=csv.reader(f)
data=[]
for row in read:
    data.append(row)
table1=pd.DataFrame(data=data)
table1.columns=['labels', 'chat']

### Extracting out the data and labels (ham,spam) from the dataset file
data = table1[['chat']].as_matrix()
labels = table1[['labels']].as_matrix()

### Importing required module for TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=1)
data = table1

label = data[['labels']]
dt = data[['chat']]

a = dt.values.tolist()
dt1 = [l[0] for l in a]
### Voice Recognition

import speech_recognition as sr

### Obtaining audio from microphone
def audio_message():

    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)
    
    

def text_message():
    return input("Type your message: ")

# recognizing speech using Google Speech Recognition

    # for testing purposes, we're just using the default API key
    # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
    # instead of `r.recognize_google(audio)`
    
    
    ###Recieving a message

while True:
    try:
        while True:
            choice = input("message or voice?: (Type c to exit)")
            if choice in ["message", "voice", "Message", "Voice"]:
                break
            elif choice=="c":
                print ("End programm")
                break
            else:
                print ("type it correctly! Type 'c' to exit: ")
        
    #Text message 
        if choice in ["message", "Message"]:
            text = text_message()
            print("Your message is: " + text)
    #Voice message into text
        elif choice in ["voice", "Voice"]:
            text = audio_message()
            print("Google Speech Recognition thinks you said: " + text)
        
    
        dt1.append(text)
        dt2 = [l[0] for l in label.values.tolist()]

        ### Vectorizing dataset with TFIDF
        X = vectorizer.fit_transform(dt1).toarray()[:-1]
        your_message = np.array([vectorizer.fit_transform(dt1).toarray()[-1]])
        y = np.array(dt2)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=2018)
        
        #XGboost
        gbm =xgb.XGBClassifier(max_depth=3,n_eatimators=1000,early_stop_rounds = 100, learning_rate=0.15).fit(X_train,y_train)
        def score(a, b):
            count = 0
            for i in range(len(a)):
                if a[i] == b[i]:
                    count+=1
            return count
        
        prediction_test = gbm.predict(X_valid)
        hits = score(prediction_test, y_valid)
        print ("Accuracy is: ", (hits/len(y_valid))*100)
        
        result = gbm.predict(your_message)
        if result[0] == "ham":
            result = "is not a spam"
        print ("Your message is: ", result)
        
        break
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio. Say it again: ")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        break
    
###training 을 루프 위로 올려 runtime을 줄이기!