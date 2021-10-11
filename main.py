#This chatbot allows modifications of intents
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy 
import tflearn
import tensorflow
import random
import json

with open("Intents.json") as file:
    data = json.load(file)

words= []
labels= []
docs_x= []
docs_y= []

#loop through all the intents within the Intent JSON file.
for intent in data["intents"]:
    #patterns are the user input
    for pattern in intent["patterns"]:
        #tokenize all the words in each pattern key (returns a list); the lines below stemming each pattern by detecting the root word 
        #This is necessary to train the deep learning model
        wrds = nltk.word_tokenize(pattern) #returns a list
        words.extend(wrds) #adds to a list
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
    #responses are the output that are trained from the JSON file
    if intent["tag"] not in labels:
        labels.append(intent["tag"])
        #stemming breaks sentences down to root words when which helps training the bot (makes the model more better)
        #tokenize the data

words = [stemmer.stem(w.lower()) for w in words if w != "?"] #the question mark does not really have any value
words = sorted(List(set(words)))

labels = sorted(labels)

#one hot encoding; tracking the amount of words that will occur (frequency) to the neural network
training = []
output= []

out_empty= [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag= []
    wrds= [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row= out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training= numpy.array(training)
output= numpy.array(output)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")