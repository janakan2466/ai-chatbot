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
docs= []

#loop through all the intents within the Intent JSON file.
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        #tokenize all the words in each pattern key (returns a list)
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs.append(pattern)

    if intent["tag"] not in labels:
        labels.append(intent["tag"])
        #stemming breaks sentences down to root words when which helps training the bot (makes the model more better)
        #tokenize the data