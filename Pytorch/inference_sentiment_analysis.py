#data analysis (arrays, built-in functions)
import numpy as np
#pytorch
#Model Architecture
import torch
import torch.nn as nn
import pickle
from model import NeuralNet
from utils import bag_of_words, tokenize, stem, preprocess, stopword, get_wordnet_pos, lemmatizer



file = open('vocab.pkl', 'rb')
# dump information to that file
all_words = pickle.load(file)

model = torch.load('SemanticsModel.pth')
# model.eval()

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    if predicted.item() == 0:
      return 'Positive'
    else:
      return 'Negative'

if __name__ == "__main__":

    print(f"--------------------------------------------\nNote: Terminate the process by pressing quit\n--------------------------------------------\n")
    print('Lets chat')
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break
        resp = get_response(sentence)
        print(f"Model Results: {resp}")
