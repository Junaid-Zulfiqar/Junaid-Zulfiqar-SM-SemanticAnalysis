#data analysis (arrays, built-in functions)
import numpy as np
import pickle
import tensorflow as tf
from utils import bag_of_words, tokenize, stem, preprocess, stopword, get_wordnet_pos, lemmatizer



file = open('vocab.pkl', 'rb')
# dump information to that file
all_words = pickle.load(file)

#load model
model = tf.keras.models.load_model('tf_semantic.h5')

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    output = model.predict(X)
    if output[0][0] > 0.5:
      return 'Negative'
    else:
      return 'Positive'

if __name__ == "__main__":

    print(f"--------------------------------------------\nNote: Terminate the process by pressing quit\n--------------------------------------------\n")
    print('Lets chat')
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break
        resp = get_response(sentence)
        print(f"Model Results: {resp}")
