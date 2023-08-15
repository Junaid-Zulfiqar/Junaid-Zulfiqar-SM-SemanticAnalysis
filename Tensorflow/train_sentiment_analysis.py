import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')

#for data analysis (specially for reading and handling files)
import pandas as pd
import re, string
#data analysis (arrays, built-in functions)
import numpy as np
#pytorch
from itertools import chain
import pickle
from utils import bag_of_words, tokenize, stem, preprocess, stopword, get_wordnet_pos, lemmatizer, finalpreprocess
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt



if __name__=='__main__':
    df = pd.read_csv("imdb_10K_sentimnets_reviews.csv")
    #clean reviews
    df['clean_review'] = df['review'].apply(lambda x: finalpreprocess(x))
    #flatten the clean reviews
    all_words = list(chain.from_iterable(df.clean_review.to_list()))
    #do some steming on the words
    all_words = [stem(w) for w in all_words]
    #get words whose length is greater than 2
    all_words = [w for w in all_words if len(w) > 2]
    #remove duplicate words
    all_words = list(set(all_words))
    #dump into the pickle file, so that we can use this vocab in other files
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(all_words, f)

    #list that contains tuple(review, sentiment)
    xy = []
    for review, label in zip(df.clean_review, df.sentiment):
        xy.append((review,label))

    # create training data
    X_train = []
    y_train = []
    tags = df.sentiment.to_list()
    for (pattern_sentence, tag) in xy:
        # X: bag of words for each pattern_sentence
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
        label = tags.index(tag)
        y_train.append(label)

    #convert List to array
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    #split into train and validations
    X_val = X_train[:9000]
    y_val = y_train[:9000]
    partial_x_train = X_train[9000:]
    partial_y_train = y_train[9000:]

    # Hyper-parameters
    # Length of the vocabulary in chars
    input_size = len(partial_x_train[0])
    BATCH_SIZE=128
    hidden_size =16
    num_classes = 1
    EPOCHS = 20

    model = models.Sequential()
    model.add(layers.Dense(hidden_size, activation='relu', input_shape=(input_size,)))
    model.add(layers.Dense(hidden_size, activation='relu'))
    model.add(layers.Dense(num_classes, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy', 'Precision', 'Recall'])

    #fit model
    history = model.fit(partial_x_train,
            partial_y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val)
            )
    history_dict = history.history

    #plot training and validation loss
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = np.arange(1, len(history_dict["accuracy"]) + 1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    #plot training and validation accuracy
    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']
    plt.plot(epochs, history_dict["accuracy"], 'bo', label='Training acc')
    plt.plot(epochs, history_dict["val_accuracy"], 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


    input_size = len(X_train[0])

    # Retrain model
    model = models.Sequential()
    model.add(layers.Dense(hidden_size, activation='relu', input_shape=(input_size,)))
    model.add(layers.Dense(hidden_size, activation='relu'))
    model.add(layers.Dense(num_classes, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy', 'Precision', 'Recall'])

    model.fit(X_train, y_train, epochs=10, batch_size=128)
    #save model
    model.save('tf_semantic.h5')