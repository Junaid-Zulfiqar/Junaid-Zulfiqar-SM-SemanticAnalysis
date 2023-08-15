import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

#for data analysis (specially for reading and handling files)
import pandas as pd
import re, string
#data analysis (arrays, built-in functions)
import numpy as np
#pytorch
from torch.utils.data import Dataset, DataLoader
from itertools import chain
import pickle
import torch
import torch.nn as nn
from model import NeuralNet
from utils import bag_of_words, tokenize, stem, preprocess, stopword, get_wordnet_pos, lemmatizer, finalpreprocess


#Attach index to the dataset class for generating dataloader
class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


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

    # Hyper-parameters
    num_epochs = 8
    batch_size = 32
    learning_rate = 0.001
    input_size = len(X_train[0])
    hidden_size = 16
    output_size = 2
    print(input_size, output_size)

    #shuffle train data and convert into the batches
    dataset = ChatDataset()
    train_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)

    #build model
    model = NeuralNet(input_size, hidden_size, output_size)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Train the model
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            labels = labels.to(dtype=torch.long)

            # Forward pass
            outputs = model(words)
            # if y would be one-hot, we must apply
            # labels = torch.max(labels, 1)[1]
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    print(f'final loss: {loss.item():.4f}')
    torch.save(model, 'SemanticsModel.pth')
