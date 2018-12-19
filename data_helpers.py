import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),\+!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\+", " \+ ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def not_empty(s):
    return s and s.strip()


def load_data_and_labels(files):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files

    labelSize = len(files)
    x_text = np.array([])
    y_label = np.array([])
    for fileIndex, file in enumerate(files):
        # read all sentences and split each sentence into words
        sentences = list(open(file, "r").readlines())
        sentences = [s.strip() for s in sentences]
        sentences = [clean_str(s) for s in sentences]
        sentences = list(filter(not_empty, sentences))
        sentences = np.array(sentences)
        x_text = np.concatenate([x_text, sentences],0)
        # generate labels for each sentence
        labels = [ [0 for x in range(labelSize)] for i in range(len(sentences)) ]
        for label in labels: label[fileIndex] = 1
        if fileIndex == 0:
            y_label = np.array(labels)
        else:
            y_label = np.concatenate([y_label,labels],0)

    return [x_text, y_label]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
