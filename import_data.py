import tensorflow as tf
import numpy as np


# read features
def read_features_from_csv(filename, usecols=range(1, 785)):
    features = np.genfromtxt(filename, delimiter=',', skip_header=1, usecols=usecols, dtype=np.float32)
    features = np.divide(features, 255.0)  # scale 0..255 to 0..1
    return features


# read labels and convert them to 1-hot vectors
def read_labels_from_csv(filename):
    labels_orig = np.genfromtxt(filename, delimiter=',', skip_header=1, usecols=0, dtype=np.int)
    labels = np.zeros([len(labels_orig), 10])
    labels[np.arange(len(labels_orig)), labels_orig] = 1
    labels = labels.astype(np.float32)
    return labels


# generate batches
def generate_batch(features, labels, batch_size):
    batch_indexes = np.random.random_integers(0, len(features) - 1, batch_size)
    batch_features = features[batch_indexes]
    batch_labels = labels[batch_indexes]
    return (batch_features, batch_labels)
