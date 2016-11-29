# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pickle

pickle_file = 'assets/notMNIST.pickle'

with open(pickle_file, mode='rb') as f:
    save = pickle.load(f, encoding='latin1')
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


batch_size = 128
num_hidden_nodes = 1024
num_second_hidden_nodes = 300
num_third_hidden_nodes = 50

restricted_train_dataset = train_dataset[0:batch_size * 3, :]
restricted_train_labels = train_labels[0:batch_size * 3, :]

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    weights_1 = tf.Variable(
        tf.truncated_normal([image_size * image_size, num_hidden_nodes], stddev=0.1))
    weights_2 = tf.Variable(tf.truncated_normal([num_hidden_nodes, num_second_hidden_nodes], stddev=0.1))
    weights_3 = tf.Variable(tf.truncated_normal([num_second_hidden_nodes, num_third_hidden_nodes], stddev=0.1))
    weights_4 = tf.Variable(tf.truncated_normal([num_third_hidden_nodes, num_labels], stddev=0.1))
    biases_1 = tf.Variable(tf.zeros([num_hidden_nodes]))
    biases_2 = tf.Variable(tf.zeros([num_second_hidden_nodes]))
    biases_3 = tf.Variable(tf.zeros([num_third_hidden_nodes]))
    biases_4 = tf.Variable(tf.zeros([num_labels]))


    # Training computation.
    def model(data, train):
        first_layer = tf.matmul(data, weights_1) + biases_1
        first_relu = tf.nn.relu(first_layer)
        second_layer = tf.matmul(first_relu, weights_2) + biases_2
        second_relu = tf.nn.relu(second_layer)
        third_layer = tf.matmul(second_relu, weights_3) + biases_3
        third_relu = tf.nn.relu(third_layer)
        fourth_layer = tf.matmul(third_relu, weights_4) + biases_4
        if train:
            fourth_layer = tf.nn.dropout(fourth_layer, 0.5)

        return fourth_layer


    logits = model(tf_train_dataset, True)
    reg_mult = 1e-3
    regularization = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2) + tf.nn.l2_loss(weights_3) + tf.nn.l2_loss(
        weights_4)
    regularization = regularization + tf.nn.l2_loss(biases_1) + tf.nn.l2_loss(biases_2) + tf.nn.l2_loss(
        biases_3) + tf.nn.l2_loss(biases_4)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + reg_mult * regularization

    # Optimizer.
    global_step = tf.Variable(0)  # count the number of steps taken.
    learning_rate = tf.train.exponential_decay(0.5, global_step, len(train_dataset), 0.95)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset, False))
    test_prediction = tf.nn.softmax(model(tf_test_dataset, False))

num_steps = 10001

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(
                valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
