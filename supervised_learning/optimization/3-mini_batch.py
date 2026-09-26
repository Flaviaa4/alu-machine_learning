#!/usr/bin/env python3
"""Defines a function that trains a loaded neural network model using
mini-batch gradient descent"""
import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(
    X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5,
    load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"
):
    """Trains a loaded neural network model using mini-batch gradient
    descent

    X_train: numpy.ndarray of shape (m, 784) containing the training data
    Y_train: one-hot numpy.ndarray of shape (m, 10) containing the
        training labels
    X_valid: numpy.ndarray of shape (m, 784) containing the validation
        data
    Y_valid: one-hot numpy.ndarray of shape (m, 10) containing the
        validation labels
    batch_size: the number of data points in a batch
    epochs: the number of times the training should pass through the
        whole dataset
    load_path: the path from which to load the model
    save_path: the path to where the model should be saved after
        training

    Returns: the path where the model was saved
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        m = X_train.shape[0]

        for epoch in range(epochs + 1):
            train_cost, train_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if epoch < epochs:
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)

                for step, start in enumerate(range(0, m, batch_size), 1):
                    end = min(start + batch_size, m)
                    X_batch = X_shuffled[start:end]
                    Y_batch = Y_shuffled[start:end]

                    sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                    if step % 100 == 0:
                        step_cost, step_accuracy = sess.run(
                            [loss, accuracy],
                            feed_dict={x: X_batch, y: Y_batch})
                        print("\tStep {}:".format(step))
                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy: {}".format(step_accuracy))

        return saver.save(sess, save_path)
