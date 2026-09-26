#!/usr/bin/env python3
"""Defines a function that builds, trains, and saves a neural network
model using Adam optimization, mini-batch gradient descent, learning
rate decay, and batch normalization"""
import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data
create_batch_norm_layer = __import__('14-batch_norm').create_batch_norm_layer


def create_placeholders(nx, classes):
    """Creates placeholders for a neural network

    nx: the number of feature columns
    classes: the number of classes in the classifier

    Returns: placeholders named x and y, respectively
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y


def create_layer(prev, n, activation):
    """Creates a plain (non-batch-normalized) layer for a neural network

    prev: the tensor output of the previous layer
    n: the number of nodes in the layer to create
    activation: the activation function that the layer should use

    Returns: the tensor output of the layer
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(
        units=n, activation=activation, kernel_initializer=initializer
    )
    return layer(prev)


def forward_prop(x, layers, activations):
    """Creates the forward propagation graph for the neural network,
    using batch normalization on every layer except the last

    x: the placeholder for the input data
    layers: a list containing the number of nodes in each layer
    activations: a list containing the activation functions for each
        layer

    Returns: the prediction of the network in tensor form
    """
    prediction = x
    for i in range(len(layers)):
        if i == len(layers) - 1:
            prediction = create_layer(
                prediction, layers[i], activations[i]
            )
        else:
            prediction = create_batch_norm_layer(
                prediction, layers[i], activations[i]
            )
    return prediction


def calculate_accuracy(y, y_pred):
    """Calculates the accuracy of a prediction

    y: a placeholder for the labels of the input data
    y_pred: a tensor containing the network's predictions

    Returns: a tensor containing the decimal accuracy of the prediction
    """
    correct = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))


def calculate_loss(y, y_pred):
    """Calculates the softmax cross-entropy loss of a prediction

    y: a placeholder for the labels of the input data
    y_pred: a tensor containing the network's predictions

    Returns: a tensor containing the loss of the prediction
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Creates the training operation using the Adam optimization
    algorithm

    loss: the loss of the network
    alpha: the learning rate
    beta1: the weight used for the first moment
    beta2: the weight used for the second moment
    epsilon: a small number to avoid division by zero

    Returns: the Adam optimization operation
    """
    optimizer = tf.train.AdamOptimizer(
        alpha, beta1=beta1, beta2=beta2, epsilon=epsilon
    )
    return optimizer.minimize(loss)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Creates a learning rate decay operation using inverse time decay

    alpha: the original learning rate
    decay_rate: the weight used to determine the rate at which alpha
        will decay
    global_step: the number of passes of gradient descent that have
        elapsed
    decay_step: the number of passes of gradient descent that should
        occur before alpha is decayed further

    Returns: the learning rate decay operation
    """
    return tf.train.inverse_time_decay(
        alpha, global_step, decay_step, decay_rate, staircase=True
    )


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """Builds, trains, and saves a neural network model in tensorflow
    using Adam optimization, mini-batch gradient descent, learning rate
    decay, and batch normalization

    Data_train: tuple containing the training inputs and training
        labels, respectively
    Data_valid: tuple containing the validation inputs and validation
        labels, respectively
    layers: list containing the number of nodes in each layer of the
        network
    activations: list containing the activation functions used for each
        layer of the network
    alpha: the learning rate
    beta1: the weight for the first moment of Adam Optimization
    beta2: the weight for the second moment of Adam Optimization
    epsilon: a small number used to avoid division by zero
    decay_rate: the decay rate for inverse time decay of the learning
        rate (the corresponding decay step is 1)
    batch_size: the number of data points that should be in a mini-batch
    epochs: the number of times the training should pass through the
        whole dataset
    save_path: the path where the model should be saved to

    Returns: the path where the model was saved
    """
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    nx = X_train.shape[1]
    classes = Y_train.shape[1]

    x, y = create_placeholders(nx, classes)
    y_pred = forward_prop(x, layers, activations)
    accuracy = calculate_accuracy(y, y_pred)
    loss = calculate_loss(y, y_pred)

    global_step = tf.Variable(0, trainable=False)
    alpha_decayed = learning_rate_decay(alpha, decay_rate, global_step, 1)
    train_op = create_Adam_op(loss, alpha_decayed, beta1, beta2, epsilon)

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    m = X_train.shape[0]

    with tf.Session() as sess:
        sess.run(init)

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
                        print("\t\tAccuracy {}".format(step_accuracy))

                sess.run(tf.assign(global_step, global_step + 1))

        return saver.save(sess, save_path)
