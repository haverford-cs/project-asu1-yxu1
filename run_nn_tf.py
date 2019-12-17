"""
Starter code for NN training and testing.
Source: Stanford CS231n course materials, modified by Sara Mathieson
Authors: Yongxin (Fiona) Xu
Date: 11/18/2019
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch

# from fc_nn import FCmodel
from cnn import CNNmodel

import util
##################

def reshape_data(data, size=int):
    '''
    Reshape 256 features into 16*16 matrix
    '''
    return np.reshape(data,(-1,size,size, 1))

@tf.function
def train_step(model, X, y, loss_object, optimizer):
    # look up documentation for tf.GradientTape
    with tf.GradientTape() as tape:
        # compute the predictions given the images, then compute the loss
        predictions = model(X)
        loss = loss_object(y, predictions)
    # compute the gradient with respect to the model parameters (weights), then
    gradients = tape.gradient(loss, model.trainable_variables)
    # apply this gradient to update the weights (i.e. gradient descent)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # return the loss and predictions
    return loss, predictions

@tf.function
def val_step(model, X, y, loss_object):
    # compute the predictions given the images, then compute the loss
    predictions = model(X)
    loss = loss_object(y, predictions)

    # return the loss and predictions
    return loss, predictions

def run_training(model, train_dset, val_dset, epoches):
    # set up a loss_object (sparse categorical cross entropy)
    # use the Adam optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    # set up metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy( \
        name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy( \
        name='val_accuracy')

    # save for plotting curves
    train_ac = []
    val_ac = []

    # train for 10 epochs (passes over the data)
    for epoch in range(epoches):
        # Example of iterating over the data once:
        for images, labels in train_dset:
            # run training step
            loss, predictions = train_step(model, images, labels, loss_object, optimizer)
            # uncomment below
            train_loss(loss)
            train_accuracy(labels, predictions)

        # loop over validation data and compute val_loss, val_accuracy too
        for images, labels in val_dset:
            loss, predictions = val_step(model, images, labels, loss_object)
            val_loss(loss)
            val_accuracy(labels, predictions)
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}'
        print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        val_loss.result(),
                        val_accuracy.result()*100))

        train_ac.append(train_accuracy.result())
        val_ac.append(val_accuracy.result())

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

    return train_ac, val_ac

def predictions(model, test_dset, test, filename=''):
    """
    building confusion matrix
    """
    cm = np.zeros((10,10), dtype=int)
    score = 0
    length = 0
    n = 0
    for images, labels in test_dset:
        predictions = model(images)
        # print(predictions)
        for i in range(len(labels)):
            length += 1
            y_ = tf.math.argmax(predictions[i])
            y = labels[i]
            cm[y][y_]+=1

            if int(y) != int(y_):
                score += 1
                if test:
                    input_ = np.array(images[i])
                    name = './img/{f}/{f}'.format(f=filename) + str(n)
                    plt.imshow(input_.reshape(16,16), cmap="gray_r")
                    plt.title('CNN: True label: {}. Prediction: {}'.format(y, y_))
                    plt.savefig('{}.png'.format(name))
                    n+=1

    print('Test accuracy: ', (length - score)/length)

    print(cm)

def main(plot):
    # Invoke the above function to get our data.
    train_X, train_y, val_X, val_y = util.load_data('./data/semeion.data', 0.8)
    my_X, my_y = util.get_my_data('./data/testset.csv', './data/digits/')
    my_X = my_X.astype('float')
    print(my_X.dtype)
    train_X = reshape_data(train_X, 16)
    val_X = reshape_data(val_X, 16)
    my_X = reshape_data(my_X, 16)
    print('Train data shape: ', train_X.shape)              # (49000, 32, 32, 3)
    print('Train labels shape: ', train_y.shape)            # (49000,)
    print('Validation data shape: ', val_X.shape)           # (1000, 32, 32, 3)
    print('Validation labels shape: ', val_y.shape)         # (1000,)


    # set up train_dset, val_dset:
    # see documentation for tf.data.Dataset.from_tensor_slices, use batch = 64
    # train should be shuffled, but not validation and testing datasets
    train_dset = tf.data.Dataset.from_tensor_slices((train_X, train_y)).batch(64)
    val_dset = tf.data.Dataset.from_tensor_slices((val_X, val_y)).batch(64)
    my_dset = tf.data.Dataset.from_tensor_slices((my_X, my_y)).batch(64)


    # uncomment the following if you want to check the shape of images and labels
    # for images, labels in train_dset:
    #    print("images", images.shape)
    #    print("labels", labels.shape)

    # for plotting
    epoches = np.arange(1,11)

    if plot:
        cnn_model = CNNmodel()
        E = 10
        train_accuracy, val_accuracy = run_training(cnn_model, train_dset, val_dset, E)
        plt.plot(epoches, train_accuracy, epoches, val_accuracy)
        plt.xlabel('training iteration')
        plt.ylabel('accuracy')
        plt.title('Train Accuracy and Validation Accuracy: CNN')
        plt.legend(['train', 'validation'])
        plt.savefig('./img/cnn.png')
        plt.show()

        print("E = 10")
        predictions(cnn_model, val_dset)

    else:
        cnn_model_new = CNNmodel()
        E = 15
        train_accuracy, val_accuracy = run_training(cnn_model_new, train_dset, val_dset, E)

        epoches = np.arange(1,E+1)
        plt.plot(epoches, train_accuracy, epoches, val_accuracy)
        plt.xlabel('training iteration')
        plt.ylabel('accuracy')
        plt.title('Train Accuracy and Validation Accuracy: CNN')
        plt.legend(['train', 'validation'])
        # plt.savefig('./img/cnn.png')
        plt.show()

        print("E = 6")
        predictions(cnn_model_new, val_dset, False)
        print('test our own data')
        predictions(cnn_model_new, my_dset, True, 'cnn_own')

# please choose either 'cnn' or 'fc' as the input of main function
#main('fc')
main(False)
