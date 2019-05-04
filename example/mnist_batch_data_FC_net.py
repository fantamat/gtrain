import os
import mnist

from gtrain import gtrain, FCNet, BatchedData
from gtrain.utils import labels2probabilities, load_weights, save_weights, join_weights_and_biases


def getMnistDataForFC():
    x_train = mnist.train_images()
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])/256
    x_test = mnist.test_images()
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])/256
    l_train = labels2probabilities(mnist.train_labels())
    l_test = labels2probabilities(mnist.test_labels())
    tenth = len(l_train)//10
    return x_train[tenth:], l_train[tenth:], x_train[:tenth], l_train[:tenth], x_test, l_test

out_dir = os.path.join("runs","mnistFCnet")

W_file = out_dir + "weights/w.npz"
b_file = out_dir + "weights/b.npz"
layer_sizes = [784, 30, 20, 10]
x_train, l_train, x_val, l_val, x_test, l_test = getMnistDataForFC()



# constants
batch_size = 32
# implementation
import numpy as np
net = FCNet(layer_sizes)
data = BatchedData(x_train, l_train, x_val, l_val, batch_size)
gtrain(net, data, num_steps=10_000, evaluate_every=500, checkpoint_every=1000,
       out_dir=out_dir, lr_dec=1.0, lr_inc=1.0, lr=.01)
save_weights([net.trained_W, net.trained_b], "mnist_FCnet_weihts")

# after that you can run command "tensorboard --logdir=runs/mnistFCnet" to run tensorboard and view it at localhost:6006
