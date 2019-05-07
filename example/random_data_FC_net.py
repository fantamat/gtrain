import numpy as np
import tensorflow as tf

from gtrain import FCNet, gtrain
from gtrain.data import AllData

np.random.seed(555)
tf.set_random_seed(555)


def generate_test_data(num_of_train_samples=1000, num_of_validation_samples=100):
    data_tr = np.random.rand(num_of_train_samples, 2)
    data_val = np.random.rand(num_of_validation_samples, 2)
    l_tr = np.zeros([num_of_train_samples, 3])
    l_val = np.zeros([num_of_validation_samples, 3])

    def sample_class(sample):
        if (sample[0] * sample[0] + sample[1] * sample[1]) < 0.0:
            return 0
        else:
            if sample[1] > 0.5:
                return 1
            else:
                return 2

    for i in range(len(l_tr)):
        l_tr[i][sample_class(data_tr[i])] = 1
    for i in range(len(l_val)):
        l_val[i][sample_class(data_val[i])] = 1
    return data_tr, l_tr, data_val, l_val

layer_input_sizes = 2
num_of_samples = 1000
num_of_classes = 3
data_tr, l_tr, data_val, l_val = generate_test_data(num_of_samples, num_of_classes)

# initialization of weights
inner_dimensions = [layer_input_sizes, 3,  num_of_classes]

train_net = FCNet(inner_dimensions, use_cross_entropy=False)
data = AllData(data_tr, l_tr, data_val, l_val)
gtrain(train_net, data, num_steps=10000, evaluate_every=1000, checkpoint_every=1000, num_checkpoints=5)

