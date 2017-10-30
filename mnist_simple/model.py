#
# model for CNN with MNIST
#

import tensorflow as tf
from mnist_simple import ModelInterface, ModelError

class ModelBase(ModelInterface):
    def __init__(self):
        super(ModelBase, self).__init__()
        ## conv 1
        self.__w1 = self.weight_variable([5, 5, 1, 16])
        self.__b1 = self.bias_variable([16])
        ## conv 2
        self.__w2 = self.weight_variable([5, 5, 16, 32])
        self.__b2 = self.bias_variable([32])
        ## full connect 1
        self.__w3 = self.weight_variable([7 * 7 * 32, 1024])
        self.__b3 = self.bias_variable([1024])
        ## full connect 2
        self.__w4 = self.weight_variable([1024, 10])
        self.__b4 = self.bias_variable([10])

    def init_model(self):
        ## input
        imgs = tf.reshape(self.images,[-1,28,28,1])
        ## conv layer 1
        conv1 = tf.nn.relu(self.conv2D(imgs, self.__w1) + self.__b1)
        pool1 = self.max_pool(conv1,2)
        ## conv layer 2
        conv2 = tf.nn.relu(self.conv2D(pool1, self.__w2) + self.__b2)
        pool2 = self.max_pool(conv2,2)
        ## full connection layer 1
        full1_in = tf.reshape(pool2,[-1,7*7*32])
        full1 = tf.nn.relu(tf.matmul(full1_in, self.__w3) + self.__b3)
        full2 = tf.nn.relu(tf.matmul(full1, self.__w4) + self.__b4)
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=full2))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(full2,1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


class ModelWithDrop(ModelInterface):
    def __init__(self):
        super(ModelWithDrop, self).__init__()
        ## conv 1
        self.__w1 = self.weight_variable([5, 5, 1, 16])
        self.__b1 = self.bias_variable([16])
        ## conv 2
        self.__w2 = self.weight_variable([5, 5, 16, 32])
        self.__b2 = self.bias_variable([32])
        ## full connect 1
        self.__w3 = self.weight_variable([7 * 7 * 32, 1024])
        self.__b3 = self.bias_variable([1024])
        ### drop of for fc 1
        self.__keep_prob = tf.placeholder(tf.float32)
        ## full connect 2
        self.__w4 = self.weight_variable([1024, 10])
        self.__b4 = self.bias_variable([10])

    def init_model(self):
        ## input
        imgs = tf.reshape(self.images,[-1,28,28,1])
        ## conv layer 1
        conv1 = tf.nn.relu(self.conv2D(imgs, self.__w1) + self.__b1)
        pool1 = self.max_pool(conv1,2)
        ## conv layer 2
        conv2 = tf.nn.relu(self.conv2D(pool1, self.__w2) + self.__b2)
        pool2 = self.max_pool(conv2,2)
        ## full connection layer 1
        full1_in = tf.reshape(pool2,[-1,7*7*32])
        full1 = tf.nn.relu(tf.matmul(full1_in, self.__w3) + self.__b3)
        ### drop of fc1
        full1d = tf.nn.dropout(full1,self.keep_prob)
        ## full connection layer 2
        full2 = tf.nn.relu(tf.matmul(full1d, self.__w4) + self.__b4)
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=full2))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(full2,1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return 0

    @property
    def keep_prob(self):
        return self.__keep_prob
    @keep_prob.setter
    def keep_prob(self,value):
        self.__keep_prob = value



