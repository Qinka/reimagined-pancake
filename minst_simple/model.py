#
# model for CNN with MNIST
#

import tensorflow as tf


class Model:
    class ModelError(ValueError):
        pass

    def __init__(self):
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
        ## place holder
        self.__images = tf.placeholder(tf.float32, shape=[None, 28 * 28], name='images')
        self.__labels = tf.placeholder(tf.float32, shape=[None, 10], name='labels')
        ## key
        self.__train_step = None
        self.__accuracy = None


    def init_model(self):
        ## input
        imgs = tf.reshape(self.__images,[-1,28,28,1])
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
            tf.nn.softmax_cross_entropy_with_logits(labels=self.__labels, logits=full2))
        self.__train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(full2,1), tf.argmax(self.__labels, 1))
        self.__accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return 0


    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)
    def bias_variable(self,shape):
        initial = tf.constant(0.1,shape = shape)
        return tf.Variable(initial)

    def conv2D(self,x,W):
        return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
    def max_pool(self,x, move=2):
        return tf.nn.max_pool(x,ksize=[1,move,move,1],strides=[1,move,move,1],padding='SAME')

    @property
    def train_step(self):
        if self.__train_step is None:
            self.init_model()
        return self.__train_step
    @train_step.setter
    def train_step(self,value):
        raise self.ModelError('You try to change train_step')

    @property
    def accuracy(self):
        if self.__accuracy is None:
            self.init_model()
        return self.__accuracy
    @accuracy.setter
    def accuracy(self,value):
        raise self.ModelError('You try to change accuracy')

    @property
    def images(self):
        return self.__images
    @images.setter
    def images(self,value):
        self.__images = value

    @property
    def labels(self):
        return self.__labels
    @labels.setter
    def labels(self,value):
        self.__labels = value




