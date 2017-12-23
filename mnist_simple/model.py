#
# model for CNN with MNIST
#

import tensorflow as tf
from mnist_simple import ModelInterface, ModelError

class ModelBase(ModelInterface):
    def __init__(self):
        super().__init__()
        with tf.name_scope('conv1'):
            self.__w1 = self.weight_variable([5, 5, 1, 16])
            self.__b1 = self.bias_variable([16])
        with tf.name_scope('conv2'):
            self.__w2 = self.weight_variable([5, 5, 16, 32])
            self.__b2 = self.bias_variable([32])
        with tf.name_scope('fc1'):
            self.__w3 = self.weight_variable([7 * 7 * 32, 1024])
            self.__b3 = self.bias_variable([1024])
        with tf.name_scope('fc2'):
            self.__w4 = self.weight_variable([1024, 10])
            self.__b4 = self.bias_variable([10])

    def init_model(self):
        with tf.name_scope('images'):
            imgs = tf.reshape(self.images,[-1,28,28,1])
        with tf.name_scope('conv1'):
            conv1 = tf.nn.relu(self.conv2D(imgs, self.__w1) + self.__b1)
            pool1 = self.max_pool(conv1,2)
        with tf.name_scope('conv2'):
            conv2 = tf.nn.relu(self.conv2D(pool1, self.__w2) + self.__b2)
            pool2 = self.max_pool(conv2,2)
        with tf.name_scope('fc1'):
            full1_in = tf.reshape(pool2,[-1,7*7*32])
            full1 = tf.nn.relu(tf.matmul(full1_in, self.__w3) + self.__b3)
        with tf.name_scope('fc2'):
            full2 = tf.nn.relu(tf.matmul(full1, self.__w4) + self.__b4)
        with tf.name_scope('loss'):
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=full2))
            tf.summary.scalar('cross_entropy',cross_entropy)
        with tf.name_scope('optimizer_adam'):
            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(full2,1), tf.argmax(self.labels, 1))
            correct_prediction = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy',correct_prediction)
            self.accuracy = correct_prediction


class ModelWithDrop(ModelInterface):
    def __init__(self):
        super().__init__()
        with tf.name_scope('conv1'):
            self.__w1 = self.weight_variable([5, 5, 1, 16])
            self.__b1 = self.bias_variable([16])
        with tf.name_scope('conv2'):
            self.__w2 = self.weight_variable([5, 5, 16, 32])
            self.__b2 = self.bias_variable([32])
        with tf.name_scope('fc1'):
            self.__w3 = self.weight_variable([7 * 7 * 32, 1024])
            self.__b3 = self.bias_variable([1024])
        with tf.name_scope('drop_out'):
            self.__keep_prob = tf.placeholder(tf.float32)
        with tf.name_scope('fc2'):
            self.__w4 = self.weight_variable([1024, 10])
            self.__b4 = self.bias_variable([10])

    def init_model(self):
        with tf.name_scope('images'):
            imgs = tf.reshape(self.images,[-1,28,28,1])
        with tf.name_scope('conv1'):
            conv1 = tf.nn.relu(self.conv2D(imgs, self.__w1) + self.__b1)
            pool1 = self.max_pool(conv1,2)
        with tf.name_scope('conv2'):
            conv2 = tf.nn.relu(self.conv2D(pool1, self.__w2) + self.__b2)
            pool2 = self.max_pool(conv2,2)
        with tf.name_scope('fc1'):
            full1_in = tf.reshape(pool2,[-1,7*7*32])
            full1 = tf.nn.relu(tf.matmul(full1_in, self.__w3) + self.__b3)
        with tf.name_scope('drop_out'):
            full1d = tf.nn.dropout(full1,self.keep_prob)
        with tf.name_scope('fc2'):
            full2 = tf.nn.relu(tf.matmul(full1d, self.__w4) + self.__b4)
        with tf.name_scope('loss'):
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=full2))
            tf.summary.scalar('cross_entropy',cross_entropy)
        with tf.name_scope('optimizer_adam'):
            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(full2,1), tf.argmax(self.labels, 1))
            correct_prediction = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy',correct_prediction)
            self.accuracy = correct_prediction

    @property
    def keep_prob(self):
        return self.__keep_prob
    @keep_prob.setter
    def keep_prob(self,value):
        self.__keep_prob = value


class ModelNormalA(ModelInterface):
    def __init__(self):
        super().__init__()
        with tf.name_scope('conv1'):
            self.__w1 = self.weight_variable([5, 5, 1, 16])
            self.__b1 = self.bias_variable([16])
        with tf.name_scope('conv2'):
            self.__w2 = self.weight_variable([5, 5, 16, 32])
            self.__b2 = self.bias_variable([32])
        with tf.name_scope('fc1'):
            self.__w3 = self.weight_variable([7 * 7 * 32, 1024])
            self.__b3 = self.bias_variable([1024])
        with tf.name_scope('fc2'):
            self.__w4 = self.weight_variable([1024, 10])
            self.__b4 = self.bias_variable([10])

    def init_model(self):
        with tf.name_scope('normal'):
            mean,variance = tf.nn.moments(self.images,[0])
            imgs = tf.nn.batch_normalization(self.images,mean,variance,0,1,0.00001)
        with tf.name_scope('images'):
            imgs = tf.reshape(imgs,[-1,28,28,1])
        with tf.name_scope('conv1'):
            conv1 = tf.nn.relu(self.conv2D(imgs, self.__w1) + self.__b1)
            pool1 = self.max_pool(conv1,2)
        with tf.name_scope('conv2'):
            conv2 = tf.nn.relu(self.conv2D(pool1, self.__w2) + self.__b2)
            pool2 = self.max_pool(conv2,2)
        with tf.name_scope('fc1'):
            full1_in = tf.reshape(pool2,[-1,7*7*32])
            full1 = tf.nn.relu(tf.matmul(full1_in, self.__w3) + self.__b3)
        with tf.name_scope('fc2'):
            full2 = tf.nn.relu(tf.matmul(full1, self.__w4) + self.__b4)
        with tf.name_scope('loss'):
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=full2))
            tf.summary.scalar('cross_entropy',cross_entropy)
        with tf.name_scope('optimizer_adam'):
            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(full2,1), tf.argmax(self.labels, 1))
            correct_prediction = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy',correct_prediction)
            self.accuracy = correct_prediction


class ModelWithDropExp(ModelInterface):
    def __init__(self):
        super().__init__()
        with tf.name_scope('conv1'):
            self.__w1 = self.weight_variable([5, 5, 1, 16])
            self.__b1 = self.bias_variable([16])
        with tf.name_scope('conv2'):
            self.__w2 = self.weight_variable([5, 5, 16, 32])
            self.__b2 = self.bias_variable([32])
        with tf.name_scope('fc1'):
            self.__w3 = self.weight_variable([7 * 7 * 32, 1024])
            self.__b3 = self.bias_variable([1024])
        with tf.name_scope('drop_out'):
            self.__keep_prob = tf.placeholder(tf.float32)
        with tf.name_scope('fc2'):
            self.__w4 = self.weight_variable([1024, 10])
            self.__b4 = self.bias_variable([10])

    def init_model(self):
        with tf.name_scope('images'):
            imgs = tf.reshape(self.images,[-1,28,28,1])
        with tf.name_scope('conv1'):
            conv1 = self.xpso(self.conv2D(imgs, self.__w1) + self.__b1,0.1)
            pool1 = self.max_pool(conv1,2)
        with tf.name_scope('conv2'):
            conv2 = self.xpso(self.conv2D(pool1, self.__w2) + self.__b2,0.1)
            pool2 = self.max_pool(conv2,2)
        with tf.name_scope('fc1'):
            full1_in = tf.reshape(pool2,[-1,7*7*32])
            full1 = self.xpso(tf.matmul(full1_in, self.__w3) + self.__b3,0.1)
        with tf.name_scope('drop_out'):
            full1d = tf.nn.dropout(full1,self.keep_prob)
        with tf.name_scope('fc2'):
            full2 = self.xpso(tf.matmul(full1d, self.__w4) + self.__b4,2)
        with tf.name_scope('loss'):
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=full2))
            tf.summary.scalar('cross_entropy',cross_entropy)
        with tf.name_scope('optimizer_adam'):
            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(full2,1), tf.argmax(self.labels, 1))
            correct_prediction = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy',correct_prediction)
            self.accuracy = correct_prediction

    @property
    def keep_prob(self):
        return self.__keep_prob
    @keep_prob.setter
    def keep_prob(self,value):
        self.__keep_prob = value

    @staticmethod
    def xpso(x,y=2):
        with tf.name_scope('PReLU'):
            return tf.pow(tf.nn.relu(x),y)