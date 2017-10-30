# The model of the common
import tensorflow as tf

class ModelError(ValueError):
    """
    The error for model
    """
    pass

class ModelInterface(object):
    def __init__(self):
        ## place holder
        self.__images = tf.placeholder(tf.float32, shape=[None, 28 * 28], name='images')
        self.__labels = tf.placeholder(tf.float32, shape=[None, 10], name='labels')
        ## key
        self.__train_step = None
        self.__accuracy = None

    def init_model(self):
        pass
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
        self.__train_step = value

    @property
    def accuracy(self):
        if self.__accuracy is None:
            self.init_model()
        return self.__accuracy
    @accuracy.setter
    def accuracy(self,value):
        self.__accuracy = value

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