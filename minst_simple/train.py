
import model
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def train(mnist,times=1000,batch_size=50):
    with tf.Session() as sess:
        m = model.Model()
        m.init_model()
        sess.run(tf.global_variables_initializer())
        for i in range (times):
            batch = mnist.train.next_batch(batch_size)
            if i % 100 == 0:
                train_accuracy = m.accuracy.eval(
                    feed_dict = {
                        m.images : batch[0],
                        m.labels : batch[1]
                    })
                print('%d: %g' % (i, train_accuracy))
            m.train_step.run(
                feed_dict={
                    m.images : batch[0],
                    m.labels : batch[1]
                })
        test_accuracy = m.accuracy.eval(
            feed_dict = {
                m.images : mnist.test.images,
                m.labels : mnist.test.labels
            })
        print('test accuracy %g' % test_accuracy)

def get_mnist(train_dir):
    return input_data.read_data_sets(train_dir=train_dir,one_hot=True)