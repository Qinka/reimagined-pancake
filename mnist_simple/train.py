from mnist_simple import model_interface
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def train_model(Modeler,mnist,times=1000,batch_size=50,target='',args={}):
    #if not issubclass(Modeler,model_interface.ModelInterface):
    #    raise model_interface.ModelError('Error Modeler')
    with tf.Session(target) as sess:
        m = Modeler()
        m.init_model()
        sess.run(tf.global_variables_initializer())
        for i in range (times):
            batch = mnist.train.next_batch(batch_size)
            if i % 100 == 0:
                train_accuracy = m.accuracy.eval(
                    feed_dict = make_train_feed(m,batch))
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

def make_train_feed(m,batch,argv={}):
    dic = {}
    dic[getattr(m,'images')] = batch[0]
    dic[getattr(m,'labels')] = batch[1]
    for k,v in argv:
        dic[getattr(m,k)] = v
    return dic

def get_mnist(train_dir):
    return input_data.read_data_sets(train_dir=train_dir,one_hot=True)