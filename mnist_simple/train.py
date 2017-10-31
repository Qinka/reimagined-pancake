from mnist_simple import model_interface
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def train_model(Modeler,mnist,times=1000,batch_size=50,target='',args={},summary_dir='.ignore/train'):
    #if not issubclass(Modeler,model_interface.ModelInterface):
    #    raise model_interface.ModelError('Error Modeler')
    with tf.Session(target) as sess:
        m = Modeler()
        m.init_model()
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(summary_dir,sess.graph)
        merged = tf.summary.merge_all()
        tf.summary.scalar('accuracy',m.accuracy)
        for i in range (times):
            batch = mnist.train.next_batch(batch_size)
            if i % 10 == 0:
                acc,summary = sess.run([m.accuracy,merged],feed_dict = make_train_feed(m,batch,args))
                summary_writer.add_summary(summary,i)
                if i % 100 == 0:
                    print('%d: %g' % (i, acc))
            m.train_step.run(
                feed_dict=make_train_feed(m,batch,args))
        acc, summary = sess.run([m.accuracy, merged],
                                feed_dict=make_train_feed(m, [mnist.test.images,mnist.test.labels], args))
        summary_writer.add_summary(summary,times)
        print('test accuracy %g' % acc)
        summary_writer.close()

def make_train_feed(m,batch,args={}):
    dic = {}
    dic[getattr(m,'images')] = batch[0]
    dic[getattr(m,'labels')] = batch[1]
    for k in args:
        if hasattr(m,k):
            dic[getattr(m,k)] = args[k]
    return dic

def get_mnist(train_dir = '.ignore'):
    return input_data.read_data_sets(train_dir=train_dir,one_hot=True)