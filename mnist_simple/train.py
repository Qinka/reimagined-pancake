from mnist_simple import model_interface
import tensorflow as tf
import signal
import os
from tensorflow.examples.tutorials.mnist import input_data
from mnist_simple import ModelInterface

def restore_reload_model(sess:tf.Session,path):
    d, f = os.path.split(path)
    if not os.path.exists(d):
        os.makedirs(d)
    saver = tf.train.Saver()#m.get_variables())
    def signal_handler(signum, frame):
        if signum in [signal.SIGINT, signal.SIGABRT, signal.SIGKILL]:
            print('catch signal and store')
            saver.save(sess,path)
            os._exit(0)
    signal.signal(signal.SIGINT,signal_handler)
    signal.signal(signal.SIGABRT,signal_handler)
    #signal.signal(signal.SIGKILL,signal_handler)
    print (saver.last_checkpoints)
    if path in saver.last_checkpoints:
        print('restore')
        saver.restore(sess,path)
    return saver

def store_model(saver:tf.train.Saver,sess:tf.Session,path,step):
    saver.save(sess,path,global_step=step)
    saver.save(sess,path)




def train_model(Modeler,mnist,times=1000,batch_size=50,target='',args={},summary_dir='.ignore/train',path='./model'):
    #if not issubclass(Modeler,model_interface.ModelInterface):
    #    raise model_interface.ModelError('Error Modeler'
    tf.reset_default_graph()
    with tf.Session(target) as sess:
        m = Modeler()
        m.init_model()
        sess.run(tf.global_variables_initializer())
        saver = restore_reload_model(sess,path)
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
                    store_model(saver,sess,path,i)

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
