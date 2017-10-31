
from mnist_simple import *
import tensorflow as tf

target = 'grpc://etvp-400:36559'

if __name__ == '__main__':
    print('get MNIST data')
    mnist = get_mnist()
    print('Model train with ModelBase 1')
    train_model(ModelBase,mnist,target=target,summary_dir='.ignore/train/base/1')
    tf.reset_default_graph()
    print('Model train with ModelBase 2')
    train_model(ModelBase,mnist,target=target,summary_dir='.ignore/train/base/2')
    tf.reset_default_graph()
    print('Model train with ModelBase 3')
    train_model(ModelBase,mnist,target=target,summary_dir='.ignore/train/base/3')
    tf.reset_default_graph()
    print('Model train with ModelWithDrop 1')
    train_model(ModelWithDrop,mnist,target=target,args={'keep_prob':0.5},summary_dir='.ignore/train/drop/1')
    tf.reset_default_graph()
    print('Model train with ModelWithDrop 2')
    train_model(ModelWithDrop,mnist,target=target,args={'keep_prob':0.5},summary_dir='.ignore/train/drop/2')
    tf.reset_default_graph()
    print('Model train with ModelWithDrop 3')
    train_model(ModelWithDrop,mnist,target=target,args={'keep_prob':0.5},summary_dir='.ignore/train/drop/3')
    tf.reset_default_graph()