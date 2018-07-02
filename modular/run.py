import os
import shutil

import numpy as np
import tensorflow as tf

from utils import conv, fc, plot

"""
Run file for testing modularity-inducing regularization term in the toy example of MNIST.
Much code adopted from Tensorflow's Tensorboard tutorial, available at:

https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
"""

LOGDIR = '/tmp/modular/mnist'
cwd = os.getcwd()
LABELS = os.path.join(cwd, "labels_1024.tsv")
SPRITES = os.path.join(cwd, "sprite_1024.png")
# downlaod mnist
mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=LOGDIR + "data", one_hot=True)

# visualization
if not (os.path.isfile(LABELS) and os.path.isfile(SPRITES)):
    print("Necessary data files were not found. Run this command from inside the "
        "repo provided at "
        "https://github.com/dandelionmane/tf-dev-summit-tensorboard-tutorial.")

# run parametes
use_two_fc = True
use_two_conv = True
learning_rate_init = 1e-4
nbatches = int(1e5)
save_plots = True
# experiment with different values of regularization penalty
reg_coef = 0.05
loop_coef = 0.01
hparam = "test"

tf.reset_default_graph()
sess = tf.Session()

# set data placeholders
x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
x_image = tf.reshape(x, [-1, 28, 28, 1])
tf.summary.image('input', x_image, 3)
y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")

if use_two_conv:
    conv1 = conv(x_image, 1, 32, "conv1")
    conv_out = conv(conv1, 32, 64, "conv2")
else:
    conv_out = conv(x_image, 1, 16, "conv")

flattened = tf.reshape(conv_out, [-1, 7 * 7 * 64])


if use_two_fc:
    fc1 = fc(flattened, 7 * 7 * 64, 1024, "fc1")
    relu = tf.nn.relu(fc1)
    embedding_input = relu
    tf.summary.histogram("fc1/relu", relu)
    embedding_size = 1024
    logits = fc(relu, 1024, 10, "fc2")
else:
    embedding_input = flattened
    embedding_size = 7*7*64
    logits = fc(flattened, 7*7*64, 10, "fc")

with tf.name_scope("xent"):
    xent = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y), name="xent")
    tf.summary.scalar("xent", xent)

def l4_loss(t):
    # raise each element in tensor t to the fourth power
    # sum over all elements (all axis)
    axis = list(range(len(t.get_shape()))) 
    return tf.reduce_sum(tf.pow(t, 4), axis=axis)
    
with tf.name_scope("regularization"):
    graph = tf.get_default_graph()
    W1 = graph.get_tensor_by_name('conv1/W:0')
    W2 = graph.get_tensor_by_name('conv2/W:0')
    W3 = graph.get_tensor_by_name('fc1/W:0')
    W4 = graph.get_tensor_by_name('fc2/W:0')
    
    # l4-norm loss for all weight variables
    l4_weight_decay = l4_loss(W1)
    l4_weight_decay += l4_loss(W2)
    l4_weight_decay += l4_loss(W3)
    l4_weight_decay += l4_loss(W4)
    
    # loops between convolutional layers
    W1_spatial_sum = tf.reduce_sum(W1, [0, 1])
    W2_spatial_sum = tf.reduce_sum(W2, [0, 1])
    W1_2 = tf.tensordot(W1_spatial_sum, W2_spatial_sum, axes=[[1], [0]])
    loops1_2 = tf.tensordot(W1_2, W1_2, axes=[[1, 0], [0, 1]])
    
    # loops between fully connected layers
    W3_4 = tf.tensordot(W3, W4, axes=[[1], [0]])
    loops3_4 = tf.tensordot(W3_4, W3_4, axes=[[1, 0], [0, 1]])
    loops = loops1_2 + loops3_4
    # define regularization: weight decay + loop encouragement
    reg = l4_weight_decay - loop_coef * tf.sqrt(loops)
    tf.summary.scalar("loops", loops)
    tf.summary.scalar("l4_weight_decay", l4_weight_decay)

with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(learning_rate_init).minimize(xent)
    train_step_reg = tf.train.AdamOptimizer(learning_rate_init).minimize(xent + reg_coef * reg)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_summary = tf.summary.scalar("accuracy", accuracy)

summ = tf.summary.merge_all()

embedding = tf.Variable(tf.zeros([1024, embedding_size]), name="test_embedding")
assignment = embedding.assign(embedding_input)
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

# set different summary writers for train and test data
train_writer = tf.summary.FileWriter(LOGDIR + 'train')
test_writer = tf.summary.FileWriter(LOGDIR + 'test')
train_writer.add_graph(sess.graph)

config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
embedding_config = config.embeddings.add()
embedding_config.tensor_name = embedding.name
embedding_config.sprite.image_path = SPRITES
embedding_config.metadata_path = LABELS
# Specify the width and height of a single thumbnail.
embedding_config.sprite.single_image_dim.extend([28, 28])
tf.contrib.tensorboard.plugins.projector.visualize_embeddings(train_writer, config)

# main training function
def train():
    if save_plots:
        # keep values of data for plotting
        batch_list = []
        train_accuracy_list = []
        test_accuracy_list = []
        loops_list = []
        reg_list = []
        
    for i in range(nbatches):
        batch = mnist.train.next_batch(100)
        # training accuracy
        if i % 5 == 0:
            [train_accuracy, s] = sess.run([accuracy, summ], \
                feed_dict={x: batch[0], y: batch[1]})
            train_writer.add_summary(s, i)
        # test accuracy
        if i % 20 == 0:
            test_batch = mnist.test.next_batch(100)
            [test_acc, s, l, r] = sess.run([accuracy, accuracy_summary, loops, reg], \
                feed_dict={x: test_batch[0], y: test_batch[1]})
            test_writer.add_summary(s, i)
            print('batches: ', str(i), '; accuracy: ', str(test_acc))
            if save_plots:
                batch_list.append(i)
                train_accuracy_list.append(train_accuracy)
                test_accuracy_list.append(test_acc)
                loops_list.append(l)
                reg_list.append(r)
        if i % 500 == 0:
            sess.run(assignment, feed_dict={x: mnist.test.images[:1024], \
                y: mnist.test.labels[:1024]})
            saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)
        sess.run(train_step_reg, feed_dict={x: batch[0], y: batch[1]})
        
    if save_plots:
        # train accuracy
        plot(batch_list, train_accuracy_list, 'Training accuracy', \
            xlabel='number of batches', ylabel='training accuracy', \
            ylegend='training accuracy')
        # test accuracy
        plot(batch_list, test_accuracy_list, 'Test accuracy', \
            xlabel='number of batches', ylabel='test accuracy', \
            ylegend='test accuracy')
        # loops
        plot(batch_list, loops_list, 'Loop regularization term', \
            xlabel='number of batches', ylabel='loop regularization term magnitude', \
            ylegend='loop reg.')
        # loops with log
        plot(batch_list, loops_list, 'Loop regularization term (log scale)', \
            xlabel='number of batches', ylabel='loop regularization term magnitude', \
            ylegend='loop reg.', log=True)
        # regularization
        plot(batch_list, reg_list, 'Weight decay loss magnitude', \
            xlabel='number of batches', ylabel='weight decay loss magnitude', \
            ylegend='weight decay')

train()
sess.close()
