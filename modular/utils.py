import os
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

"""
Utility functions for modular model, called in run.py
"""

# layer definitions
def conv(input, size_in, size_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def fc(input, size_in, size_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        act = tf.matmul(input, w) + b
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act

# for saving
def make_hparam_string(learning_rate, use_two_fc, use_two_conv):
    conv_param = "conv=2" if use_two_conv else "conv=1"
    fc_param = "fc=2" if use_two_fc else "fc=1"
    return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)
 
# for plotting
def plot(xs, ys, title, smooth_window = None, xlabel=None, ylabel=None, \
    ylegend=None, log=False):
    if smooth_window is None:
        smooth_window = int(len(xs)/10)
    if ylegend = None:
        ylegend = 'values'
    if log == True:
        ys = [math.log(y) for y in ys]
    plt.style.use('seaborn')
    ysmooth = smooth(ys, smooth_window)
    fig, ax = plt.subplots(figsize=(10,10))
    # smoothed line
    ax.plot(xs, ysmooth, '-', linewidth=2.0, label=ylegend, color='C1')
    # unsmoothed line (original)
    ax.plot(xs, ys, '-', linewidth=1.0, alpha=0.5, color='C1')
    #ax.plot(xs, ys2 + es2, '-', label='train')
    ax.set_title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # save figure to current directory
    pylab.savefig(title+'.png', bbox_extra_artists=(legend,), bbox_inches='tight')
    
def smooth(xs, width):
    assert width < len(xs)
    xsmoothed = []
    smooth = 0.
    w = 0
    for i in range(width):
        smooth = smooth*w + xs[i]
        w += 1
        smooth /= w
        xsmoothed.append(smooth)
    for i in range(width, len(xs)):
        smooth = smooth*w - xs[i-width] + xs[i]
        smooth /= w
        xsmoothed.append(smooth)
    return xsmoothed
