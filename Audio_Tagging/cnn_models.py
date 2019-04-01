import tensorflow as tf
import numpy as np
import argparse
import sys
from preprocess_data import train_input
import matplotlib.pyplot as plt

slim = tf.contrib.slim

# see https://github.com/DCASE-REPO/dcase2018_baseline/tree/master/task2

###### build models ######

def save_learning_curve(epoch, loss):
    plt.title('Learning curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(np.arange(0,epoch), loss)
    plt.gcf().savefig('plots/learning_curve.png')

def define_model(model_name, features=None, training=False, hparams=None,
             num_classes=None, labels=None):
    """
    Builds and prepares model we want to work with.

    Parameters
    ----------
    model_name : String
        Defines which model we want to train. Currently only supports 'baseline'.
    features : Tensor
        Tensor containing a batch of input features (log mel spectrum examples).
    training : Bool
        True iff model is trained.
    hparams : tf.contrib.training.HParams
        Hyperparameters of model.
    num_classes : int
        Number of possible tags.
    labels : Tensor
        Correct labels we need for training the model.

    Return
    ------
    global_step : Tensor
        Tensor containing global step.
    prediction : Tensor
        Tensor containing predictions from classifier layer.
    loss : Tensor
        Tensor containing loss for each batch.
    train_op : Optimizer
        Optimizer that runs training for batches.
    """

    global_step = tf.Variable(0, name='global_step', trainable=training,
                              collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                           tf.GraphKeys.GLOBAL_STEP])

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=hparams.weights_init_stddev),
                        biases_initializer=tf.zeros_initializer(),
                        activation_fn=tf.nn.relu,
                        trainable=training):
        # define model (yet without classification)
        if model_name == 'baseline':
            embedding = build_baseline_cnn(features=features)
        else:
            raise ValueError('Unknown model %s' % model)

        # add classification
        logits = slim.fully_connected(embedding, num_classes, activation_fn=None)
        if hparams.classifier == 'softmax':
            prediction = tf.nn.softmax(logits)
        else:
            raise ValueError('Bad classifier: %s' % classifier)

    if training:
        if hparams.classifier == 'softmax':
            xent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)

        loss = tf.reduce_mean(xent)
        tf.summary.scalar('loss', loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=hparams.lr,
                                           epsilon=hparams.adam_eps)
        train_op = optimizer.minimize(loss, global_step=global_step)
    else:
        loss = None
        train_op = None

    return global_step, prediction, loss, train_op


def build_baseline_cnn(features=None):
    """
    Defines a convolutional neural network model, without the classifier layer.

    Parameters
    ----------
    features : Tensor
        Tensor containing a batch of input features (log mel spectrum examples).

    Return
    ------
    net :
        Baseline CNN without classification layer.
    """

    net = tf.expand_dims(features, axis=3)
    with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'), \
         slim.arg_scope([slim.max_pool2d], stride=2, padding='SAME'):
        net = slim.conv2d(net, 100, kernel_size=[7, 7])
        net = slim.max_pool2d(net, kernel_size=[3, 3])
        net = slim.conv2d(net, 150, kernel_size=[5, 5])
        net = slim.max_pool2d(net, kernel_size=[3, 3])
        net = slim.conv2d(net, 200, kernel_size=[3, 3])
        net = tf.reduce_max(net, axis=[1,2], keepdims=True)
        net = slim.flatten(net)
    return net

###### train models ######

def train(model_name, year, hparams=None, train_clip_dir=None,
          train_csv_path=None, train_dir=None):
    """
    Trains neural network with defined hyperparameters.

    Parameters
    ----------
    model_name : String
        Defines which model we want to train. Currently only support 'baseline'.
    year : int
        Year we want to take the data from.
    hparams : tf.contrib.training.HParams
        Hyperparameters of the model we want to train.
    train_clip_dir : String
        Path to directory containing training data.
    train_csv_path : String
        Path to train.csv file.
    train_dir : String
        Path to store all checkpoints of our trained model.
    """

    print('\nTraining model:{} with hparams:{}'.format(model_name, hparams))
    print('Training data: clip dir {} and labels {}'.format(train_clip_dir, train_csv_path))
    print('Training dir {}\n'.format(train_dir))
    losses = []

    with tf.Graph().as_default():
        # prepare input
        features, labels, num_classes, input_init = train_input(year=year,
            train_csv_path=train_csv_path, train_clip_dir=train_clip_dir, hparams=hparams)
        # create model in training mode
        global_step, prediction, loss_tensor, train_op = define_model(model_name=model_name,
            features=features, labels=labels, num_classes=num_classes,
            hparams=hparams, training=True)

        # define own checkpoint saves
        saver = tf.train.Saver(max_to_keep=30, keep_checkpoint_every_n_hours=0.25)
        saver_hook = tf.train.CheckpointSaverHook(save_steps=250, checkpoint_dir=train_dir, saver=saver)

        summary_op = tf.summary.merge_all()
        summary_hook = tf.train.SummarySaverHook(save_steps=50,
            output_dir=train_dir, summary_op=summary_op)

        with tf.train.SingularMonitoredSession(hooks=[saver_hook, summary_hook],
                                           checkpoint_dir=train_dir) as sess:
            sess.raw_session().run(input_init)
            while not sess.should_stop():
                step, _, pred, loss = sess.run([global_step, train_op, prediction, loss_tensor])
                print(step, loss)
                losses.append(loss)
                # save_learning_curve(step, loss)
                sys.stdout.flush()

        save_learning_curve(np.arange(1,len(losses)+1), losses)
