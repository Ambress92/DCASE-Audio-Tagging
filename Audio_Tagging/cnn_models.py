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

    Parameters
    ----------
    model_name : String
        Defines which model we want to train. Currently only supports 'baseline'.
    features :

    training : Bool
        True iff model is trained.
    hparams :
        Hyperparameters of model.
    num_classes : int
        Number of possible tags.
    labels :


    Return
    ------
    global_step : Tensor
        Tensor containing global step.
    prediction : Tensor
        Tensor containing predictions from classifier layer.
    loss : Tensor
        Tensor containing loss for each batch.
    train_op :

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
    features :


    Return
    ------
    net :

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
    hparams :

    train_clip_dir :

    train_csv_path :

    train_dir :

    Return
    ------

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
                save_learning_curve(step, loss)
                sys.stdout.flush()


###### main ######

def parse_flags():
    """
    Parses and returns input flags.
    """
    parser = argparse.ArgumentParser()

    # common flags
    all_modes_group = parser.add_argument_group('Common flags')
    all_modes_group.add_argument('--mode', type=str, choices=['train', 'eval', 'inference'],
        required=True, help='Run one of training, evaluation, or inference.')
    all_modes_group.add_argument('--model', type=str, choices=['baseline'],
        default='baseline', required=False,
        help='Name of a model architecture. Currently, only "baseline" possible.')
    all_modes_group.add_argument('--hparams', type=str, default='',
        help='Model hyperparameters in comma-separated name=value format.')
    all_modes_group.add_argument('--year', type=int, default=2018, required=False,
        help='Year of the data we want to process')

    # flags for training
    training_group = parser.add_argument_group('Flags for training only')
    training_group.add_argument('--train_clip_dir', type=str,
        default='../datasets/2018/audio_train/',
        help='Path to training-clips-directory.')
    training_group.add_argument('--train_csv_path', type=str,
        default='../datasets/2018/train.csv',
        help='Path to CSV file containing training clip filenames and labels.')
    training_group.add_argument('--train_dir', type=str, default='../models/',
        help='Path to a directory which will hold model checkpoints and other outputs.')

    flags = parser.parse_args()

    try:
        if flags.mode == 'train':
            assert flags.train_clip_dir, 'Must specify --train_clip_dir'
            assert flags.train_csv_path, 'Must specify --train_csv_path'
            assert flags.train_dir, 'Must specify --train_dir'
    except AssertionError as e:
        print('\nError: ', e, '\n')
        parser.print_help()
        sys.exit(1)

    return flags

def parse_hparams(flag_hparams):
    """
    Parses and returns hyperparameters of specified model.
    """

    # defaul values
    hparams = tf.contrib.training.HParams(
        # window and hop length for STFT
        stft_window_seconds=0.025,
        stft_hop_seconds=0.010,
        # spectrogram to mel spectrogram parameters
        mel_bands=64,
        mel_min_hz=125,
        mel_max_hz=7500,
        # log mel spectrogram = log(mel-spectrogram + mel_log_offset)
        mel_log_offset=0.001,
        # window and hop length to frame the log mel spectrogram into examples
        example_window_seconds=0.250,
        example_hop_seconds=0.125,
        # number of examples in a batch
        batch_size=64,
        # for MLP, nl=# layers, nh=# units per layer
        #nl=2,
        #nh=256,
        # SD of normal distribution to initialize the weights of the model
        weights_init_stddev=1e-3,
        # learning rate
        lr=1e-4,
        # epsilon for Adam optimiser
        adam_eps=1e-8,
        # classifier layer: softmax or logistic
        classifier='softmax')

    # flags can override default hparam values
    hparams.parse(flag_hparams)

    return hparams

def main():
    # parse flags and hyperparameters
    flags = parse_flags()
    hparams = parse_hparams(flags.hparams)

    if flags.mode == 'train':
        train(model_name=flags.model, year=flags.year, hparams=hparams,
              train_csv_path=flags.train_csv_path,
              train_clip_dir=flags.train_clip_dir,
              train_dir=flags.train_dir)


if __name__ == '__main__':
    main()
