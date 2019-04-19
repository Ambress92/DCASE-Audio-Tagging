import os
import tqdm
import datetime
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from dataloader import *
from sklearn.metrics import label_ranking_average_precision_score
import logging
logger = logging.getLogger()

NUM_CLASSES = 80

def save_model(modelfile, network, path='../models/'):
    """
    Saves the learned weights to `filename`, and the corresponding
    configuration to ``os.path.splitext(filename)[0] + '.vars'``.
    """
    network_yaml = network.to_yaml()
    with open(path+modelfile+'.yaml', 'w') as yaml_file:
        yaml_file.write(network_yaml)

def save_learning_curve(metric, val_metric, filename, title, ylabel):
    plt.plot(metric)
    plt.plot(val_metric)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid()
    plt.ylim([0, 1.05])
    plt.savefig(filename)
    plt.close()

def log(message):
    # outputs to Jupyter console
    print('{} {}'.format(datetime.datetime.now(), message))
    # outputs to file
    logger.info(message)

def setup_file_logger(log_file):
    hdlr = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

def calculate_overall_lwlrap_sklearn(truth, scores):
    """Calculate the overall lwlrap using sklearn.metrics.lrap."""
    # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.
    sample_weight = np.sum(truth > 0, axis=1)
    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
    overall_lwlrap = label_ranking_average_precision_score(
      truth[nonzero_weight_sample_indices, :] > 0,
      scores[nonzero_weight_sample_indices, :],
      sample_weight=sample_weight[nonzero_weight_sample_indices])
    return overall_lwlrap

def train_on_fold(log_path, model, train_fold, eval_fold, model_path,
                    feature_path, data_path):
    """
    Parameters
    ----------
    log_path : String
        Path to log file.
    model : function
        Function that builds a network based on a data format and the number
        of classes. Must return built model.
    train_fold : String
        Path to file containing names of training files.
    eval_fold : String
        Path to file containing names of evaluation files.
    model_path : String
        Path to directory in which model parameters should be stored.
    feature_path : String
        Path pointing to respective feature-folder.
    data_path : String
        Path pointing to `train_curated.csv` and `train_noisy.csv`.
    """
    num_epochs = 50
    epochsize = 497
    batchsize = 10
    epochs_without_decrease = 0
    patience = 20
    lr_decrease = 0.5
    lr = 0.001
    features = 'mel'

    # prep logger, train and evaluation files
    fold = [int(f) for f in train_fold if f.isdigit()]
    setup_file_logger(log_path)
    with open(train_fold, 'r') as in_file:
        train_files = in_file.readlines()
    with open(eval_fold, 'r') as in_file:
        eval_files = in_file.readlines()

    # instantiate neural network
    log('Preparing training function...')
    train_formats = (128, 348, 1)
    network = model(train_formats, NUM_CLASSES)
    # network = cpjku_2018_cnn(train_formats, num_classes)
    # tpu_model = tf.contrib.tpu.keras_to_tpu_model(network,
    # strategy=tf.contrib.tpu.TPUDistributionStrategy(
    # tf.contrib.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])))

    log('Compiling model...')
    optimizer = keras.optimizers.Adam(lr=lr)
    network.compile(optimizer=optimizer, loss='binary_crossentropy')

    log('Preserving architecture and configuration...')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    save_model(model.__name__+'_fold{}'.format(fold), network, model_path)

    # Add batch creator, and training procedure
    val_loss = []
    train_loss = []
    epochs_without_decrease = 0
    lwlraps_eval = []
    lwlraps_train = []

    # run training loop
    log('Training:')
    for epoch in range(num_epochs):
        epoch_train_loss = []
        epoch_lwlrap_train = []
        epoch_lwlrap_eval = []
        batch_val_loss = []

        train_batches = load_batches(train_files, batchsize, infinite=True, shuffle=True,
                                     feature_path=feature_path, data_path=data_path)
        eval_batches = load_batches(eval_files, batchsize, infinite=False, feature_path=feature_path, data_path=data_path)

        for _ in tqdm.trange(epochsize, desc='Epoch %d/%d:' % (epoch + 1, num_epochs)):
            X_train, y_train = next(train_batches)
            # X_train, y_train = load_features(batch, features=features, num_classes=NUM_CLASSES,
            #                                   feature_path=feature_path, data_path=data_path)
            # X_train = X_train[:,:,:,np.newaxis]

            metrics = network.train_on_batch(x=X_train, y=y_train)
            epoch_train_loss.append(metrics)

            preds = network.predict(X_train, batch_size=batchsize, verbose=0)
            epoch_lwlrap_train.append(calculate_overall_lwlrap_sklearn(np.asarray(y_train), np.asarray(preds)))

        log('Loss on training set after epoch {}: {}'.format(epoch, np.mean(epoch_train_loss)))
        train_loss.append(np.mean(epoch_train_loss))
        # epoch_lwlrap_train = calculate_overall_lwlrap_sklearn(np.asarray(truth), np.asarray(predictions))
        lwlraps_train.append(np.mean(epoch_lwlrap_train))
        log('Lwlrap on training set after epoch {}: {}'.format(epoch, np.mean(epoch_lwlrap_train)))

        predictions = []
        truth = []

        for X_test, y_test in tqdm.tqdm(eval_batches, desc='Batch'):
            # X_test, y_test = load_features(batch_valid, features=features, num_classes=NUM_CLASSES,
            #                                feature_path=feature_path, data_path=data_path)
            # X_test = X_test[:, :, :, np.newaxis]

            metrics = network.test_on_batch(x=X_test, y=y_test)

            batch_val_loss.append(metrics)
            preds = network.predict(X_test, batch_size=batchsize, verbose=0)
            predictions.extend(preds)
            truth.extend(y_test)

        epoch_lwlrap_eval = calculate_overall_lwlrap_sklearn(np.asarray(truth), np.asarray(predictions))
        lwlraps_eval.append(epoch_lwlrap_eval)
        val_loss.append(np.mean(batch_val_loss))

        log('Loss on validation set after epoch {}: {}'.format(epoch, np.mean(batch_val_loss)))
        log('Lwlrap on validation set after epoch {}: {}'.format(epoch, epoch_lwlrap_eval))

        current_loss = np.mean(batch_val_loss)

        if epoch > 0:
            if epoch_lwlrap_eval > np.amax(lwlraps_eval):
                epochs_without_decrease = 0
                log('Average lwlrap increased - Saving weights...\n')
                network.save_weights('{}.hd5'.format(model_path+model.__name__))
            else:
                epochs_without_decrease += 1
                if epochs_without_decrease == patience:
                    lr = K.get_value(network.optimizer.lr)
                    lr = lr * lr_decrease
                    K.set_value(network.optimizer.lr, lr)
                    log('lwlrap did not increase for the last {} epochs - halfing learning rate...'.format(
                      patience))
                    epochs_without_decrease = 0
        else:
            log('Average lwlrap increased - Saving weights...\n')
            network.save_weights('{}_fold{}.hd5'.format(model_path+model.__name__, fold))

        # Save loss and learning curve of trained model
        # save_learning_curve(train_acc, val_acc, "gdrive/My Drive/models/{}_fold{}_accuracy_learning_curve.pdf".format(model, fold), 'Accuracy', 'Accuracy')
        save_learning_curve(train_loss, val_loss, '{}_fold{}_loss_curve.pdf'.format(model_path+model.__name__, fold), 'Loss Curve', 'Loss')
        save_learning_curve(lwlraps_train, lwlraps_eval, '{}_fold{}_lwlrap_curve.pdf'.format(model_path+model.__name__, fold),
                            'Label Weighted Label Ranking Average Precision', 'lwlrap')
