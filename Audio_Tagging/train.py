import dataloader
from dataloader import generate_in_background
import numpy as np
np.random.seed(101)
import os
import utils
import keras
from argparse import ArgumentParser
import config
import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import label_ranking_average_precision_score
import keras.backend as K
import tensorflow as tf
import sys

def opts_parser():
    descr = "Trains a neural network."
    parser = ArgumentParser(description=descr)
    parser.add_argument('modelfile', metavar='MODELFILE',
            type=str,
            help='File to save the learned weights to')
    config.prepare_argument_parser(parser)
    return parser

def save_learning_curve(metric, val_metric, filename, title, ylabel):
    plt.plot(metric)
    plt.plot(val_metric)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid()
    plt.ylim([0, 1.05])
    plt.savefig('plots/' + filename)
    plt.close()

def lwlrap_metric(y_true, y_pred):
    sample_weight = np.sum(y_true > 0, axis=1)
    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
    overall_lwlrap = label_ranking_average_precision_score(
        y_true[nonzero_weight_sample_indices, :] > 0,
        y_pred[nonzero_weight_sample_indices, :],
        sample_weight=sample_weight[nonzero_weight_sample_indices])
    return overall_lwlrap

def save_model_params(modelfile, cfg):
    """
    Saves the learned weights to `filename`, and the corresponding
    configuration to ``os.path.splitext(filename)[0] + '.vars'``.
    """
    config.write_config_file(modelfile + '_auto.vars', cfg)

def main():
    parser = opts_parser()
    options = parser.parse_args()
    modelfile = options.modelfile
    cfg = config.from_parsed_arguments(options)

    # keras configurations
    keras.backend.set_image_data_format('channels_last')

    for fold in range(1, 5):
        train_files = []
        train_files_noisy = []
        eval_files = []
        with open('../datasets/cv/fold{}_curated_train'.format(fold), 'r') as in_file:
            train_files.extend(in_file.readlines())
        for f in range(1, 5):
            with open('../datasets/cv/fold{}_noisy_eval'.format(fold), 'r') as in_file:
                train_files_noisy.extend(in_file.readlines())

        with open('../datasets/cv/fold{}_curated_eval'.format(fold), 'r') as in_file:
            eval_files.extend(in_file.readlines())

        print('Loading model')
        # import model from file
        selected_model = utils.import_model(modelfile)

        # instantiate neural network
        print("Preparing training function...")

        train_formats = (cfg['feature_height'], cfg['feature_width'], cfg['channels'])
        network = selected_model.architecture(train_formats, cfg['num_classes'])

        # Add optimizer and compile model
        print("Compiling model ...")
        optimizer = keras.optimizers.Adam(lr=cfg['lr'])
        if 'IIC' in modelfile:
            network.compile(optimizer=optimizer, loss={'clf_output':cfg['loss']}, metrics=['acc'])
        else:
            network.compile(optimizer=optimizer, loss=cfg["loss"], metrics=['acc'])

        print("Preserving architecture and configuration ..")
        if not os.path.exists('models/{}'.format(cfg['features'])):
            os.makedirs('models/{}'.format(cfg['features']))
        save_model_params(os.path.join('models/{}'.format(cfg['features']), modelfile.replace('.py', '')) + '_fold{}'.format(fold), cfg)

        # Add batch creator, and training procedure
        val_loss = []
        val_acc = []
        train_loss = []
        train_acc = []
        epochs_without_decrase = 0
        lwlraps_eval = []
        lwlraps_train = []
        lr_decay = K.get_value(network.optimizer.lr)/(cfg['epochs']-cfg['start_linear_decay']+1)
        switch_train_set = cfg['switch_train_set']
        lr = cfg['lr']

        # run training loop
        print("Training:")
        for epoch in range(1, cfg['epochs']+1):

            epoch_train_loss = []
            epoch_train_acc = []
            batch_val_loss = []
            batch_val_acc = []
            epoch_lwlrap_train = []
            epoch_lwlrap_eval = []

            if (epoch % switch_train_set) == 0:
                steps_per_epoch = len(train_files_noisy)//cfg['batchsize']
            else:
                steps_per_epoch = len(train_files) // cfg['batchsize']


            train_batches = generate_in_background(dataloader.load_batches(train_files, cfg['batchsize'], shuffle=True, infinite=True,
                                                    features=cfg['features'], feature_width=cfg['feature_width'],
                                                    fixed_length=cfg['fixed_size']), num_cached=100)
            train_noisy_batches = generate_in_background(dataloader.load_batches(train_files_noisy, cfg['batchisze'], shuffle=True,
                                                          infinite=True, feature_width=cfg['feature_width'], features=cfg['features'],
                                                          fixed_length=cfg['fixed_size']), num_cached=100)
            eval_batches = generate_in_background(dataloader.load_batches(eval_files, cfg['batchsize'], infinite=False, features=cfg['features'],
                                                   feature_width=cfg['feature_width'], fixed_length=cfg['fixed_size'],
                                                                          augment=False), num_cached=100)


            for _ in tqdm.trange(
                    steps_per_epoch,
                    desc='Epoch %d/%d:' % (epoch, cfg['epochs'])):

                if (epoch % switch_train_set) == 0:
                    X_train, y_train = next(train_noisy_batches)
                    noisy_lr = K.get_value(network.optimizer.lr)/100
                    K.set_value(network.optimizer.lr, noisy_lr)
                else:
                    K.set_value(network.optimizer.lr, lr)
                    X_train, y_train = next(train_batches)

                metrics = network.train_on_batch(x=X_train, y=y_train)
                epoch_train_acc.append(metrics[1])
                epoch_train_loss.append(metrics[0])

                preds = network.predict(x=X_train, batch_size=cfg['batchsize'], verbose=0)

                epoch_lwlrap_train.append(lwlrap_metric(np.asarray(y_train), np.asarray(preds)))

            print('Loss on training set after epoch {}: {}'.format(epoch, np.mean(epoch_train_loss)))
            print('Accuracy on training set after epoch {}: {}\n'.format(epoch, np.mean(epoch_train_acc)))
            print('Label weighted label ranking average precision on training set after epoch {}: {}'.format(epoch,
                                                                                                             np.mean(epoch_lwlrap_train)))

            train_loss.append(np.mean(epoch_train_loss))
            train_acc.append(np.mean(epoch_train_acc))
            lwlraps_train.append(np.mean(epoch_lwlrap_train))

            for X_test, y_test in tqdm.tqdm(eval_batches, desc='Batch'):

                metrics = network.test_on_batch(x=X_test, y=y_test)

                batch_val_loss.append(metrics[0])
                batch_val_acc.append(metrics[1])

                predictions = network.predict(x=X_test, batch_size=cfg['batchsize'], verbose=0)

                epoch_lwlrap_eval.append(lwlrap_metric(np.asarray(y_test), np.asarray(predictions)))


            val_acc.append(np.mean(batch_val_acc))
            val_loss.append(np.mean(batch_val_loss))

            print('Loss on validation set after epoch {}: {}'.format(epoch, np.mean(batch_val_loss)))
            print('Accuracy on validation set after epoch {}: {}'.format(epoch, np.mean(batch_val_acc)))
            print('Label weighted label ranking average precision on validation set after epoch {}: {}'.format(epoch,
                                                                                                   np.mean(epoch_lwlrap_eval)))
            current_lwlrap = np.mean(epoch_lwlrap_eval)

            if epoch > 1:
                if current_lwlrap > np.max(lwlraps_eval):
                    epochs_without_decrase = 0
                    print("Average lwlrap increased - Saving weights...\n")
                    network.save("models/{}/{}_fold{}.hd5".format(cfg['features'], modelfile.replace('.py', ''), fold))
                elif not cfg['linear_decay']:
                    epochs_without_decrase += 1
                    if epochs_without_decrase == cfg['epochs_without_decrease']:
                        lr = K.get_value(network.optimizer.lr)
                        lr = lr * cfg['lr_decrease']
                        K.set_value(network.optimizer.lr, lr)
                        print("lwlrap did not increase for the last {} epochs - halfing learning rate...".format(
                            cfg['epochs_without_decrease']))
                        epochs_without_decrase = 0

                if cfg['linear_decay']:
                    if epoch >= cfg['start_linear_decay']:
                        lr = keras.backend.get_value(network.optimizer.lr)
                        lr = lr - lr_decay
                        keras.backend.set_value(network.optimizer.lr, lr)
                        print("Decreasing learning rate by {}...".format(lr_decay))

            else:
                print("Average lwlrap increased - Saving weights...\n")
                network.save("models/{}/{}_fold{}.hd5".format(cfg['features'], modelfile.replace('.py', ''), fold))

            lwlraps_eval.append(np.mean(epoch_lwlrap_eval))

            if not os.path.exists('plots/{}'.format(cfg['features'])):
                os.makedirs('plots/{}'.format(cfg['features']))

            # Save loss and learning curve of trained model
            save_learning_curve(train_acc, val_acc, "{}/{}_fold{}_accuracy_learning_curve.pdf".format(cfg['features'], modelfile.replace('.py', ''), fold), 'Accuracy', 'Accuracy')
            save_learning_curve(train_loss, val_loss, "{}/{}_fold{}_loss_curve.pdf".format(cfg['features'], modelfile.replace('.py', ''), fold), 'Loss Curve', 'Loss')
            save_learning_curve(lwlraps_train, lwlraps_eval, '{}/{}_fold{}_lwlrap_curve.pdf'.format(cfg['features'], modelfile.replace('.py', ''), fold),
                                "Label Weighted Label Ranking Average Precision", 'lwlrap')


if __name__ == '__main__':
    main()