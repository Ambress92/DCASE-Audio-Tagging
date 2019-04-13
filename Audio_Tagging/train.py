import argparse
import dataloader
import numpy as np
import os
import utils
import keras
from argparse import ArgumentParser
import config
import tqdm
from evaluate import calculate_overall_lwlrap_sklearn
import matplotlib.pyplot as plt

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
    plt.grid("on")
    plt.ylim([0, 1.05])
    plt.savefig('plots/' + filename)
    plt.close()


def save_model(modelfile, network, cfg):
    """
    Saves the learned weights to `filename`, and the corresponding
    configuration to ``os.path.splitext(filename)[0] + '.vars'``.
    """
    config.write_config_file(modelfile + '_auto.vars', cfg)
    network_yaml = network.to_yaml()
    with open(modelfile+".yaml", 'w') as yaml_file:
        yaml_file.write(network_yaml)

def main():
    label_mapping, inv_label_mapping = dataloader.get_label_mapping()
    parser = opts_parser()
    options = parser.parse_args()
    modelfile = options.modelfile
    cfg = config.from_parsed_arguments(options)
    keras.backend.set_image_data_format('channels_last')
    clf_threshold=0.95

    verified_files_dict = dataloader.get_verified_files_dict()
    noisy_files_dict = dataloader.get_unverified_files_dict()
    total_files_dict = dict(verified_files_dict, **noisy_files_dict)
    fold = 1

    print('Loading data...')
    #for fold in range(1,5):
    train_files = []
    eval_files = []
    with open('../datasets/cv/fold{}_curated_train'.format(fold), 'r') as in_file:
        train_files.extend(in_file.readlines())
    with open('../datasets/cv/fold{}_noisy_train'.format(fold), 'r') as in_file:
        train_files.extend(in_file.readlines())

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
    network.compile(optimizer=optimizer, loss=cfg["loss"], metrics=["acc"])

    print("Preserving architecture and configuration ..")
    save_model(os.path.join('models', modelfile.replace('.py', '')), network, cfg)

    # Add batch creator, and training procedure
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    epochs_without_decrase = 0
    lwlraps_eval = []
    lwlraps_train = []

    # run training loop
    print("Training:")
    for epoch in range(cfg['epochs']):

        epoch_train_loss = []
        epoch_train_acc = []
        batch_val_loss = []
        batch_val_acc = []

        train_batches = dataloader.load_batches(train_files, cfg['batchsize'], infinite=True, shuffle=True)
        train_eval_batches = dataloader.load_batches(train_files, cfg['batchsize'], infinite=False)
        eval_batches = dataloader.load_batches(eval_files, cfg['batchsize'], infinite=False)

        for _ in tqdm.trange(
                cfg['epochsize'],
                desc='Epoch %d/%d:' % (epoch + 1, cfg['epochs'])):

            batch = next(train_batches)
            X_train, y_train = dataloader.load_features(batch, features='mel', num_classes=cfg['num_classes'])
            X_train = X_train[:,:,:,np.newaxis]

            metrics = network.train_on_batch(x=X_train, y=y_train)
            epoch_train_acc.append(metrics[1])
            epoch_train_loss.append(metrics[0])

        print('Loss on training set after epoch {}: {}'.format(epoch, np.mean(epoch_train_loss)))
        print('Accuracy on training set after epoch {}: {}\n'.format(epoch, np.mean(epoch_train_acc)))
        train_loss.append(np.mean(epoch_train_loss))
        train_acc.append(np.mean(epoch_train_acc))

        predictions = []
        truth = []

        for batch_valid in tqdm.tqdm(train_eval_batches, desc='Batch'):
            X_train, y_train = dataloader.load_features(batch_valid, features='mel', num_classes=cfg['num_classes'])
            X_train = X_train[:, :, :, np.newaxis]

            preds = network.predict(X_train, batch_size=cfg['batchsize'], verbose=0)
            preds = dataloader.one_hot_encode(np.nonzero(preds > clf_threshold)[0])
            predictions.extend(preds)
            truth.extend(y_train)

        epoch_lwlrap_train = calculate_overall_lwlrap_sklearn(np.asarray(predictions), np.asarray(truth))
        lwlraps_train.append(epoch_lwlrap_train)

        print('Label weighted label ranking average precision on training set after epoch {}: {}'.format(epoch,
                                                                                                epoch_lwlrap_train))

        predictions = []
        truth = []

        for batch_valid in tqdm.tqdm(eval_batches, desc='Batch'):
            X_test, y_test = dataloader.load_features(batch_valid, features='mel', num_classes=cfg['num_classes'])
            X_test = X_test[:, :, :, np.newaxis]

            metrics = network.test_on_batch(x=X_test, y=y_test)

            batch_val_loss.append(metrics[0])
            batch_val_acc.append(metrics[1])
            preds = network.predict(X_test, batch_size=cfg['batchsize'], verbose=0)
            preds = dataloader.one_hot_encode(np.nonzero(preds > clf_threshold)[0])
            predictions.extend(preds)
            truth.extend(y_test)

        epoch_lwlrap_eval = calculate_overall_lwlrap_sklearn(np.asarray(predictions), np.asarray(truth))
        lwlraps_eval.append(epoch_lwlrap_eval)

        print('Loss on validation set after epoch {}: {}'.format(epoch, np.mean(batch_val_loss)))
        print('Accuracy on validation set after epoch {}: {}'.format(epoch, np.mean(batch_val_acc)))
        print('Label weighted label ranking average precision on validation set after epoch {}: {}'.format(epoch,
                                                                                               epoch_lwlrap_eval))

        current_loss = np.mean(batch_val_loss)
        current_acc = np.mean(batch_val_acc)

        if epoch > 0:
            if epoch_lwlrap_eval > np.amax(lwlraps_eval):
                epochs_without_decrase = 0
                print("Average lwlrap increased - Saving weights...\n")
                network.save_weights("models/baseline_fold{}.hd5".format(fold))
            elif not cfg['linear_decay']:
                epochs_without_decrase += 1
                if epochs_without_decrase == cfg['epochs_without_decrease']:
                    lr = keras.backend.get_value(network.optimizer.lr)
                    lr = lr * cfg['lr_decrease']
                    keras.backend.set_value(network.optimizer.lr, lr)
                    print("lwlrap did not increase for the last {} epochs - halfing learning rate...".format(
                        cfg['epochs_without_decrease']))
                    epochs_without_decrase = 0

            if cfg['linear_decay']:
                if epoch >= cfg['start_linear_decay']:
                    lr = keras.backend.get_value(network.optimizer.lr)
                    lr = lr - cfg['lr_decrease']
                    keras.backend.set_value(network.optimizer.lr, lr)
                    print("Decreasing learning rate by {}...".format(cfg['lr_decrease']))
        else:
            print("Average lwlrap increased - Saving weights...\n")
            network.save_weights("models/baseline_fold{}.hd5".format(fold))

        # Save loss and learning curve of trained model
        save_learning_curve(train_acc, val_acc, "baseline_accuracy_learning_curve.pdf", 'Accuracy', 'Accuracy')
        save_learning_curve(train_loss, val_loss, "baseline_loss_curve.pdf", 'Loss Curve', 'Loss')
        save_learning_curve(lwlraps_train, lwlraps_eval, 'baseline_lwlrap_curve.pdf',
                            "Label Weighted Label Ranking Average Precision", 'lwlrap')


if __name__ == '__main__':
    main()
