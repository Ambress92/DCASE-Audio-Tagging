import os
from argparse import ArgumentParser

import config
import dataloader
import keras
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import utils
from dataloader import generate_in_background
from sklearn.metrics import label_ranking_average_precision_score
np.random.seed(101)


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
        eval_files = []
        with open('../datasets/cv/fold{}_curated_train'.format(fold), 'r') as in_file:
            train_files.extend(in_file.readlines())
        train_files_noisy = os.listdir('../datasets/train_noisy')
        train_files_noisy = [d.rstrip() for d in train_files_noisy]

        with open('../datasets/cv/fold{}_curated_eval'.format(fold), 'r') as in_file:
            eval_files.extend(in_file.readlines())

        print('Loading model')
        # import model from file
        selected_model = utils.import_model(modelfile)

        # instantiate neural network
        print("Preparing training function...")

        train_formats = (cfg['feature_height'], cfg['feature_width'], cfg['channels'])
        if not 'densenet' in modelfile:
            network = selected_model.architecture(train_formats, cfg['num_classes'])
        else:
            network = selected_model.architecture(train_formats, cfg['num_classes'], 128)

        # Add optimizer and compile model
        print("Compiling model ...")
        optimizer = keras.optimizers.Adam(lr=cfg['lr'])
        if 'IIC' in modelfile:
            network.compile(optimizer=optimizer, loss={'clf_output':cfg['loss'],
                                                       'clf_output_augmented':cfg['loss']})
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
        mutual_inf = []
        val_mutual_inf = []
        epochs_without_decrase = 0
        lwlraps_eval = []
        lwlraps_train = []
        lr = cfg['lr']
        noisy_lr = lr / 10
        min_curated_lr = 1e-5
        lr_decay = (lr-min_curated_lr)/(cfg['epochs']-cfg['start_linear_decay']+1)
        switch_train_set = cfg['switch_train_set']
        optimizer_changed=False


        if not 'IIC' in modelfile:
            train_batches = generate_in_background(
                dataloader.load_batches(train_files, cfg['batchsize'], shuffle=True, infinite=True,
                                        features=cfg['features'], feature_width=cfg['feature_width'],
                                        fixed_length=cfg['fixed_size'], jump=cfg['jump'], mixup=True, augment=False))
            train_noisy_batches = dataloader.load_batches(train_files_noisy, cfg['batchsize'], shuffle=True,
                                                          infinite=True, feature_width=cfg['feature_width'],
                                                          features=cfg['features'], mixup=True, augment=False,
                                                          fixed_length=cfg['fixed_size'], jump=cfg['jump'])
        else:
            train_batches = generate_in_background(
                dataloader.load_batches(train_files, cfg['batchsize'], shuffle=True, infinite=True,
                                        features=cfg['features'], feature_width=cfg['feature_width'],
                                        fixed_length=cfg['fixed_size'], jump=cfg['jump'], mixup=False, augment=True))
            train_noisy_batches = dataloader.load_batches(train_files_noisy, cfg['batchsize'], shuffle=True,
                                                          infinite=True, feature_width=cfg['feature_width'],
                                                          features=cfg['features'], mixup=False, augment=True,
                                                          fixed_length=cfg['fixed_size'], jump=cfg['jump'])

        swa_weights = []

        # run training loop
        print("Training:")
        for epoch in range(1, cfg['epochs']+1):

            epoch_train_loss = []
            epoch_train_acc = []
            batch_val_loss = []
            batch_val_acc = []
            epoch_lwlrap_train = []
            epoch_lwlrap_eval = []
            epoch_mutual_inf = []
            epoch_val_mutual_inf = []
            steps_per_epoch = len(train_files) // cfg['batchsize']

            if optimizer_changed:
                K.set_value(network.optimizer.lr, lr)
                optimizer_changed=False
            elif (epoch % switch_train_set) == 0:
                noisy_lr = lr / 10
                K.set_value(network.optimizer.lr, noisy_lr)


            if not 'IIC' in modelfile:
                eval_batches = dataloader.load_batches(eval_files, cfg['batchsize'], infinite=False, features=cfg['features'],
                                                       feature_width=cfg['feature_width'], fixed_length=cfg['fixed_size'],
                                                                          mixup=False, augment=False, jump=cfg['jump'])
            else:
                eval_batches = dataloader.load_batches(eval_files, cfg['batchsize'], infinite=False,
                                                       features=cfg['features'],
                                                       feature_width=cfg['feature_width'],
                                                       fixed_length=cfg['fixed_size'],
                                                       mixup=False, augment=True, jump=cfg['jump'])

            for _ in tqdm.trange(
                    steps_per_epoch,
                    desc='Epoch %d/%d:' % (epoch, cfg['epochs'])):

                if (epoch % switch_train_set) == 0:
                    X_train, y_train = next(train_noisy_batches)

                else:
                    X_train, y_train = next(train_batches)

                metrics = network.train_on_batch(x=X_train, y=y_train)
                if not 'IIC' in modelfile:
                    epoch_train_acc.append(metrics[1])
                    epoch_train_loss.append(metrics[0])
                else:
                    epoch_mutual_inf.append(metrics[0])
                    epoch_train_loss.append(metrics[1])

                preds = network.predict(x=X_train, batch_size=cfg['batchsize'], verbose=0)
                if not 'IIC' in modelfile:
                    epoch_lwlrap_train.append(lwlrap_metric(np.asarray(y_train), np.asarray(preds)))
                else:
                    epoch_lwlrap_train.append(lwlrap_metric(np.asarray(y_train[0]), np.asarray(preds[0])))

            if not 'IIC' in modelfile:
                print('Accuracy on training set after epoch {}: {}\n'.format(epoch, np.mean(epoch_train_acc)))
                train_acc.append(np.mean(epoch_train_acc))

            else:
                print('Mutual Information on training set after epoch {}: {}\n'.format(epoch, np.mean(epoch_mutual_inf)))
                mutual_inf.append(np.mean(epoch_mutual_inf))

            print('Loss on training set after epoch {}: {}'.format(epoch, np.mean(epoch_train_loss)))
            print('Label weighted label ranking average precision on training set after epoch {}: {}'.format(epoch,
                                                                                                             np.mean(
                                                                                                                 epoch_lwlrap_train)))

            train_loss.append(np.mean(epoch_train_loss))
            lwlraps_train.append(np.mean(epoch_lwlrap_train))

            for X_test, y_test in tqdm.tqdm(eval_batches, desc='Batch'):

                metrics = network.test_on_batch(x=X_test, y=y_test)

                if not 'IIC' in modelfile:
                    batch_val_loss.append(metrics[0])
                    batch_val_acc.append(metrics[1])
                else:
                    batch_val_loss.append(metrics[1])
                    epoch_val_mutual_inf.append(metrics[0])


                predictions = network.predict(x=X_test, batch_size=cfg['batchsize'], verbose=0)

                if not 'IIC' in modelfile:
                    epoch_lwlrap_eval.append(lwlrap_metric(np.asarray(y_test), np.asarray(predictions)))
                else:
                    epoch_lwlrap_eval.append(lwlrap_metric(np.asarray(y_test[0]), np.asarray(predictions[0])))


            if not 'IIC' in modelfile:
                val_acc.append(np.mean(batch_val_acc))
                print('Accuracy on validation set after epoch {}: {}'.format(epoch, np.mean(batch_val_acc)))
            else:
                val_mutual_inf.append(np.mean(epoch_val_mutual_inf))
                print('Mutual Information on validation set after epoch {}: {}'.format(epoch, np.mean(epoch_val_mutual_inf)))

            val_loss.append(np.mean(batch_val_loss))
            print('Loss on validation set after epoch {}: {}'.format(epoch, np.mean(batch_val_loss)))
            print('Label weighted label ranking average precision on validation set after epoch {}: {}'.format(epoch,
                                                                                                               np.mean(
                                                                                                                   epoch_lwlrap_eval)))

            current_lwlrap = np.mean(epoch_lwlrap_eval)

            if epoch > 1:
                if current_lwlrap > np.max(lwlraps_eval):
                    epochs_without_decrase = 0
                    print("Average lwlrap increased - Saving weights...\n")
                    network.save("models/{}/{}_fold{}.hd5".format(cfg['features'], modelfile.replace('.py', ''), fold))
                elif not cfg['linear_decay'] and not cfg['sharp_drop']:
                    epochs_without_decrase += 1
                    if epochs_without_decrase == cfg['epochs_without_decrease']:
                        lr = K.get_value(network.optimizer.lr)
                        lr = lr * cfg['lr_decrease']
                        K.set_value(network.optimizer.lr, lr)
                        print("lwlrap did not increase for the last {} epochs - halfing learning rate...".format(
                            cfg['epochs_without_decrease']))
                        epochs_without_decrase = 0

                if cfg['linear_decay']:
                    if epoch >= cfg['start_linear_decay'] and lr >= min_curated_lr:
                        lr = lr - lr_decay
                        keras.backend.set_value(network.optimizer.lr, lr)
                        print("Decreasing learning rate by {}...".format(lr_decay))

                if cfg['sharp_drop']:
                    if epoch == cfg['sharp_drop_epoch']:
                        lr = lr * cfg['drop_rate']
                        K.set_value(network.optimizer.lr, lr)

            else:
                print("Average lwlrap increased - Saving weights...\n")
                network.save("models/{}/{}_fold{}.hd5".format(cfg['features'], modelfile.replace('.py', ''), fold))

            lwlraps_eval.append(np.mean(epoch_lwlrap_eval))

            if (epoch % switch_train_set) == 0:
                optimizer_changed = True

            if not os.path.exists('plots/{}'.format(cfg['features'])):
                os.makedirs('plots/{}'.format(cfg['features']))

            # Save loss and learning curve of trained model
            if not 'IIC' in modelfile:
                save_learning_curve(train_acc, val_acc, "{}/{}_fold{}_accuracy_learning_curve.pdf".format(cfg['features'], modelfile.replace('.py', ''), fold), 'Accuracy', 'Accuracy')
            else:
                save_learning_curve(mutual_inf, val_mutual_inf,
                                    "{}/{}_fold{}_mutual_inf_learning_curve.pdf".format(cfg['features'],
                                                                                      modelfile.replace('.py', ''),
                                                                                      fold), 'Mutual Information', 'Mutual Information')
            save_learning_curve(train_loss, val_loss, "{}/{}_fold{}_loss_curve.pdf".format(cfg['features'], modelfile.replace('.py', ''), fold), 'Loss Curve', 'Loss')
            save_learning_curve(lwlraps_train, lwlraps_eval, '{}/{}_fold{}_lwlrap_curve.pdf'.format(cfg['features'], modelfile.replace('.py', ''), fold),
                                "Label Weighted Label Ranking Average Precision", 'lwlrap')

            swa_epoch = cfg['epochs'] + 1 - cfg['swa_epochs']
            if epoch == swa_epoch:
                # first swa epoch
                swa_weights = network.get_weights()
            elif epoch > swa_epoch:
                # beginning averaging
                for i in range(len(swa_weights)):
                    swa_weights[i] = (swa_weights[i] * (epoch - swa_epoch) + network.get_weights()[i]) \
                                                    / ((epoch - swa_epoch) + 1)
        # after end of training, store averaged (swa) model
        network.set_weights(swa_weights)
        network.save('models/{}/{}_fold{}.hd5'.format(cfg['features'], modelfile.replace('.py', '_swa'), fold))


if __name__ == '__main__':
    main()
