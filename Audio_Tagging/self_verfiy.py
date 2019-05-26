import dataloader
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import config
from sklearn.metrics import label_ranking_average_precision_score
from keras.models import load_model
from keras.models import model_from_yaml
import tqdm
import keras.backend as K
from keras.optimizers import Adam
from sklearn.utils import shuffle

def opts_parser():
    descr = "Trains a neural network."
    parser = ArgumentParser(description=descr)
    parser.add_argument('modelfile', metavar='MODELFILE',
            type=str,
            help='File to save the learned weights to')
    config.prepare_argument_parser(parser)
    return parser

def save_learning_curve(metric, filename, title, ylabel):
    plt.plot(metric)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Epoch')
    plt.legend(['validation'], loc='upper left')
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

def main():
    parser = opts_parser()
    options = parser.parse_args()
    modelfile = options.modelfile
    cfg = config.from_parsed_arguments(options)
    label_mapping, inv_label_mapping = dataloader.get_label_mapping()

    for fold in range(1,5):
        # load pretrained model
        if '.yaml' in modelfile:
            with open(modelfile, 'r') as yaml_file:
               yaml_model = yaml_file.read()
            network = model_from_yaml(yaml_model)
            network.load_weights(modelfile.replace('.yaml', '.hd5'))
            optimizer = Adam(lr=cfg['finetune_lr'])
            network.compile(optimizer=optimizer, loss=cfg["loss"], metrics=['acc'])
            modelfile = modelfile.replace('.yaml', '')
        else:
            network = load_model("models/{}/{}_fold{}.hd5".format(cfg['features'], modelfile.replace('.py', ''), fold))
            modelfile = modelfile.replace('.py', '')

        verify_files_noisy = []
        validation_files = []
        with open('../datasets/cv/fold{}_curated_eval'.format(fold), 'r') as in_file:
            validation_files.extend(in_file.readlines())
        for f in range(1,5):
            with open('../datasets/cv/fold{}_noisy_eval'.format(fold), 'r') as in_file:
                verify_files_noisy.extend(in_file.readlines())

        print('Start self verification loop...')
        verified_frames = []
        verified_frame_labels = []
        labels_per_epoch = cfg['labels_per_epoch']
        min_lr = 1e-5
        lr_decay = (cfg['finetune_lr'] - min_lr)/(cfg['self_verify_epochs']-cfg['start_linear_decay_finetune']+1)

        accs = []
        losses = []
        lwlraps = []
        threshold = 0.05

        for epoch in range(1,cfg['self_verify_epochs']+1):
            epoch_train_acc = []
            epoch_train_loss = []
            epoch_lwlrap_train = []
            label_count = {k: 0 for k in label_mapping.keys()}

            verify_noisy_batches = dataloader.generate_in_background(
                dataloader.load_batches_verification(verify_files_noisy, k=cfg['k'],
                                                     shuffle=True, infinite=False,
                                                     features=cfg['features'], feature_width=cfg['feature_width'],
                                                     fixed_length=cfg['fixed_size'], jump=cfg['jump']))

            print('Predict on noisy data...')
            for X, y in tqdm.tqdm(verify_noisy_batches, desc='Batch'):
                predictions = network.predict(x=X, batch_size=cfg['verify_batchsize'], verbose=0)
                current_lwlrap = lwlrap_metric(y, predictions)
                for i, preds in enumerate(zip(predictions, y)):
                    p = preds[0]
                    t = preds[1]
                    true_labels = np.nonzero(t > 0)[0]
                    predicted_labels = np.nonzero(p > np.max(p)-threshold)[0]

                    for ind_t, ind_p in zip(true_labels, predicted_labels):
                        if ind_t == ind_p:
                            l = inv_label_mapping[ind_t]
                            if label_count[l] < labels_per_epoch:
                                verified_frames.append(X[i])
                                verified_frame_labels.append(y[i])
                                label_count[l] += 1

                if current_lwlrap > 0.9:
                    print('lwlrap is greater than 90%')
                    for data, label in zip(X, y):
                        if len(np.nonzero(label > 0)[0]) > 1:
                            labels = inv_label_mapping[np.nonzero(label > 0)[0]]
                            labels = [inv_label_mapping[l] for l in labels]
                            for l in labels:
                                if label_count[l] < labels_per_epoch:
                                    verified_frames.extend(data)
                                    verified_frame_labels.extend(label)
                                    label_count[l] += 1
                        else:
                            label = inv_label_mapping[np.nonzero(label > 0)[0]]
                            if label_count[label] < labels_per_epoch:
                                verified_frames.extend(data)
                                verified_frame_labels.extend(label)
                                label_count[label] += 1

                if (np.asarray(list(label_count.values())) == labels_per_epoch).all():
                    print('{} labels were added, starting fine tuning'.format(np.sum(list(label_count.values()))))
                    break


            train_batches = dataloader.load_batches(validation_files, cfg['verify_batchsize'], shuffle=True,
                                                    infinite=True, features=cfg['features'], jump=cfg['jump'],
                                                    mixup=False, augment=False)
            steps_per_epoch = len(validation_files) // cfg['verify_batchsize']

            print('Finetuning on curated validation set...')
            for _ in tqdm.trange(
                    steps_per_epoch,
                    desc='Epoch %d/%d:' % (epoch, cfg['self_verify_epochs'])):

                X_train, y_train = next(train_batches)

                metrics = network.train_on_batch(x=X_train, y=y_train)
                epoch_train_acc.append(metrics[1])
                epoch_train_loss.append(metrics[0])

                preds = network.predict(x=X_train, batch_size=cfg['verify_batchsize'], verbose=0)

                epoch_lwlrap_train.append(lwlrap_metric(np.asarray(y_train), np.asarray(preds)))

            print('Finetuning on self verified labels...')
            steps_per_epoch = len(verified_frames) // cfg['verify_batchsize']
            verified_frame_labels_training = np.asarray(verified_frame_labels)
            verified_frames_training = np.asarray(verified_frames)
            verified_frames_training, verified_frame_labels_training = shuffle(verified_frames_training, verified_frame_labels_training, random_state=101)
            for i in tqdm.trange(
                    steps_per_epoch,
                    desc='Epoch %d/%d:' % (epoch, cfg['self_verify_epochs'])):

                start = i*cfg['verify_batchsize']
                end = i*cfg['verify_batchsize']+cfg['verify_batchsize']
                X_train = verified_frames_training[start:end, :, :, :]
                y_train = verified_frame_labels_training[start:end, :]

                network.train_on_batch(x=X_train, y=y_train)

            del verified_frame_labels_training
            del verified_frames_training
            print('Loss on training set after epoch {}: {}'.format(epoch, np.mean(epoch_train_loss)))
            print('Accuracy on training set after epoch {}: {}\n'.format(epoch, np.mean(epoch_train_acc)))
            print('Label weighted label ranking average precision on training set after epoch {}: {}'.format(epoch,
                                                                                                             np.mean(epoch_lwlrap_train)))

            accs.append(np.mean(epoch_train_acc))
            losses.append(np.mean(epoch_train_loss))
            current_lwlrap = np.mean(epoch_lwlrap_train)

            if epoch > 1:
                if current_lwlrap > np.max(lwlraps):
                    print("Average lwlrap increased - Saving weights...\n")
                    network.save("models/{}/{}_fold{}_finetuned.hd5".format(cfg['features'], modelfile, fold))
                if epoch >= cfg['start_linear_decay_finetune']:
                    lr = K.get_value(network.optimizer.lr)
                    lr = lr - lr_decay
                    K.set_value(network.optimizer.lr, lr)
                    print("Decreasing learning rate by {}...".format(lr_decay))
            else:
                print("Average lwlrap increased - Saving weights...\n")
                network.save("models/{}/{}_fold{}_finetuned.hd5".format(cfg['features'], modelfile, fold))

            lwlraps.append(current_lwlrap)

            # Save loss and learning curve of trained model
            save_learning_curve(accs,
                                "{}/{}_fold{}_finetuned_accuracy_learning_curve.pdf".format(cfg['features'], modelfile, fold),
                                'Accuracy Curve', 'Accuracy')
            save_learning_curve(losses,
                                "{}/{}_fold{}_finetuned_loss_curve.pdf".format(cfg['features'], modelfile,fold),
                                'Loss Curve', 'Loss')
            save_learning_curve(lwlraps,
                                '{}/{}_fold{}_finetuned_lwlrap_curve.pdf'.format(cfg['features'], modelfile,fold),
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
        network.save('models/{}/{}_fold{}.hd5'.format(cfg['features'], modelfile, fold))


if __name__ == '__main__':
    main()