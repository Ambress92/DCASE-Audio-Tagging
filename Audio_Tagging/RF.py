from sklearn.ensemble import RandomForestClassifier
import argparse
from dataloader import get_label_mapping, load_features, get_total_file_dict
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import os

parser = argparse.ArgumentParser()
parser.add_argument('-features', required=True)
args = parser.parse_args()

def save_confusion_matrix(predictions, true_labels, normalize=False):
    cnf_matrix = confusion_matrix(true_labels, predictions)
    labels = np.unique(true_labels)

    if normalize:
        cnf_matrix = cnf_matrix.astype(np.float) / cnf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(15,15))
    plt.imshow(np.log(cnf_matrix), interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title('Confusion Matrix')
    plt.xticks(np.arange(len(labels)), labels, rotation=45)
    plt.yticks(np.arange(len(labels)), labels)

    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.gcf().savefig('plots/confusion_matrix_RF.png')

def get_top_predicted_classes(predicted):
    """
    Computes the top N predicted classes given the prediction scores for all examples in a clip.
    """
    # see https://github.com/DCASE-REPO/dcase2018_baseline
    predicted = np.average(predicted, axis=0)
    predicted_classes = np.argsort(predicted)[::-1][:3]
    return predicted_classes

def main():
    label_mapping, inv_label_mapping = get_label_mapping()
    num_classes = len(label_mapping)

    for fold in range(1,5):
        print('Loading data...')
        with open('../datasets/cv/fold{}_train'.format(fold)) as in_file:
            filelist_train = in_file.readlines()

        with open('../datasets/cv/fold{}_eval'.format(fold)) as in_file:
            filelist_eval = in_file.readlines()

        X_train, y_train = load_features(filelist_train, args.features, num_classes)
        X_eval, y_eval = load_features(filelist_eval, args.features, num_classes)

        print('Load complete')
        clf = RandomForestClassifier(n_estimators=20, verbose=2, max_depth=None, n_jobs=-1)
        clf.fit(X_train, y_train)

        # load and prep test data
        print('Predicting')
        predictions = clf.predict(X_eval)

        y_eval_labels = [np.nonzero(t > 0)[0] for t in y_eval]
        predictions_labels = [np.nonzero(p > 0)[0] for p in predictions]

        print('Dumping Predictions and true labels')
        np.save('predictions/RF_predictions_fold{}'.format(fold), predictions_labels)
        np.save('predictions/RF_true_labels_fold{}'.format(fold), y_eval_labels)

        # predict on test set
        test_files = os.listdir('../datasets/test')
        filelist_test = [file.replace('.wav', '') for file in test_files]
        X_test, y_test = load_features(filelist_test, args.features, num_classes)

        print('Predicting on test set')
        predictions = clf.predict(X_test)

        y_test = [np.nonzero(t > 0)[0] for t in predictions]
        predictions_labels = [np.nonzero(p > 0)[0] for p in y_test]
        print('Dumping predictions on the test set')
        np.save('predictions/RF_test_set_fold{}'.format(fold), predictions_labels)


if __name__ == '__main__':
    main()
