from sklearn.svm import SVC
import argparse
from dataloader import load_verified_files, get_label_mapping, load_test_files
import numpy as np
from evaluate import avg_precision, plot_results_table
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import itertools
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('-year', required=True)
parser.add_argument('-features', required=True)
args = parser.parse_args()

def save_confusion_matrix(predictions, true_labels, normalize=False):
    cnf_matrix = confusion_matrix(true_labels, predictions)
    labels = np.unique(true_labels)

    if normalize:
        cnf_matrix = cnf_matrix.astype(np.float) / cnf_matrix.sum(axis=1)[:, np.newaxis]

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
    label_mapping, inv_label_mapping = get_label_mapping(args.year)

    print('Loading data...')
    data = load_verified_files(args.year, args.features)

    X = []
    y = []
    for x in data:
        label = x[1]
        x = x[0]
        for datapoint in x:
            X.append(datapoint)
            y.append(label_mapping[label])

    print('Load complete')
    X = np.asarray(X)
    y = np.asarray(y)

    clf = SVC(random_state=101, probability=True, verbose=True)
    clf.fit(X, y)

    # load and prep test data
    print('Loading test clips...')
    test_data = load_test_files(args.year, args.features)
    label_mapping, _ = get_label_mapping(args.year)

    X = []
    y = []
    y_true = []
    audio_splits = []
    start = 0
    for x in test_data:
        label = label_mapping[[x[1]]]
        y.append(label)
        x = x[0]
        audio_splits.append(slice(start, start + len(x)))
        start = start + len(x)
        for datapoint in x:
            X.append(datapoint)
            y_true.append(label)

    print('Predicting')
    predictions = clf.predict_proba(X)

    # compute mean average label precision
    preds = [get_top_predicted_classes(predictions[slc]) for slc in audio_splits]
    avg_precisions = np.mean([avg_precision(a, p) for a, p in zip(y, preds)])
    print('Random Forest achieved an average label precision of {}.'.format(avg_precisions))
    # show additional metrics
    predictions = np.argmax(predictions, axis=1)
    p, r, f, s = precision_recall_fscore_support(y_true, predictions)
    counts = Counter(y_true)
    num_classes = len(np.unique(y))
    label_mapping = get_label_mapping(args.year)

    print("\n")
    print("%9s  |   %s  |  %4s  |  %4s  |   %4s   |" % ("CLASS", "CNT", "PR ", "RE ", "F1 "))
    print('-' * 50)
    for c in range(num_classes):
        print("%9s  |  % 4d  |  %.2f  |  %.2f  |  %.3f   |" % (label_mapping[c], counts[c], p[c], r[c], f[c]))
    print('-' * 50)
    print("%9s  |  % 4d  |  %.2f  |  %.2f  |  %.3f   |" % (
    'average', np.sum(list(counts.values())), np.mean(p), np.mean(r), np.mean(f)))
    print('=' * 50)
    save_confusion_matrix(predictions, y_true)
    plot_results_table(p, r, f, counts, inv_label_mapping, num_classes, 'SVM')


if __name__ == '__main__':
    main()
