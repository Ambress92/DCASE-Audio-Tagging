# https://github.com/DCASE-REPO/dcase2018_baseline
import argparse
import numpy as np
from dataloader import load_test_files, get_label_mapping
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import itertools
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('-year', required=True)
parser.add_argument('-features', required=True)
parser.add_argument('-clf', help='Classifier to use, by default RF is used', default='RF')
args = parser.parse_args()

TOP_N = 3

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
    plt.gcf().savefig('plots/confusion_matrix_{}.png'.format(args.clf))

def print_precision_recall_fscore(predictions, true_labels):
    p,r,f,s = precision_recall_fscore_support(true_labels, predictions)
    counts = Counter(true_labels)
    num_classes = len(np.unique(true_labels))
    label_mapping = get_label_mapping(args.year)

    print("\n")
    print("%9s  |   %s  |  %4s  |  %4s  |   %4s   |" % ("CLASS", "CNT", "PR ", "RE ", "F1 "))
    print('-' * 50)
    for c in range(num_classes):
        print("%9s  |  % 4d  |  %.2f  |  %.2f  |  %.3f   |" % (label_mapping[c], counts[c], p[c], r[c], f[c]))
    print('-' * 50)
    print("%9s  |  % 4d  |  %.2f  |  %.2f  |  %.3f   |" % ('average', np.sum(list(counts.values())), np.mean(p), np.mean(r), np.mean(f)))
    print('=' * 50)

def plot_results_table(p, r, f, count, id_class_mapping, num_classes, clf):
    columns = ['CLASS', 'COUNT', 'PR', 'RE', 'F1']
    row_text = []
    classes = [id_class_mapping[c] for c in range(num_classes)]

    row_text = []
    latex_table = '\\begin{tabular}{rrrrr}\n\
    \\toprule\n\
         \\multicolumn{1}{c}{CLASS}\n\
         & \\multicolumn{1}{c}{COUNT}\n\
         & \\multicolumn{1}{c}{PR}\n\
         & \\multicolumn{1}{c}{RE}\n\
         & \\multicolumn{1}{c}{F1} \\\\\n\
    \\midrule\n'
    for c in range(len(classes)):
        # row = []
        # row.append("%9s" % classes[c])
        # row.append("% 4d" % count[c])
        # row.append("%.2f" % p[c])
        # row.append("%.2f" % r[c])
        # row.append("%.3f" % f[c])
        # row_text.append(row)
        latex_table += "%9s & %4d & %.2f & %.2f & %.3f\\\\\n" % (classes[c], count[c], p[c], r[c], f[c])

    row_text.append(["average", "% 4d " % np.sum(list(count.values())), "%.2f" % np.mean(p), "%.2f" % np.mean(r), "%.3f" % np.mean(f)])
    latex_table += "\midrule\naverage & %4d & %.2f & %.2f & %.3f\\\\\n\\bottomrule\n\end{tabular}" % (np.sum(list(count.values())), np.mean(p), np.mean(r), np.mean(f))
    plt.figure(figsize=(15,15))
    fig, ax = plt.subplots()

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    with open('plots/{}_latex_table.tex'.format(clf), "w") as latex_out:
        latex_out.write(latex_table)

    table = ax.table(cellText=row_text,
                     colLabels=columns,
                     loc='center')

    fig.tight_layout()

    plt.gcf().savefig('plots/RF_table.pdf')

def main():
    # load and prep test data
    print('Loading test clips...')
    test_data = load_test_files(args.year, args.features)
    label_mapping, _ = get_label_mapping(args.year)

    X = []
    y = []
    audio_splits = []
    start = 0
    for x in test_data:
        y.append(label_mapping[x[1]])
        x = x[0]
        audio_splits.append(slice(start, start+len(x)))
        start = start + len(x)
        for datapoint in x:
            X.append(datapoint)

    # reconstruct models and classify test clips
    if args.clf == 'RF':
        clf = np.load('models/RF_verified.npy')
        predictions = clf.predict_proba(X)
    elif args.clf == 'SVM':
        clf = np.load('models/RF_verified.npy')
        predictions = clf.predict_proba(X)
    else:
        pass

    # compute mean average label precision
    predictions = [get_top_predicted_classes(predictions[slc]) for slc in audio_splits]
    avg_precisions = np.mean([avg_precision(a, p) for a, p in zip(y, predictions)])
    print('Model {} achieved an average label precision of {}.'.format(args.clf, avg_precisions))
    # show additional metrics
    print_precision_recall_fscore(predictions, y)

def get_top_predicted_classes(predicted):
    """
    Computes the top N predicted classes given the prediction scores for all examples in a clip.
    """
    # see https://github.com/DCASE-REPO/dcase2018_baseline
    predicted = np.average(predicted, axis=0)
    predicted_classes = np.argsort(predicted)[::-1][:TOP_N]
    return predicted_classes

def avg_precision(actual=None, predicted=None):
    """
    Computes average label precision.
    """
    # see https://github.com/DCASE-REPO/dcase2018_baseline
    for (i, p) in enumerate(predicted):
        if actual == p:
            return 1.0 / (i + 1.0)
    return 0.0

if __name__ == '__main__':
    main()
