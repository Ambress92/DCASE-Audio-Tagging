import numpy as np
from dataloader import get_label_mapping
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import itertools
from collections import Counter
from argparse import ArgumentParser
import config
import os

TOP_N = 3

def opts_parser():
    descr = "Evaluates predictions against ground truth."
    parser = ArgumentParser(description=descr)
    parser.add_argument('infile', nargs='+', metavar='INFILE',
            type=str,
            help='File to load the predictions from (.npz/.pkl format). '
                 'If given multiple times, predictions will be averaged. '
                 'If ending in ":VALUE", will weight by VALUE.')
    parser.add_argument('--filelist',
            type=str, default='valid',
            help='Name of the file list to use (default: %(default)s)')
    parser.add_argument('--year', default='2019', type=str,
                        help='year to evaluate on')
    config.prepare_argument_parser(parser)
    return parser

def save_confusion_matrix(predictions, true_labels, model, normalize=False):
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
    plt.gcf().savefig('plots/confusion_matrix_{}.png'.format(model))

def print_precision_recall_fscore(predictions, true_labels, label_mapping):
    p,r,f,s = precision_recall_fscore_support(true_labels, predictions)
    counts = Counter(true_labels)
    num_classes = len(np.unique(true_labels))

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

    plt.gcf().savefig('plots/{}_table.pdf'.format(clf))

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

def print_maps(ap_sums, ap_counts, class_map=None):
    """
    Prints per-class and overall MAP.
    """

    map_count = 0
    map_sum = 0.
    print('\n')
    for class_index in sorted(ap_counts.keys()):
        m_ap = ap_sums[class_index] / ap_counts[class_index]
        print('MAP for %s: %.4f' % (class_map[class_index], m_ap))
        map_count += ap_counts[class_index]
        map_sum += ap_sums[class_index]
    m_ap = map_sum / map_count
    print('Overall MAP: %.4f\n' % m_ap)

def load_true_labels(year, filelist):
    with open('../datasets/{}/{}'.format(year, filelist), 'r') as in_file:
        true_labels = in_file.readlines()
        true_labels = true_labels[1:]

    truth = {line.split(',')[0]: int(line.split(',')[1].rstrip()) for line in true_labels}
    files = [line.split(',')[0] for line in true_labels]
    return truth, files


def load_predictions(infile):
    with open('predictions/{}'.format(infile), 'r') as in_file:
        predicted_labels = in_file.readlines()
        predicted_labels = predicted_labels[1:]

    preds = {line.split(',')[0]: line.split(',')[1].rstrip().split(' ') for line in predicted_labels}
    return preds


def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()
    infiles = options.infile
    if os.path.exists(os.path.splitext(infiles[0])[0] + '.vars'):
        options.vars.insert(1, os.path.splitext(infiles[0])[0] + '.vars')
    cfg = config.from_parsed_arguments(options)

    # get a list of infiles - later on we can perform different fusion methods on different prediction files
    infiles = [infile.rsplit(':', 1)[0] for infile in infiles]

    # load and prep test data
    label_mapping, inv_label_mapping = get_label_mapping(options.year)

    if len(infiles) > 1:
        pass
    else:
        preds = load_predictions(infiles[0])
        truth, files = load_true_labels(options.year, options.filelist)
        counts = Counter(truth.values())

        all_predictions = [label_mapping[p] for file in files for p in preds[file]]
        all_predictions = np.asarray(all_predictions).reshape(-1, 3)
        best_predictions = [pred[0] for pred in all_predictions]
        true_labels = [truth[file] for file in files]

        p, r, f, s = precision_recall_fscore_support(true_labels, best_predictions)
        print_precision_recall_fscore(best_predictions, true_labels, inv_label_mapping)
        plot_results_table(p, r, f, counts, inv_label_mapping, cfg['num_classes'], 'baseline')
        save_confusion_matrix(best_predictions, true_labels, 'baseline')
        avg_precisions = np.mean([avg_precision(a, p) for a, p in zip(true_labels, all_predictions)])
        print('Model {} achieved an average label precision of {}.'.format('baseline', avg_precisions))

if __name__ == '__main__':
    main()
