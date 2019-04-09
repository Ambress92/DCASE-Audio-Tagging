import numpy as np
from dataloader import get_label_mapping
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import itertools
from collections import Counter
from argparse import ArgumentParser
import config
import os
import sklearn.metrics

TOP_N = 3

def opts_parser():
    descr = "Evaluates predictions against ground truth."
    parser = ArgumentParser(description=descr)
    parser.add_argument('infile', nargs='+', metavar='INFILE',
            type=str,
            help='File to load the predictions from (.npz/.pkl format). '
                 'If given multiple times, predictions will be averaged. '
                 'If ending in ":VALUE", will weight by VALUE.')
    parser.add_argument('-truth', required=True,
                        type=str,
                        help='File to load the true labels from (.npz/.pkl format).')

    config.prepare_argument_parser(parser)
    return parser


def _one_sample_positive_class_precisions(scores, truth):
    """Calculate precisions for each true class for a single sample.

    Args:
      scores: np.array of (num_classes,) giving the individual classifier scores.
      truth: np.array of (num_classes,) bools indicating which classes are true.

    Returns:
      pos_class_indices: np.array of indices of the true classes for this sample.
      pos_class_precisions: np.array of precisions corresponding to each of those
        classes.
    """
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)
    # Only calculate precisions if there are some true classes.
    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)
    # Retrieval list of classes for this sample.
    retrieved_classes = np.argsort(scores)[::-1]
    # class_rankings[top_scoring_class_index] == 0 etc.
    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)
    # Which of these is a true label?
    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
    # Num hits for every truncated retrieval list.
    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
    # Precision of retrieval list truncated at each hit, in order of pos_labels.
    precision_at_hits = (
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(np.float)))
    return pos_class_indices, precision_at_hits

# All-in-one calculation of per-class lwlrap.
def calculate_per_class_lwlrap(truth, scores):
    """Calculate label-weighted label-ranking average precision.

    Arguments:
      truth: np.array of (num_samples, num_classes) giving boolean ground-truth
        of presence of that class in that sample.
      scores: np.array of (num_samples, num_classes) giving the classifier-under-
        test's real-valued score for each class for each sample.

    Returns:
      per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each
        class.
      weight_per_class: np.array of (num_classes,) giving the prior of each
        class within the truth labels.  Then the overall unbalanced lwlrap is
        simply np.sum(per_class_lwlrap * weight_per_class)
    """
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    # Space to store a distinct precision value for each class on each sample.
    # Only the classes that are true for each sample will be filled in.
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = (
            _one_sample_positive_class_precisions(scores[sample_num, :],
                                                  truth[sample_num, :]))
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = (
            precision_at_hits)
    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))
    # Form average of each column, i.e. all the precisions assigned to labels in
    # a particular class.
    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /
                        np.maximum(1, labels_per_class))
    # overall_lwlrap = simple average of all the actual per-class, per-sample precisions
    #                = np.sum(precisions_for_samples_by_classes) / np.sum(precisions_for_samples_by_classes > 0)
    #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples
    #                = np.sum(per_class_lwlrap * weight_per_class)
    return per_class_lwlrap, weight_per_class

# Calculate the overall lwlrap using sklearn.metrics function.

def calculate_overall_lwlrap_sklearn(truth, scores):
  """Calculate the overall lwlrap using sklearn.metrics.lrap."""
  # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.
  sample_weight = np.sum(truth > 0, axis=1)
  nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
  overall_lwlrap = sklearn.metrics.label_ranking_average_precision_score(
      truth[nonzero_weight_sample_indices, :] > 0,
      scores[nonzero_weight_sample_indices, :],
      sample_weight=sample_weight[nonzero_weight_sample_indices])
  return overall_lwlrap

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

def save_lwlrap_results_table(lwlrap, labels, weights, clf):

    latex_table = '% !TeX root = initial_structure.tex\n \
                  \\begin{tabular}{rrr}\n\
    \\toprule\n\
         \\multicolumn{1}{c}{CLASS}\n\
         & \\multicolumn{1}{c}{LWLRAP}\n \
        & \\multicolumn{l}{c}{WEIGHT}\n \\\\  \
    \\midrule\n'
    for c in range(len(labels)):
        latex_table += "%9s & %.4f & %.4f\\\\\n" % (labels[c], lwlrap[c], weights[c])

    latex_table += "\\\\bottomrule\n\end{tabular}"

    with open('plots/{}_lwlrap_latex_table.tex'.format(clf), "w") as latex_out:
        latex_out.write(latex_table)


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

def load_true_labels(file):
    return np.load('predictions/{}'.format(file))


def load_predictions(infile):
    return np.load('predictions/{}'.format(infile))


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
    label_mapping, inv_label_mapping = get_label_mapping()
    labels = list(label_mapping.keys())

    if len(infiles) > 1:
        # perform late fusion (average, min, max)
        pass
    else:
        preds = load_predictions(infiles[0])
        truth = load_true_labels(options.truth)
        # counts = Counter(truth.values())

        # all_predictions = [label_mapping[p] for file in files for p in preds[file]]
        # all_predictions = np.asarray(all_predictions).reshape(-1, 3)
        # best_predictions = [pred[0] for pred in all_predictions]
        # true_labels = [truth[file] for file in files]

        # p, r, f, s = precision_recall_fscore_support(true_labels, best_predictions)
        # print_precision_recall_fscore(best_predictions, true_labels, inv_label_mapping)
        # plot_results_table(p, r, f, counts, inv_label_mapping, cfg['num_classes'], 'baseline')
        # save_confusion_matrix(best_predictions, true_labels, 'baseline')
        # avg_precisions = np.mean([avg_precision(a, p) for a, p in zip(true_labels, all_predictions)])
        per_class_lwlrap, weights = calculate_per_class_lwlrap(truth, preds)
        lwlrap = calculate_overall_lwlrap_sklearn(truth, preds)
        save_lwlrap_results_table(per_class_lwlrap, labels, weights, 'RF')
        print('Model {} achieved an lwlrap of {}.'.format('baseline', lwlrap))

if __name__ == '__main__':
    main()
