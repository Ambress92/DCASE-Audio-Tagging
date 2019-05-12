import os
import re
import config
import itertools
import dataloader
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from argparse import ArgumentParser
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

np.random.seed(101)


def opts_parser():
    descr = "Evaluates predictions against ground truth."
    parser = ArgumentParser(description=descr)
    parser.add_argument('infile', nargs='+', metavar='INFILE',
            type=str,
            help='File to load the predictions from (.npz/.pkl format). '
                 'If given multiple times, predictions will be averaged. '
                 'If ending in ":VALUE", will weight by VALUE.')
    parser.add_argument('-truth', required=False,
                        type=str,
                        help='File to load the true labels from (.npz/.pkl format).')
    parser.add_argument('--test', help='Make late fusion on test set', action='store_true')
    parser.add_argument('-features', type=str, help='for which features to use the predictions', required=True)
    parser.add_argument('-level', type=str, help='Whether to make late fusion on file or on frame level', default='frame')
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


def save_confusion_matrix(predictions, true_labels, model, features, inv_label_mapping, normalize=False):
    cnf_matrix = confusion_matrix(true_labels, predictions)
    labels = np.unique(true_labels)
    labels = [inv_label_mapping[l] for l in labels]

    if normalize:
        cnf_matrix = cnf_matrix.astype(np.float) / cnf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(20, 20))
    plt.imshow(np.log(cnf_matrix), interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title('Confusion Matrix')
    plt.xticks(np.arange(len(labels)), labels, rotation=90)
    plt.yticks(np.arange(len(labels)), labels)
    plt.subplots_adjust(bottom=0.3, left=0.3)

    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.gcf().savefig('plots/{}/{}_confusion_matrix.png'.format(features, model))


def print_precision_recall_fscore(predictions, true_labels, counts, label_mapping):
    p,r,f,s = precision_recall_fscore_support(true_labels, predictions)
    num_classes = len(counts)

    print("\n")
    print("%9s  |   %s  |  %4s  |  %4s  |   %4s   |" % ("CLASS", "CNT", "PR ", "RE ", "F1 "))
    print('-' * 50)
    for c in range(num_classes):
        print("%9s  |  % 4d  |  %.2f  |  %.2f  |  %.3f   |" % (label_mapping[c], counts[c], p[c], r[c], f[c]))
    print('-' * 50)
    print("%9s  |  % 4d  |  %.2f  |  %.2f  |  %.3f   |" % ('average', np.sum(list(counts.values())), np.mean(p), np.mean(r), np.mean(f)))
    print('=' * 50)


def plot_results_table(p, r, f, count, id_class_mapping, num_classes, clf, features):
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

    plt.gcf().savefig('plots/{}/{}_table.pdf'.format(features, clf))


def save_lwlrap_results_table(lwlrap, labels, weights, clf, features):

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

    with open('plots/{}/{}_lwlrap_latex_table.tex'.format(features, clf), "w") as latex_out:
        latex_out.write(latex_table)


def load_preds(file, features):
    return np.load('predictions/{}/{}'.format(features, file)).item()


def make_average_late_fusion(infiles, features, frame=True):
    initial_preds = load_preds(infiles[0], features)
    initial_preds = {k: initial_preds[k]/len(infiles) for k in initial_preds.keys()}
    for infile in infiles[1:]:
        morepreds = load_preds(infile, features)
        for key in initial_preds.keys():
            initial_preds[key] += morepreds[key] / len(infiles)
        del morepreds

    if frame:
        for key in initial_preds.keys():
            initial_preds[key] = np.average(initial_preds[key], axis=0)

    return initial_preds


def plot_per_class_metric(metric, inv_label_mapping, exp_name, features, metric_name):
    labels = [inv_label_mapping[i] for i in range(len(metric))]

    plt.figure(figsize=(20,10))
    plt.subplots_adjust(bottom=0.3)
    plt.bar(labels, metric)
    plt.title('labelwise {} measure'.format(metric_name))
    plt.xlabel('Classes')
    plt.ylabel('{}'.format(metric_name))
    plt.xticks(np.arange(len(labels)), labels, rotation=90)
    # plt.tight_layout()
    plt.gcf().savefig('plots/{}/{}_{}.pdf'.format(features, exp_name, metric_name))


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
    label_mapping, inv_label_mapping = dataloader.get_label_mapping()
    labels = list(label_mapping.keys())

    exp_name = ''
    if len(infiles) > 1:
        # predictions need to be on frame wise level for late fusion
        preds = make_average_late_fusion(infiles, options.features, True if options.level == 'frame' else False)

        if options.test:
            already_listed = []
            for infile in infiles:
                if not infile[:infile.index('f')] in already_listed:
                    exp_name += infile[:infile.index('f')]
                already_listed.append(infile[:infile.index('f')])
            exp_name += 'average_late_fusion_test'
            np.save('predictions/{}/{}'.format(cfg['features'], exp_name), preds)
            return
        else:
            for infile in infiles:
                exp_name += re.match(r'(.*)_predictions.*', infile).group(1) + '_'
            exp_name += 'average_late_fusion'

    else:
        exp_name = re.match(r'(.*)_predictions.*', infiles[0]).group(1)
        preds = load_preds(infiles[0], options.features)

    clf_delta = 0.005
    truth = load_preds(options.truth, options.features)
    true_labels = [l for t in truth.values() for l in np.nonzero(t > 0)[0]]
    true_counts = Counter(true_labels)

    truth_labels = [l for l in truth.values()]
    preds_labels = [l for l in preds.values()]
    per_class_lwlrap, weights = calculate_per_class_lwlrap(np.asarray(truth_labels), np.asarray(preds_labels))
    lwlrap = calculate_overall_lwlrap_sklearn(np.asarray(truth_labels), np.asarray(preds_labels))
    save_lwlrap_results_table(per_class_lwlrap, labels, weights, exp_name, cfg['features'])
    print('Model {} achieved an lwlrap of {}.'.format(infiles[0], lwlrap))

    # After calculating lwlrap we disentangle the multilabel predictions to single label predictions to calculate precision, recall and fscore
    preds_single_label = []
    truth_single_label = []
    for p, t in zip(preds_labels, truth_labels):
        if np.max(p) > 0:
            idxs_p = np.nonzero(p > np.max(p)-clf_delta)[0]
            idxs_t = np.nonzero(t == 1)[0]
            if len(idxs_t) > 1 or len(idxs_p) > 1:
                if len(idxs_p) > len(idxs_t):
                    diff = len(idxs_p)-len(idxs_t)
                    idxs_t = np.pad(idxs_t, (0,diff), mode='constant', constant_values=-1)
                elif len(idxs_t) > len(idxs_p):
                    diff = len(idxs_t) - len(idxs_p)
                    idxs_p = np.pad(idxs_p, (0, diff), mode='constant', constant_values=-1)

                for idx_p, idx_t in zip(idxs_p, idxs_t):
                    if idx_p == -1:
                        preds_single_label.append(dataloader.one_hot_encode([], cfg['num_classes']))
                    else:
                        preds_single_label.append(dataloader.one_hot_encode(idx_p, cfg['num_classes']))
                    if idx_t == -1:
                        truth_single_label.append(dataloader.one_hot_encode([], cfg['num_classes']))
                    else:
                        truth_single_label.append(dataloader.one_hot_encode(idx_t, cfg['num_classes']))
            else:
                preds_single_label.append(dataloader.one_hot_encode(idxs_p, cfg['num_classes']))
                truth_single_label.append(t)
        else:
            preds_single_label.append(p)
            truth_single_label.append(t)

    preds_single_label = [np.argmax(p) for p in preds_single_label]
    truth_single_label = [np.argmax(t) for t in truth_single_label]
    p, r, f, s = precision_recall_fscore_support(truth_single_label, preds_single_label)
    print_precision_recall_fscore(preds_single_label, truth_single_label, true_counts, inv_label_mapping)
    plot_results_table(p, r, f, true_counts, inv_label_mapping, cfg['num_classes'], exp_name, options.features)
    save_confusion_matrix(preds_single_label, truth_single_label, exp_name, options.features, inv_label_mapping)
    plot_per_class_metric(f, inv_label_mapping, exp_name,  options.features, metric_name='fscore')
    plot_per_class_metric(per_class_lwlrap, inv_label_mapping, exp_name, options.features, metric_name='lwlrap')
    # avg_precisions = np.mean([avg_precision(a, p) for a, p in zip(true_labels, preds)])
    # print('Average precision: {}'.format(avg_precisions))


if __name__ == '__main__':
    main()
