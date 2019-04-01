# https://github.com/DCASE-REPO/dcase2018_baseline
import argparse
import numpy as np
import tensorflow as tf
import random
import sys
from dataloader import load_test_files, get_label_mapping, total_label_mapping
from preprocess_data import record_to_labeled_log_mel_examples
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import itertools
from collections import Counter
from cnn_models import define_model
from collections import defaultdict

# parser = argparse.ArgumentParser()
# parser.add_argument('-year', required=True)
# parser.add_argument('-features', required=True)
# parser.add_argument('-clf', help='Classifier to use, by default RF is used', default='RF')
# args = parser.parse_args()

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

# def main():
    # load and prep test data
    # print('Loading test clips...')
    # test_data = load_test_files(args.year, args.features)
    # label_mapping, _ = get_label_mapping(args.year)
    #
    # X = []
    # y = []
    # audio_splits = []
    # start = 0
    # for x in test_data:
    #     y.append(label_mapping[x[1]])
    #     x = x[0]
    #     audio_splits.append(slice(start, start+len(x)))
    #     start = start + len(x)
    #     for datapoint in x:
    #         X.append(datapoint)
    #
    # # reconstruct models and classify test clips
    # if args.clf == 'RF':
    #     clf = np.load('models/RF_verified.npy')
    #     predictions = clf.predict_proba(X)
    # elif args.clf == 'SVM':
    #     clf = np.load('models/RF_verified.npy')
    #     predictions = clf.predict_proba(X)
    # else:
    #     pass
    #
    # # compute mean average label precision
    # predictions = [get_top_predicted_classes(predictions[slc]) for slc in audio_splits]
    # avg_precisions = np.mean([avg_precision(a, p) for a, p in zip(y, predictions)])
    # print('Model {} achieved an average label precision of {}.'.format(args.clf, avg_precisions))
    # # show additional metrics
    # print_precision_recall_fscore(predictions, y)

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

def eval_cnn(model_name, year, hparams=None, eval_csv_path=None,
             eval_clip_dir=None, checkpoint_path=None):
    """
    Runs defined model in evaluation-mode and checks how
    well it performs for defined validation data.

    Paramaters
    ----------
    model_name : String
        Defines which model we want to train. Currently only supports 'baseline'.
    year : int
        Year of the data we want to work with.
    hparams : tf.contrib.training.HParams
        Hyperparameters of model to be evaluated.
    eval_csv_path : String
        Path to test.csv file.
    eval_clip_dir : String
        Path to directory containing audio clips for evaluation.
    checkpoint_path : String
        Path to checkpoints of the model obtained during training.
    """

    print('\nEvaluation for model:{} with hparams:{}'.format(model_name, hparams))
    print('Evaluation data: clip dir {} and labels {}'.format(eval_clip_dir, eval_csv_path))
    print('Checkpoint: {}\n'.format(checkpoint_path))

    with tf.Graph().as_default():
        label_mapping, class_map, num_classes = total_label_mapping(year)
        csv_record = tf.placeholder(tf.string, [])

        # features are batch of all framed log mel spectrum examples of a clip
        # labels contain a batch of identical 1-hot vectors
        features, labels = record_to_labeled_log_mel_examples(csv_record,
                                clip_dir=eval_clip_dir, hparams=hparams,
                                label_mapping=label_mapping,
                                num_classes=num_classes)

        # create model (NOT in training mode)
        global_step, prediction, _, _ = define_model(
                model_name=model_name, features=features, num_classes=num_classes,
                hparams=hparams, training=False)

        with tf.train.SingularMonitoredSession(checkpoint_filename_with_path=checkpoint_path) as sess:
            # counters to aid printing
            ap_counts = defaultdict(int) # maps class index to the number of clips with that label
            ap_sums = defaultdict(float) # maps class index to the sum of AP for all clips with that label

            # read validation csv, skip header and shuffle
            eval_records = open(eval_csv_path).readlines()[1:]
            random.shuffle(eval_records)

            for (i, record) in enumerate(eval_records):
                record = record.strip()
                actual, predicted = sess.run([labels, prediction], {csv_record: record})
                # actual consists of identical rows (same clip), get this class with np.argmax()
                actual_class = np.argmax(actual[0])
                predicted_classes = get_top_predicted_classes(predicted)
                # compute AP
                ap = avg_precision(actual=actual_class, predicted=predicted_classes)
                ap_counts[actual_class] += 1
                ap_sums[actual_class] += ap
                print(class_map[actual_class], [class_map[index] for index in predicted_classes], ap)

                # print per-class and overall AP from time to time
                if i % 50 == 0:
                    print_maps(ap_sums=ap_sums, ap_counts=ap_counts, class_map=class_map)
                sys.stdout.flush()

            # print final results
            print_maps(ap_sums=ap_sums, ap_counts=ap_counts, class_map=class_map)


# if __name__ == '__main__':
#     main()
