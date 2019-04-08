from sklearn.ensemble import RandomForestClassifier
import argparse
from dataloader import get_label_mapping, load_features, get_total_file_dict
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

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

    print('Loading data...')
    with open('../datasets/cv/fold1_train') as in_file:
        filelist_train = in_file.readlines()

    with open('../datasets/cv/fold1_eval') as in_file:
        filelist_eval = in_file.readlines()

    X_train, y_train = load_features(filelist_train, args.features, num_classes)
    X_eval, y_eval = load_features(filelist_eval, args.features, num_classes)

    print('Load complete')
    clf = RandomForestClassifier(n_estimators=20, verbose=2, max_depth=None, n_jobs=-1)
    clf.fit(X_train, y_train)

    # load and prep test data
    print('Predicting')
    predictions = clf.predict(X_eval)

    print('Dumping Predictions')
    np.save('predictions/RF_predictions', predictions)

if __name__ == '__main__':
    main()
