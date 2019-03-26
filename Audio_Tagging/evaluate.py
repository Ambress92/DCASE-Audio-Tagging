# https://github.com/DCASE-REPO/dcase2018_baseline
import argparse
import yaml
import numpy as np
from dataloader import load_test_files, get_label_mapping
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser()
parser.add_argument('-year', required=True)
parser.add_argument('-features', required=True)
parser.add_argument('-clf', help='Classifier to use, by default RF is used', default='RF')
args = parser.parse_args()

TOP_N = 3

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
        with open('models/RF_verified.yml', 'r') as in_file:
            params = yaml.load(in_file)
        clf = RandomForestClassifier()
        clf.set_params(params)
        predictions = clf.predict_proba(X)
    elif args.clf == 'SVM':
        with open('models/SVM_verified.yml', 'r') as in_file:
            params = yaml.load(in_file)
        clf = SVM()
        clf.set_params(params)
        predictions = clf.predict_proba(X)
    else:
        pass

    # compute mean average label precision
    # print([predictions[slc] for slc in audio_splits])
    predictions = [get_top_predicted_classes(predictions[slc]) for slc in audio_splits]
    map = [avg_precision(a, p) for a, p in zip(y, predictions)]
    print(map)
    map = np.mean(map)
    print('Model {} achieved an average label precision of {}.'.format(args.clf, map))

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
