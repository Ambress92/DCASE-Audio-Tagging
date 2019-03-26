from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import os
import argparse
from dataloader import get_verified_files_dict, load_verified_files, get_label_mapping, one_hot_encode
import yaml
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('-year', required=True)
parser.add_argument('-features', required=True)
parser.add_argument('-clf', help='Classifier to use, by default RF is used', default='RF')
args = parser.parse_args()

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

    if args.clf == 'RF':
        clf = RandomForestClassifier(n_estimators=20, verbose=2, max_depth=200)
        clf.fit(X, y)

        if not os.path.exists('../models'):
            os.makedirs('../models')

        with open('../models/RF_verified.yml', 'w') as out_file:
            yaml.dump(clf, out_file)

    elif args.clf == 'SVM':

        print('beginning SVM')
        clf = SVC(C=1.0, kernel='rbf', verbose=True, max_iter=50, probability=True)
        clf.fit(X, y)

        if not os.path.exists('../models'):
            os.makedirs('../models')

        with open('../models/SVM_verified.yml', 'w') as out_file:
            yaml.dump(clf, out_file)
    else:
        pass


if __name__ == '__main__':
    main()
