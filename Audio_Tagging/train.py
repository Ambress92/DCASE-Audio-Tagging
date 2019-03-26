from sklearn.ensemble import RandomForestClassifier
import argparse
from dataloader import get_verified_files_dict, load_verified_files, get_label_mapping, one_hot_encode
import yaml
import numpy as np

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

    if args.clf == 'RF':
        clf = RandomForestClassifier(n_estimators=100, verbose=1, n_jobs=-2)
        clf.fit(X, y)
        params = clf.get_params()

        with open('models/RF_verified.yml', 'w') as out_file:
            yaml.dump(params, out_file)

    elif args.clf == 'SVM':
        pass
    else:
        pass


if __name__ == '__main__':
    main()