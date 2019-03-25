from sklearn.ensemble import RandomForestClassifier
import argparse
from dataloader import get_verified_files_dict, load_verified_files
import os
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('-year', required=True)
parser.add_argument('-features', required=True)
parser.add_argument('-clf', help='Classifier to use, by default RF is used', default='RF')
args = parser.parse_args()

def main():
    verified_files_dict = get_verified_files_dict(args.year)

    data = load_verified_files(args.year, args.features)

    if args.clf == 'RF':
        clf = RandomForestClassifier(n_estimators=100)
    elif args.clf == 'SVM':
        pass
    else:
        pass


if __name__ == '__main__':
    main()