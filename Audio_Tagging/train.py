from sklearn.ensemble import RandomForestClassifier
import argparse
from dataloader import get_verified_files_dict
import os
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('-year', required=True)
parser.add_argument('-features', required=True)
parser.add_argument('-clf', 'Classifier to use, by default RF is used', default='RF')
args = parser.parse_args()

def main():
    verified_files_dict = get_veriifed_files(args.year)

    data = []
    filelist = os.listdir('features/{}/{}'.format(args.year, args.features))
    print('loading features ...')
    for file in filelist:
        with open('features/{}/{}/{}'.format(args.year, args.features, file)) as in_file:
            data.append(yaml.load(in_file))

    if args.clf == 'RF':
        pass
    elif args.clf == 'SVM':
        pass
    else:
        pass


if __name__ == '__main__':
    main()