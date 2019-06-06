import numpy as np
from argparse import ArgumentParser
from dataloader import get_label_mapping
import os

parser = ArgumentParser()
parser.add_argument('infile', nargs='+', metavar='INFILE',
                    type=str,
                    help='File to load the predictions from (.npz/.pkl format). '
                         'If given multiple times, predictions will be averaged. '
                         'If ending in ":VALUE", will weight by VALUE.')
parser.add_argument('--outfile', required=True, help='File to store the predictions in',
                    type=str)
parser.add_argument('-features', required=True, type=str, help='For which features to merge test preds')
args = parser.parse_args()

def main():
    pred_dict_list = []
    infiles = args.infile
    infiles = [infile.rsplit(':', 1)[0] for infile in infiles]
    label_mapping, inv_label_mapping = get_label_mapping()

    for pred_file in infiles:
        pred_dict = np.load('predictions/{}/{}'.format(args.features, pred_file)).item()
        pred_dict_list.append(pred_dict)

    merged_file_dict = {}
    for file in pred_dict_list[0].keys():
        file_preds = []
        for p_d in pred_dict_list:
            file_preds.append(p_d[file])
        file_pred = np.average(np.asarray(file_preds), axis=0)
        merged_file_dict[file] = file_pred

    if not os.path.exists('../submissions'):
        os.makedirs('../submissions')

    with open('../submissions/{}'.format(args.outfile), 'w') as out_file:
        out_file.write('fname,' + ','.join(list(label_mapping.keys())) + '\n')
        for file in merged_file_dict.keys():
            out_file.write(file + '.wav,' + ','.join(merged_file_dict[file].astype(np.str)) + '\n')

if __name__ == '__main__':
    main()