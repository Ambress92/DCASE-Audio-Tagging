import os
import dataloader
import feature_extractor
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-features', type=str, help='extract mel features', required=True)
parser.add_argument('--spec_weighting', help='extract mel features', action='store_true')
parser.add_argument('--noisy', help='extract features of noisy set', action='store_true')
parser.add_argument('--test', help='extract features of test set', action='store_true')
parser.add_argument('-sr', help='Extract features with sample rate', default=32000)
args = parser.parse_args()

def main():
    print('Load and clip audio files')
    if args.noisy:
        filelist = dataloader.get_verified_files_dict().keys()
        filelist = [f.rstrip().replace('.wav', '') for f in filelist]
        files = dataloader.load_unverified_files(filelist, args.sr, features=args.features, silence_clipping=True,
                                                 already_saved=False)

    elif args.test:
        filelist = dataloader.get_test_files_list()
        filelist = [f.rstrip().replace('.wav', '') for f in filelist]
        files = dataloader.load_test_files(filelist, args.sr, features=None, silence_clipping=True, already_saved=False)
    else:
        filelist = dataloader.get_verified_files_dict().keys()
        filelist = [f.rstrip().replace('.wav', '') for f in filelist]
        files = dataloader.load_verified_files(filelist, args.sr, features=args.features, silence_clipping=True,
                                                 already_saved=False)

    if not os.path.exists('../features/{}'.format(args.features)):
        os.makedirs('../features/{}'.format(args.features))

    print('Extract and dump features...')
    if args.features == 'mel':
        feature_extractor.get_mel_specs(files, filelist, sr=32000, spec_weighting=args.spec_weighting, plot=False, dump=True, mixup=True,
                      fixed_length=2784, test=args.test, already_saved=False)
    elif args.features == 'cqt':
        feature_extractor.get_cqt_specs(files, filelist, sr=32000, spec_weighting=args.spec_weighting, plot=False,
                                        dump=True, mixup=True,
                                        fixed_length=2784, test=args.test, already_saved=False)
if __name__ == '__main__':
    main()