import argparse
import tensorflow as tf
import sys
from evaluate import eval_cnn
from cnn_models import train

# see https://github.com/DCASE-REPO/dcase2018_baseline/tree/master/task2

def parse_flags():
    """
    Parses and returns input flags.
    """
    parser = argparse.ArgumentParser()

    # common flags
    all_modes_group = parser.add_argument_group('Common flags')
    all_modes_group.add_argument('--mode', type=str, choices=['train', 'eval', 'inference'],
        required=True, help='Run one of training, evaluation, or inference.')
    all_modes_group.add_argument('--model', type=str, choices=['baseline'],
        default='baseline', required=False,
        help='Name of a model architecture. Currently, only "baseline" possible.')
    all_modes_group.add_argument('--hparams', type=str, default='',
        help='Model hyperparameters in comma-separated name=value format.')
    all_modes_group.add_argument('--year', type=int, default=2018, required=False,
        help='Year of the data we want to process')

    # flags for training
    training_group = parser.add_argument_group('Flags for training only')
    training_group.add_argument('--train_clip_dir', type=str,
        default='../datasets/2018/audio_train/',
        help='Path to training-clips-directory.')
    training_group.add_argument('--train_csv_path', type=str,
        default='../datasets/2018/train.csv',
        help='Path to CSV file containing training clip filenames and labels.')
    training_group.add_argument('--train_dir', type=str, default='models/',
        help='Path to a directory which will hold model checkpoints and other outputs.')

    # evaluation flags
    eval_group = parser.add_argument_group('Flags for evaluation only')
    eval_group.add_argument('--eval_clip_dir', type=str,
        default='../datasets/2018/audio_test/',
        help='Path to directory containing evaluation clips.')
    eval_group.add_argument('--eval_csv_path', type=str,
        default='../datasets/2018/test.csv',
        help='Path to CSV file containing evaluation clip filenames and labels.')
    eval_group.add_argument('--checkpoint_path', type=str,
        default='', help='Path to a model checkpoint to use for evaluation.')

    flags = parser.parse_args()

    try:
        if flags.mode == 'train':
            assert flags.train_clip_dir, 'Must specify --train_clip_dir'
            assert flags.train_csv_path, 'Must specify --train_csv_path'
            assert flags.train_dir, 'Must specify --train_dir'
        elif flags.mode == 'eval':
            assert flags.checkpoint_path, 'Must specify --checkpoint_path'
            assert flags.eval_clip_dir, 'Must specify --eval_clip_dir'
            assert flags.eval_csv_path, 'Must specify --eval_csv_path'
    except AssertionError as e:
        print('\nError: ', e, '\n')
        parser.print_help()
        sys.exit(1)

    return flags

def parse_hparams(flag_hparams):
    """
    Parses and returns hyperparameters of specified model.
    """

    # defaul values
    hparams = tf.contrib.training.HParams(
        # window and hop length for STFT
        stft_window_seconds=0.025,
        stft_hop_seconds=0.010,
        # spectrogram to mel spectrogram parameters
        mel_bands=64,
        mel_min_hz=125,
        mel_max_hz=7500,
        # log mel spectrogram = log(mel-spectrogram + mel_log_offset)
        mel_log_offset=0.001,
        # window and hop length to frame the log mel spectrogram into examples
        example_window_seconds=0.250,
        example_hop_seconds=0.125,
        # number of examples in a batch
        batch_size=64,
        # for MLP, nl=# layers, nh=# units per layer
        #nl=2,
        #nh=256,
        # SD of normal distribution to initialize the weights of the model
        weights_init_stddev=1e-3,
        # learning rate
        lr=1e-4,
        # epsilon for Adam optimiser
        adam_eps=1e-8,
        # classifier layer: softmax or logistic
        classifier='softmax')

    # flags can override default hparam values
    hparams.parse(flag_hparams)
    return hparams

def main():
    # parse flags and hyperparameters
    flags = parse_flags()
    hparams = parse_hparams(flags.hparams)

    if flags.mode == 'train':
        train(model_name=flags.model, year=flags.year, hparams=hparams,
              train_csv_path=flags.train_csv_path,
              train_clip_dir=flags.train_clip_dir,
              train_dir=flags.train_dir)
    elif flags.mode == 'eval':
        eval_cnn(model_name=flags.model, year=flags.year, hparams=hparams,
                 eval_csv_path=flags.eval_csv_path, eval_clip_dir=flags.eval_clip_dir,
                 checkpoint_path=flags.checkpoint_path)

if __name__ == '__main__':
    main()
