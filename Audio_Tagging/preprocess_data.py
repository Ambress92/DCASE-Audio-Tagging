import os
import numpy as np
import tensorflow as tf
import functools
from scipy.io import wavfile
from dataloader import total_label_mapping
from tensorflow.contrib.framework.python.ops import audio_ops as tf_audio

# see https://github.com/DCASE-REPO/dcase2018_baseline/tree/master/task2

BUFFER_SIZE = 10000
SAMPLE_RATE = 44100

def clip_to_waveform(clip, clip_dir=None):
    """
    Decode audio clip (.wav) into a waveform tensor with a value within [-1, +1].

    Parameters
    ----------
    clip : String
        Name of audio clip within clip_dir.
    clip_dir : String
        Path to directory containing clips (referenced by record entry).

    Returns
    -------
    waveform : Tensor
        Waveform tensor with values between -1 and +1.
    """

    clip_path = tf.string_join([clip_dir, clip], separator=os.sep)
    clip_data = tf.read_file(clip_path)
    waveform, sr = tf_audio.decode_wav(clip_data)
    # check sample rate
    check_sr = tf.assert_equal(sr, SAMPLE_RATE)
    # check for mono
    check_channels = tf.assert_equal(tf.shape(waveform)[1], 1)
    with tf.control_dependencies([tf.group(check_sr, check_channels)]):
        return tf.squeeze(waveform)

def clip_to_log_mel_examples(clip, clip_dir=None, hparams=None):
    """
    Decodes audio clip into a batch of log mel spectrum examples.

    Parameters
    ----------
    clip : String
        Name of audio clip within clip_dir.
    clip_dir : String
        Path to directory containing clips (referenced by record entry).
    hparams : tf.contrib.training.HParams
        Hyperparameters of neural network.

    Returns
    -------
    features : Tensor
        Tensor containing a batch of log mel spectrum examples.
    """

    waveform = clip_to_waveform(clip, clip_dir=clip_dir)

    # stft
    window_length_samples = int(round(SAMPLE_RATE * hparams.stft_window_seconds))
    hop_length_samples = int(round(SAMPLE_RATE * hparams.stft_hop_seconds))
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
    magnitude_spectrogram = tf.abs(tf.contrib.signal.stft(signals=waveform,
        frame_length=window_length_samples, frame_step=hop_length_samples,
        fft_length=fft_length))

    # convert to log mel
    num_spectrogram_bins = fft_length // 2 + 1
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins=hparams.mel_bands, num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=SAMPLE_RATE, lower_edge_hertz=hparams.mel_min_hz,
        upper_edge_hertz=hparams.mel_max_hz)
    mel_spectrogram = tf.matmul(magnitude_spectrogram, linear_to_mel_weight_matrix)
    log_mel_spectrogram = tf.log(mel_spectrogram + hparams.mel_log_offset)

    # divide into examples
    spectrogram_sr = 1 / hparams.stft_hop_seconds
    example_window_length_samples = int(round(spectrogram_sr * hparams.example_window_seconds))
    example_hop_length_samples = int(round(spectrogram_sr * hparams.example_hop_seconds))
    features = tf.contrib.signal.frame(signal=log_mel_spectrogram,
        frame_length=example_window_length_samples,
        frame_step=example_hop_length_samples, axis=0)

    return features

def record_to_labeled_log_mel_examples(csv_record, clip_dir=None, hparams=None,
                                       label_mapping=None, num_classes=None):
    """
    Creates log mel spectrum examples for a given training record.

    Parameters
    ----------
    csv_record : String
        A line from the train.csv file.
    clip_dir : String
        Path to a directory containing clips referenced by csv_record.
    hparams : tf.contrib.training.HParams
        Hyperparameters of neural network.
    label_mapping : tf.contrib.lookup.HashTable
        Maps labels to integer identifier, with index: label name.
    num_classes : int
        Number of different available labels.

    Returns
    -------
    features :
        Tensor containing a batch of log mel spectrum examples.
    labels :
        Tensor containing corresponding labels in 1-hot format.
    """

    [clip, label, _] = tf.decode_csv(csv_record, record_defaults=[[''],[''],[0]])
    features = clip_to_log_mel_examples(clip, clip_dir=clip_dir, hparams=hparams)
    class_index = label_mapping.lookup(label)
    label_onehot = tf.one_hot(class_index, num_classes)
    num_examples = tf.shape(features)[0]
    labels = tf.tile([label_onehot], [num_examples, 1])

    return features, labels

def train_input(year, train_csv_path=None, train_clip_dir=None, hparams=None):
    """
    Creates input for training neural networks.

    Parameters
    ----------
    year : int
        Year we want to take the data from.
    train_csv_path : String
        Path to train.csv.
    train_clip_dir : String
        Path to the unzipped audio_train / directory from the
        audio_train.zip file.
    hparams : tf.contrib.training.HParams
        Parameters defining (neural network) model we prepare input for.

    Return
    ------
    features : Tensor
        Tensor containing a batch of log mel spectrum examples.
    labels : Tensor
        Tensor containing corresponding labels in 1-hot format.
    num_classes : int
        Number of different possible labels.
    input_init :
        Initializer op for the iterator that provides features and
        labels, to be run before the input pipeline is read.
    """

    _, num_classes, label_mapping = total_label_mapping(year)

    dataset = tf.data.TextLineDataset(train_csv_path)
    # skip header
    dataset = dataset.skip(1)
    # shuffle, map to log mel
    dataset = dataset.shuffle(buffer_size=BUFFER_SIZE)
    dataset = dataset.map(map_func=functools.partial(
          record_to_labeled_log_mel_examples,
          clip_dir=train_clip_dir, hparams=hparams,
          label_mapping=label_mapping, num_classes=num_classes),
          num_parallel_calls=4)
    dataset = dataset.apply(tf.contrib.data.unbatch())
    dataset = dataset.shuffle(buffer_size=2*BUFFER_SIZE)
    dataset = dataset.repeat(100)
    # get batches
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size=hparams.batch_size))
    dataset = dataset.prefetch(10)
    iterator = dataset.make_initializable_iterator()
    features, labels = iterator.get_next()

    return features, labels, num_classes, iterator.initializer
