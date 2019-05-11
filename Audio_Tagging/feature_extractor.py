import os
import librosa
import matplotlib.pyplot as plt
import numpy as np

def plot_spectrogram(spectrogram, title, type):
    print("Spectrogram Shape:", spectrogram.shape)

    plt.figure()
    plt.subplots_adjust(right=0.98, left=0.1, bottom=0.1, top=0.99)
    if type == 'mel':
        plt.imshow(spectrogram, origin="lower", interpolation="nearest", cmap="viridis")
    else:
        plt.imshow(np.abs(spectrogram), origin='lower', interpolation='nearest', cmap='viridis')
    plt.xlabel("Time")
    plt.ylabel("%d bins" % spectrogram.shape[0])
    plt.title(title)
    # plt.colorbar()
    plt.tight_layout()
    plt.gcf().savefig('plots/{}.png'.format(title))


def normalize_features(features):
    normalized_features = (features - np.mean(features)) / np.std(features)
    return normalized_features


def get_cqt_specs(clips, filelist, sr=32000, spec_weighting=False, plot=False, dump=False, mixup=True, fixed_length=2784,
                  test=False, already_saved=False):
    """
    Computes constant Q-transform and dumps results into according directory.

    Parameters
    ----------
    use_train : boolean
        `True` if we want to compute cqt for training audio clips.
        `False` if we want to compute cqt to test clips.
    """

    n_bins = 84
    n_fft = 1024
    hop_length = 192

    spectrograms = []
    if not test:
        audio_clips = clips[0]
    else:
        audio_clips = clips

    if not already_saved:
        for clip, file in zip(audio_clips, filelist):

            cqt = librosa.core.cqt(clip, sr=sr, hop_length=hop_length, n_bins=n_bins, pad_mode='reflect',
                                       fmin=librosa.note_to_hz('A1'))

            # keep only amplitudes
            cqt = np.abs(cqt)

            if cqt.shape[1] < fixed_length:
                # repeat spectrogram and split into frames
                cqt = repeat_spectrogram(cqt, fixed_length=fixed_length)
            else:
                # spectrogram is too long - cut it to fixed length
                cqt = cqt[:, :fixed_length]

            spectrograms.append(cqt)

            if dump:
                if not os.path.exists('../features/cqt/'):
                    os.makedirs('../features/cqt/')
                if not os.path.exists('../features/cqt_weighted/'):
                    os.makedirs('../features/cqt_weighted/')

                if spec_weighting:
                    np.save('../features/cqt_weighted/{}'.format(file.split('.')[0]), cqt)
                else:
                    np.save('../features/cqt/{}'.format(file.split('.')[0]), cqt)

        spectrograms = np.asarray(spectrograms)
    else:
        for file in filelist:
            if spec_weighting:
                spectrograms.append(np.load('../features/cqt_weighted/{}.npy'.format(file.split('.')[0])))
            else:
                spectrograms.append(np.load('../features/cqt/{}.npy'.format(file.split('.')[0])))
        spectrograms = np.asarray(spectrograms)

    if mixup:
        x, labels = mixup_augmentation(spectrograms, clips[1], alpha=0.3)
    elif not test:
        x = spectrograms
        labels = clips[1]
    else:
        x = spectrograms

    spectrograms = []
    for cqt, file in zip(x, filelist):
        if spec_weighting:
            freqs = librosa.cqt_frequencies(n_bins, fmin=librosa.note_to_hz('A1'))
            cqt = librosa.perceptual_weighting(x ** 2, freqs, ref=np.max)

        if plot:
            plot_spectrogram(cqt, 'CQT Spectrogram after silence clipping', 'cqt')

        spec = normalize_features(cqt)
        spectrograms.append(spec)

    if not dump:
        if not test:
            return np.asarray(spectrograms), labels
        else:
            return np.asarray(spectrograms)


def get_mel_specs(clips, filelist, sr=32000, spec_weighting=False, plot=False, dump=False, mixup=True, fixed_length=2784,
                  test=False, already_saved=False):
    """
    Computes Mel-scaled spectrogram and dumps results into according directory.

    Parameters
    ----------
    use_train : boolean
        `True` if we want to compute mel spectrograms for training audio clips.
        `False` if we want to compute mel spectrograms to test clips.
    """

    n_fft = 1024
    n_mels = 128
    hop_length = 192
    fmax = None

    spectrograms = []
    if not test:
        audio_clips = clips[0]
    else:
        audio_clips = clips

    if not already_saved:
        for clip, file in zip(audio_clips, filelist):
            stft = librosa.stft(clip, n_fft=n_fft, hop_length=hop_length, win_length=None, window='hann', center=True,
                                    pad_mode='reflect')

            # keep only amplitudes of spectrograms
            stft = np.abs(stft)

            if stft.shape[1] < fixed_length:
                # repeat spectrogram and split into frames
                stft = repeat_spectrogram(stft, fixed_length=fixed_length)
            else:
                # spectrogram is too long - cut it to fixed length
                stft = stft[:, :fixed_length]

            spectrograms.append(stft)

            if dump:
                if not os.path.exists('../features/mel'):
                    os.makedirs('../features/mel')
                if not os.path.exists('../features/mel_weighted'):
                    os.makedirs('../features/mel_weighted')

                if spec_weighting:
                    np.save('../features/mel_weighted/{}'.format(file.split('.')[0]), stft)
                else:
                    np.save('../features/mel/{}'.format(file.split('.')[0]), stft)

        spectrograms = np.asarray(spectrograms)
    else:
        for file in filelist:
            if spec_weighting:
                spectrograms.append(np.load('../features/mel_weighted/{}.npy'.format(file.split('.')[0])))
            else:
                spectrograms.append(np.load('../features/mel/{}.npy'.format(file.split('.')[0])))
        spectrograms = np.asarray(spectrograms)

    # apply mixup augmentation
    if mixup:
        x, labels = mixup_augmentation(spectrograms, clips[1], alpha=0.3)
    elif not test:
        x = spectrograms
        labels = clips[1]
    else:
        x = spectrograms

    spectrograms = []
    for stft in x:
        if spec_weighting:
            freqs = librosa.core.fft_frequencies(sr=sr, n_fft=n_fft)
            stft = librosa.perceptual_weighting(stft**2, freqs, ref=1.0, amin=1e-10, top_db=99.0)
        else:
            stft = np.log10(stft+1)

        # apply mel filterbank
        spec = librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=n_mels, fmax=fmax)

        if plot:
            plot_spectrogram(spec, 'Mel Spectrogram after silence clipping', 'mel')

        spec = normalize_features(spec)
        spectrograms.append(spec)

    if not dump:
        if not test:
            return np.asarray(spectrograms), labels
        else:
            return np.asarray(spectrograms)

def get_mfcc_features(clips, filelist, sr=32000, spec_weighting=False, plot=False, dump=False, already_saved=False):
    """
    Computes MFCCs and dumps features into according directory.

    Parameters
    ----------
    use_train : boolean
        `True` if we want to compute mfccs for training audio clips.
        `False` if we want to compute mfccs to test clips.
    """

    mfccs = []
    for clip, file in zip(clips, filelist):
        try:
            mfcc = librosa.feature.mfcc(clip.astype(np.float), sr, n_mfcc=40)
        except:
            print('Extraction failed for file {}'.format(file))

        deltas = librosa.feature.delta(mfcc)
        delta_delta = librosa.feature.delta(mfcc, order=2)

        mfcc = np.vstack((mfcc, deltas, delta_delta))
        mfcc = normalize_features(mfcc.T)
        mfccs.append(mfcc)

        if dump:
            if not os.path.exists('../features/mfcc/'):
                os.makedirs('../features/mfcc/')

            np.save('../features/mfcc/{}'.format(file.split('.')[0]), mfcc)
    if not dump:
        return np.asarray(mfccs)

def get_spectral_centroids(clips, filelist, sr=32000, dump=False, already_saved=False):
    """
    Computes spectral centroid of audio samples and dumps results
    into according directory.

    Parameters
    ----------
    use_train : boolean
        `True` if we want to compute spectral centroids for training audio clips.
        `False` if we want to compute spectral centroids to test clips.
    """

    centroids = []
    for clip, file in zip(clips, filelist):
        cts = librosa.feature.spectral_centroid(clip.astype(np.float), sr)

        if dump:
            if not os.path.exists('../features/cts'):
                os.makedirs('../features/cts')

            np.save('../features/{}/centroids/'.format(file.split('.')[0]), cts.T)
        else:
            centroids.append(cts)

    if not dump:
        return np.asarray(cts)

def mixup_augmentation(X, y, alpha=0.3):

    batch_size, h, w = X.shape
    X_l = np.random.beta(alpha, alpha, batch_size)
    X_l = X_l.reshape(batch_size, 1, 1)
    y_l = X_l.reshape(batch_size, 1)


    # mix observations
    X1, X2 = X[:], X[::-1]
    X = X1 * X_l + X2 * (1.0 - X_l)
    one_hot = y

    # mix labels
    y1 = one_hot[:]
    y2 = one_hot[::-1]
    y = y1 * y_l +  y2 * (1.0 - y_l)

    return X.astype(np.float32), y.astype(np.float32)

def repeat_spectrogram(spec, fixed_length):
    if spec.shape[1] < fixed_length:
        while spec.shape[1] < fixed_length:
            spec = np.concatenate((spec, spec), axis=-1)

    if spec.shape[1] > fixed_length:
        spec = spec[:, :fixed_length]

    return spec

def concat_mixup_augmentation(X, y, alpha=0.3, p=0.5):

    batch_size, h, w, c = X.shape
    if np.random.random() < p:
        l = np.random.beta(alpha, alpha, batch_size)
        y_l = l.reshape(batch_size, 1)

        # mix observations
        X1 = X[:]
        X2 = X[::-1]
        w1 = int(w * (1.0-alpha))
        X = np.concatenate((X1[:, :, :w1, :], X2[:,:,w1::,:]), axis=2)

        # mix labels
        one_hot = y
        y1 = one_hot[:]
        y2 = one_hot[::-1]
        y = y1 * y_l + y2 * (1.0 - y_l)

        return X.astype(np.float32), y.astype(np.float32)
    else:
        return X, y
