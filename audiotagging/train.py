import os
import keras

from audiotagging.models import cvssp
from audiotagging.data import DcaseAudioTagging


def lwlrap(y_true, y_pred):     # TODO: needs a fix
    import keras.backend as K
    _, num_classes = K.int_shape(y_pred)
    # K.arange hack
    idx = K.cumsum(K.ones_like(y_pred, dtype='int32'))
    # sort predicted probabilities
    ordering = K.tf.argsort(y_pred, axis=-1)[:, ::-1]
    reorder = K.tf.argsort(ordering, axis=-1)
    # sort positive labels
    cols = K.stack([idx, ordering], axis=-1)
    hits = K.tf.gather_nd(y_true, cols) > 0
    cumulative_hits = K.cumsum(K.cast(hits, K.floatx()), axis=-1)
    precisions = cumulative_hits / (1 + K.arange(num_classes, dtype=K.floatx()))
    # resort precisions back to original order and drop negatives
    cols = K.stack([idx, reorder], axis=-1)
    hit_precisions = y_true * K.tf.gather_nd(precisions, cols)
    # number of positive labels
    hit_counts = K.sum(K.cast(y_true > 0, K.floatx()), axis=0)
    weights = hit_counts / K.sum(hit_counts)
    lrap = K.sum(hit_precisions, axis=0) / K.maximum(1., hit_counts)
    return K.sum(weights * lrap)


if __name__ == '__main__':
    mel_len = 3000
    epochs = 50

    for train_fold, valid_fold in zip(
        ['fold{:d}_train'.format(i) for i in range(1, 5)],
        ['fold{:d}_eval'.format(i) for i in range(1, 5)]
    ):
        train = DcaseAudioTagging(os.path.join('cv', train_fold), curated=True,
                                  feature_length=mel_len, path='../datasets')
        valid = DcaseAudioTagging(os.path.join('cv', valid_fold), curated=True, feature_length=mel_len,
                                  shuffle=False, path='../datasets')

        checkpoint = keras.callbacks.ModelCheckpoint('../savemodels/cvssp.hdf5',
                                                     save_best_only=True, save_weights_only=True)
        tensorboard = keras.callbacks.TensorBoard('../logs/')
        cbs = [checkpoint, tensorboard]
        model = cvssp.get_model([128, mel_len, 1], train.num_classes)
        model.compile('adam', 'binary_crossentropy', metrics=[lwlrap])
        model.fit_generator(train, epochs=epochs, validation_data=valid, workers=8, callbacks=cbs)
