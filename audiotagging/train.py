import os
import keras

from keras.callbacks import TensorBoard
from audiotagging.models import cvssp
from audiotagging.data import DcaseAudioTagging


class TrainValidTB(TensorBoard):
    # see https://stackoverflow.com/questions/47877475/keras-tensorboard-plot-train-and-validation-scalars-in-a-same-figure?rq=1
    def __init__(self, path='../logs/', **kwargs):
        # create subdirectories 'train' and 'valid'
        train_log_dir = os.path.join(path, 'train')
        super(TrainValidTB, self).__init__(train_log_dir, **kwargs)
        self.val_log_dir = os.path.join(path, 'valid')

    def set_model(self, model):
        import keras.backend as K
        # setup writer for validation metrics
        self.val_writer = K.tf.summary.FileWriter(self.val_log_dir)
        super(TrainValidTB, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        import keras.backend as K
        # handle validation logs separately
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = K.tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # remaining logs are passed to super.on_epoch_end
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValidTB, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValidTB, self).on_train_end(logs)
        self.val_writer.close()


def lwlrap(y_true, y_pred): 
    import keras.backend as K
    _, num_classes = K.int_shape(y_pred)
    # K.arange hack
    idx = K.cumsum(K.ones_like(y_pred, dtype='int32'), axis=0) - 1
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
    epochs = 20

    for train_fold, valid_fold in zip(
        ['fold{:d}_train'.format(i) for i in range(1, 5)],
        ['fold{:d}_eval'.format(i) for i in range(1, 5)]
    ):
        train = DcaseAudioTagging(os.path.join('cv', train_fold), curated=True,
                                  feature_length=mel_len, path='../datasets')
        valid = DcaseAudioTagging(os.path.join('cv', valid_fold), curated=True, feature_length=mel_len,
                                  shuffle=False, path='../datasets')

        checkpoint = keras.callbacks.ModelCheckpoint('../savemodels/cvssp.hdf5')
        tensorboard = TrainValidTB('../logs/')
        cbs = [checkpoint, tensorboard]
        model = cvssp.get_model([128, mel_len, 1], train.num_classes)
        model.compile('adam', 'binary_crossentropy', metrics=[lwlrap])
        model.fit_generator(train, epochs=epochs, validation_data=valid, workers=8, callbacks=cbs)
