import os
import numpy as np

from keras.models import load_model
from audiotagging.train import lwlrap
from audiotagging.data import DcaseAudioTesting

if __name__ == '__main__':
    mel_len = 3000

    # prep data
    test_files = np.array([file.rsplit('.wav')[0] for file in os.listdir('../datasets/test')
                           if 'clipped' not in file])
    test = DcaseAudioTesting(test_files, path='../datasets', feature_length=mel_len)
    # load model, get predictions
    model = load_model('../savemodels/cvssp.hdf5', custom_objects={'lwlrap': lwlrap})
    predictions = model.predict_generator(test)

    # write predictions
    if not os.path.exists('../submissions'):
        os.makedirs('../submissions')

    with open('../submissions/cvssp.csv', 'w') as of:
        of.write('fname,' + ','.join(test.label_map.keys()) + '\n')
        for i in range(len(predictions)):
            of.write(test.filenames[i] + '.wav,' + ','.join(predictions[i].astype(np.str)) + '\n')

