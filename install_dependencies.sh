#!/bin/bash

# First check whether anaconda is installed - if not execute the following code lines
# wget https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh
# chmod +x Anaconda3-2018.12-Linux-x86_64.sh
# ./Anaconda3-2018.12-Linux-x86_64.sh
# This installs the Anaconda distribution and then this file can be run

echo "Please be patient, conda environment will be created and dependencies will be installed"

conda create -y -n dcase_audio_tagging python=3.6 scipy pandas h5py cython numpy pyyaml mkl setuptools cmake cffi

source activate dcase_audio_tagging


conda config --add channels conda-forge

conda install -y -c conda-forge tqdm
conda install -y -c conda-forge librosa
conda install -y -c conda-forge ffmpeg


#conda install -y pytorch torchvision cuda91 -c pytorch

conda install -y tensorflow-gpu-base
conda install -y -c anaconda tensorflow-gpu cupy
pip install keras
pip install sed_eval
pip install pynvrtc

# tensorflow gpu, go to: https://www.tensorflow.org/install
#pip install git+https://github.com/lanpa/tensorboard-pytorch
#pip install dcase_util

echo Clean/remove environment using :
echo $ source deactivate
echo $ conda env remove -n dcase_audio_tagging
echo remember always \"source activate dcase_audio_tagging\"
