#!/usr/bin/env python
# coding: utf-8

import librosa
import numpy as np
import os
import sys
import time
from glob import glob


def compute_mfcc(data):
    data = librosa.feature.melspectrogram(
        data,
        sr=16000, # self.sr,
        n_mels=40, # self.n_mels,
        hop_length=160, # self.hop_length,
        n_fft=480, # self.n_fft,
        fmin=20, # self.f_min,
        fmax=4000) # self.f_max)
    data[data > 0] = np.log(data[data > 0])
    data = [np.matmul(librosa.filters.dct(40, 40), x) for x in np.split(data, data.shape[1], axis=1)]
    data = np.array(data, order="F").astype(np.float32)
    ## adding z-score normalize
    mean = np.mean(data)
    std = np.std(data)
    print(f'mean is {mean}, std is {std}')
    data = (data - mean)/std
    # scaled to -1 and 1
    data = data/(np.abs(data).max())
    # max_ = data.max()
    # min_ = data.min()
    # data = (data - min_)/(max_-min_)
    # data = data*2-1
    return data.reshape(1, -1, 40)


in_len = 16000
normalize = False
for_cyclegan = True
dict_ = {
        "/your/path/to/audio folder":\
        "/your/path/to/destination"
}

s = time.time()
for key in dict_.keys():
    root = key
    target_root = dict_[key]
    if not os.path.exists(target_root):
        os.mkdir(target_root)

    train_target_path = os.path.join(target_root,'trainB')
    test_target_path = os.path.join(target_root,'testB')
    if not os.path.exists(train_target_path):
        os.mkdir(train_target_path)
    if not os.path.exists(test_target_path):
        os.mkdir(test_target_path)

    
    audio_list = glob(root+'/*.wav')
    for num in range(len(audio_list)):
        audio = audio_list[num]
        data = librosa.core.load(audio, sr=16000)[0]
        data = np.pad(data, (0, max(0, in_len - len(data))), "constant") # pad to 1s
        mfcc = compute_mfcc(data)
        chk = np.sum(mfcc)
        if np.isnan(chk):
            print(f'mean = {mfcc.mean()} min = {mfcc.min()} max = {mfcc.max()}')
            continue
        else:
            name = audio.split('/')[-1].split('.')[0] + '.npy'
            if num < len(audio_list)*0.9:
                new_name = os.path.join(train_target_path,name)
            else:
                new_name = os.path.join(test_target_path,name)
            print(num,"saving ",name)
            np.save(new_name,mfcc)
e = time.time()
print("processing time (min): ",(e-s)/60)
