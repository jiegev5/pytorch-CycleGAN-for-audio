#!/usr/bin/env python
# coding: utf-8

import librosa
import numpy as np
import os
import sys
import time
from glob import glob


def compute_mfcc(y):
    n_fft = 320
    win_length = n_fft
    hop_length = int(n_fft/2)
    window = 'hamming'
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    data, phase = librosa.magphase(D)

    # S = log(S+1)
    data = np.log1p(data)

    ## adding z-score normalize
    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean)/std
    # data = data/(np.abs(data).max())
    print(f'mean is {mean}, std is {std}, shape is {data.shape}')
    return data.reshape(1, -1, 601)

in_len = 16000*6
normalize = False
for_cyclegan = True
# root = "/data1/wenjie/github/honk/speech_dataset/house"
dict_ = {
        # "/data1/wenjie/github/honk/speech_dataset/house":\
        # "/data1/wenjie/github/pytorch-CycleGAN-and-pix2pix/datasets/original2ATR_spec_minmax"
        # "/data1/wenjie/github/honk/speech_dataset_wenjie/meetingroom/loc2-45cm-10cmd/ATR/house":\
        # "/data1/wenjie/github/pytorch-CycleGAN-and-pix2pix/datasets/original2ATR_spec_minmax"
        # "/data1/wenjie/github/honk/speech_dataset_wenjie/meetingroom/loc2-45cm-10cmd/maono/house":\
        # "/data1/wenjie/github/pytorch-CycleGAN-and-pix2pix/datasets/original2maono_minmax",
        # "/data1/wenjie/github/honk/speech_dataset_wenjie/meetingroom/loc2-45cm-10cmd/clipon/house":\
        # "/data1/wenjie/github/pytorch-CycleGAN-and-pix2pix/datasets/original2clipon_minmax",
        # "/data1/wenjie/github/honk/speech_dataset_wenjie/meetingroom/loc2-45cm-10cmd/USB/house":\
        # "/data1/wenjie/github/pytorch-CycleGAN-and-pix2pix/datasets/original2USB_minmax",
        # "/data1/wenjie/github/honk/speech_dataset_wenjie/meetingroom/loc2-45cm-10cmd/USBplug/house":\
        # "/data1/wenjie/github/pytorch-CycleGAN-and-pix2pix/datasets/ATR2USBplug_spec_minmax"
        "/data2/wenjie/deepspeech.pytorch/librispeech/test_wj/cyclegan_train_1100/combined",
        "/data1/wenjie/github/pytorch-CycleGAN-and-pix2pix/datasets/deepspeech_original2maono_extended"
}
# root = "/data1/wenjie/github/honk/speech_dataset/house"
# target_root = "/data1/wenjie/github/pytorch-CycleGAN-and-pix2pix/datasets/original2ATR_minmax"
# target_root = "/data1/wenjie/github/honk/speech_dataset_wenjie/meetingroom/loc2-45cm-10cmd-v1/USB_minmax"
# root = "/data1/wenjie/github/honk/speech_dataset_wenjie/original_testset"
# target_root = "/data1/wenjie/github/honk/speech_dataset_wenjie/original_testset_npy_no_norm"
s = time.time()
for key in dict_.keys():
    root = key
    target_root = dict_[key]
    if not os.path.exists(target_root):
        os.mkdir(target_root)

    train_target_path = os.path.join(target_root,'trainA')
    test_target_path = os.path.join(target_root,'testA')
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
        # sys.exit()
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
