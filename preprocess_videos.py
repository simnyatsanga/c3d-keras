#!/usr/bin/env python

import os
import copy
import cv2
import numpy as np
import sys
import glob
from pickle import dump
from pickle import load
import c3d_model
from keras.models import Model
from keras.models import model_from_json
# from keras.layers.core import Dense, Dropout, LSTM, Input, Embedding
from keras.layers.merge import add
import keras.backend as K
dim_ordering = K.image_dim_ordering()
print ("[Info] image_dim_order (from default ~/.keras/keras.json)={}".format(
        dim_ordering))
backend = dim_ordering

def load_pretrained_c3d():
    show_images = False
    diagnose_plots = False
    model_dir = './models'
    global backend

    # override backend if provided as an input arg
    if len(sys.argv) > 1:
        if 'tf' in sys.argv[1].lower():
            backend = 'tf'
        else:
            backend = 'th'
    print ("[Info] Using backend={}".format(backend))

    if backend == 'th':
        model_weight_filename = os.path.join(model_dir, 'sports1M_weights_th.h5')
        model_json_filename = os.path.join(model_dir, 'sports1M_weights_th.json')
    else:
        model_weight_filename = os.path.join(model_dir, 'sports1M_weights_tf.h5')
        model_json_filename = os.path.join(model_dir, 'sports1M_weights_tf.json')

    print("[Info] Reading model architecture...")
    model = model_from_json(open(model_json_filename, 'r').read())
    #model = c3d_model.get_model(backend=backend)

    # visualize model
    model_img_filename = os.path.join(model_dir, 'c3d_model.png')
    if not os.path.exists(model_img_filename):
        from keras.utils import plot_model
        plot_model(model, to_file=model_img_filename)

    print("[Info] Loading model weights...")
    model.load_weights(model_weight_filename)
    print("[Info] Loading model weights -- DONE!")
    model.compile(loss='mean_squared_error', optimizer='sgd')

    return model


def extract_video_features(model):
    # sample_videos = ['tBozgYVgeDE.mp4', 'TKEbws4QhEk.mp4', 'dM06AMFLsrc.mp4', 'CUujE52na_c.mp4']
    # sample_videos = ['tBozgYVgeDE.mp4']
    videos = [vid for vid in glob.glob('data/videos/*')]
    video_features = {}

    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-3].output)

    for video in videos:
        print("[Info] Processing sample video %s ..." % (video))
        cap = cv2.VideoCapture(video)
        vid = []
        # extract and resize frames from video
        while True:
            ret, img = cap.read()
            if not ret:
                break
            vid.append(cv2.resize(img, (171, 128)))
        vid = np.array(vid, dtype=np.float32)

        # sample 16-frame clip
        #start_frame = 100
        start_frame = 0
        X = vid[start_frame:(start_frame + 16), :, :, :]
        # subtract mean
        mean_cube = np.load('models/train01_16_128_171_mean.npy')
        mean_cube = np.transpose(mean_cube, (1, 2, 3, 0))

        X -= mean_cube

        # center crop
        X = X[:, 8:120, 30:142, :] # (l, h, w, c)

        if backend == 'th':
            X = np.transpose(X, (3, 0, 1, 2)) # input_shape = (3,16,112,112)
        else:
            pass

        print("[Info] Extracting feature for sample video %s ..." % (video))
        # extraction yields a (1, 4096) shaped vector so need to collapse to a 1D (4096,) vector
        video_feature = model.predict_on_batch(np.array([X]))
        video_feature = np.squeeze(video_feature)
        video_features[video.split("/")[-1].split(".")[0]] = video_feature
    return video_features


def main():
    model = load_pretrained_c3d()
    print("[Info] Extraction video features...")
    video_features = extract_video_features(model)
    print("[Info] Extracted features %s ..." % (len(video_features)))
    dump(video_features, open('video_features.pkl', 'wb'))
    print("[Info] Loading labels...")
    with open('sports1m/labels.txt', 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    print('Total labels: {}'.format(len(labels)))


if __name__ == '__main__':
    main()
