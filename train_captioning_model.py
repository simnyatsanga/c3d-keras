import json
import argparse
import random
import time
import cv2
import numpy as np
from numpy import array
from numpy import argmax
from pickle import load, dump
import preprocess_captions
import captioning_model
from nltk.translate.bleu_score import corpus_bleu
from keras.utils import to_categorical
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint


def load_captions(input_file):
    with open(input_file, 'rb') as jsonfile:
        captions = json.loads(jsonfile.read())
    return captions

def load_features(filename):
    return load(open(filename, 'rb'))

def load_pretrained_model(filename):
    return load_model(filename)

def load_random_video_caption_pair(features):
    # load validation caption keys
    validation_captions = load_captions('validation_captions.json')
    validation_caption_keys = list(validation_captions.keys())
    random.shuffle(validation_caption_keys)
    # choose a random key using a randomly generated index
    random_key = validation_caption_keys[random.randint(0,len(validation_caption_keys))]
    captions = validation_captions[random_key]
    video = features[random_key]
    return random_key, video, captions

def play_video(video_name):
    cap = cv2.VideoCapture(video_name)
    print("="*47)
    print("Playing video named: %s" % (video_name))
    print("="*47)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        #  NOTE: The COLOR_BGR2GRAY transformation can be used to convert the color frame into gray-scale
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        time.sleep(0.025)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def create_sequences(tokenizer, max_length, caption_list, video, vocab_size):
    X1, X2, y = list(), list(), list()
    # walk through each caption for the video
    for caption in caption_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([caption])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(video)
            X2.append(in_seq)
            y.append(out_seq)
    return array(X1), array(X2), array(y)

# Generate a description for a video
def generate_caption(model, tokenizer, video, max_length):
    # HACK to make video look like (1, 4096). Will fix in preprocess_videos
    video = np.array([video])

    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([video, sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text

# Data generator, intended to be used in a call to model.fit_generator()
def data_generator(captions, videos, tokenizer, max_length, vocab_size):
    # loop for ever over videos
    while 1:
        for key, captions_list in captions.items():
            # retrieve the video feature
            video = videos[key]
            in_video, in_seq, out_word = create_sequences(tokenizer, max_length, captions_list, video, vocab_size)
            yield [[in_video, in_seq], out_word]

def train(vocab_size, training_captions, validation_captions, all_features, tokenizer, max_length):
    # define the model
    model = captioning_model.get_model(vocab_size, max_length)

    # load checkpoint
    # model = load_pretrained_model('weights-improvement-10-0.35.hdf5')

    # train the model, run epochs manually and save after each epoch
    epochs = 100
    training_steps = len(training_captions)
    validation_steps = len(validation_captions)

    # Simple implementation of training loop that saves all models with no validation accuracy check
    training_generator = data_generator(training_captions, all_features, tokenizer, max_length, vocab_size)

    for i in range(epochs):
        print('Running epoch %d' %i)
        # create the data generator
        model.fit_generator(training_generator, epochs=1, steps_per_epoch=training_steps, verbose=1)
        # save model
        model.save('model_stride_1_' + str(i) + '.h5')

    # Using training and validation generator so that the model is only saved if the validation accuracy improves
    # training_generator = data_generator(training_captions, all_features, tokenizer, max_length, vocab_size)
    # validation_generator = data_generator(validation_captions, all_features, tokenizer, max_length, vocab_size)
    #
    # filepath="weights-improvement-stride_16-{epoch:02d}-{val_acc:.2f}.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # callbacks_list = [checkpoint]
    # # fit for one epoch
    # model.fit_generator(training_generator,
    #                     epochs=100,
    #                     steps_per_epoch=training_steps,
    #                     validation_data=validation_generator,
    #                     validation_steps=validation_steps,
    #                     callbacks=callbacks_list,
    #                     verbose=1)

# Evaluate the skill of the model
def evaluate(model, captions, videos, tokenizer, max_length):
    actual, predicted = list(), list()
    # step over the whole set
    for key, captions_list in captions.items():
        # generate description
        yhat = generate_caption(model, tokenizer, videos[key], max_length)
        # store actual and predicted
        references = [caption.split() for caption in captions_list]
        actual.append(references)
        predicted.append(yhat.split())

    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

    result_string = 'BLEU-1: %f\n' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    result_string += 'BLEU-2: %f\n' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    result_string += 'BLEU-3: %f\n' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0))
    result_string += 'BLEU-4: %f\n' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))

    with open('results.txt', 'w+') as result_file:
        result_file.write(result_string)

# Play videos and generate captions using pre-selected videos
def demo(model, all_features, tokenizer, max_length):
    validation_captions = load_captions('validation_captions.json')
    good_video_keys = ['6BrHPMdyVtU_1_10', '-_hbPLsZvvo_19_26', 'J_evFB7RIKA_104_120', 'Uc63MFVwfrs_355_372', 'aC-KOYQsIvU_25_31', 'Gn4Iv5ARIXc_83_93','0lh_UWF9ZP4_215_226', 'z2kUc8wp9l8_40_46']
    for key in good_video_keys:
        captions = validation_captions[key]
        video_feature = all_features[key]
        play_video('data/' + key + '.avi')
        caption = generate_caption(model, tokenizer, video_feature, max_length)

        print("="*47)
        print('Generated caption: %s \n' % (caption))
        print("="*47)

        # Show ground-truth captions
        print("="*47)
        print("Ground truth captions:")
        for caption in captions:
            print(caption)
        print("="*47)

# Test model against one video
def test(model, all_features, tokenizer, max_length):
    # load one random (video_name, video_feature, caption)
    video_name, video_feature, captions = load_random_video_caption_pair(all_features)

    play_video('data/' + video_name + '.avi')

    # Generate description
    caption = generate_caption(model, tokenizer, video_feature, max_length)

    print("="*47)
    print('Generated caption: %s \n' % (caption))
    print("="*47)

    # Show ground-truth captions
    print("="*47)
    print("Ground truth captions:")
    for caption in captions:
        print(caption)
    print("="*47)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--op', default='train')

    # NOTE: We load features for all videos here, but the train/validation/test caption maps
    # will ensure to only select the allocated videos according the 70/20/10 split
    all_features_1 = load_features('video_features_first_16_frames.pkl')
    all_features_2 = load_features('video_features_stride_2.pkl')
    all_features_3 = load_features('video_features_stride_3.pkl')
    all_features_4 = load_features('video_features_stride_4.pkl')
    all_features_8 = load_features('video_features_stride_8.pkl')
    all_features_10 = load_features('video_features_stride_10.pkl')
    all_features_16 = load_features('video_features_stride_16.pkl')

    training_captions = load_captions('training_captions.json')
    test_captions = load_captions('test_captions.json')
    validation_captions = load_captions('validation_captions.json')
    tokenizer = preprocess_captions.create_tokenizer(training_captions)
    vocab_size = preprocess_captions.summarize_vocab(tokenizer)
    max_length = preprocess_captions.get_max_length(training_captions)

    # Load model check points (NOTE: pretrained_model_3 for Stride 3 was the best performing in BLEU)
    pretrained_model_1 = load_pretrained_model('model_checkpoints/model_99.h5')
    pretrained_model_2 = load_pretrained_model('model_checkpoints/model_stride_2_99.h5')
    pretrained_model_3 = load_pretrained_model('model_checkpoints/weights-improvement-stride_3-10-0.35.hdf5')
    pretrained_model_4 = load_pretrained_model('model_checkpoints/weights-improvement-stride_4-17-0.35.hdf5')
    pretrained_model_8 = load_pretrained_model('model_checkpoints/weights-improvement-stride_8-07-0.35.hdf5')
    pretrained_model_10 = load_pretrained_model('model_checkpoints/weights-improvement-stride_10-21-0.35.hdf5')
    pretrained_model_16 = load_pretrained_model('model_checkpoints/weights-improvement-stride_16-17-0.35.hdf5')

    args = parser.parse_args()

    if args.op == 'train':
        print('ALL SET FOR TRAINING ...')
        train(vocab_size, training_captions, validation_captions, all_features_1, tokenizer, max_length)
    elif args.op == 'evaluate':
        # NOTE: Use validation set for fine-tuning and evaluation. Only use test set for final inference! - Willie Brink
        print('ALL SET FOR EVALUATING ...')
        print('Stride 1')
        evaluate(pretrained_model_1, validation_captions, all_features_1, tokenizer, max_length)
        print("="*20)
        print('Stride 2')
        evaluate(pretrained_model_2, validation_captions, all_features_2, tokenizer, max_length)
        print("="*20)
        print('Stride 3')
        evaluate(pretrained_model_3, validation_captions, all_features_3, tokenizer, max_length)
        print("="*20)
        print('Stride 4')
        evaluate(pretrained_model_4, validation_captions, all_features_4, tokenizer, max_length)
        print("="*20)
        print('Stride 8')
        evaluate(pretrained_model_8, validation_captions, all_features_8, tokenizer, max_length)
        print("="*20)
        print('Stride 10')
        evaluate(pretrained_model_10, validation_captions, all_features_10, tokenizer, max_length)
        print("="*20)
        print('Stride 16')
        evaluate(pretrained_model_16, validation_captions, all_features_16, tokenizer, max_length)
    elif args.op == 'test':
        print('ALL SET FOR TESTING ...')
        test(pretrained_model_3, all_features_3, tokenizer, max_length)
    elif args.op == 'demo':
        print('ALL SET FOR QUICK DEMO ...')
        demo(pretrained_model_3, all_features_3, tokenizer, max_length)
    else:
        raise Exception('Choose valid operation: \'train\', \'evaluate\' or \'test\'')
