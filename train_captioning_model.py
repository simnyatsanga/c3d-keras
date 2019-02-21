import json
import argparse
import random
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


def load_captions(input_file):
    with open(input_file, 'rb') as jsonfile:
        captions = json.loads(jsonfile.read())
    return captions

def load_features():
    return load(open('video_features.pkl', 'rb'))

def load_pretrained_model(filename):
    return load_model(filename)

def load_random_video_caption_pair(features):
    # load test caption keys
    test_captions = load_captions('test_captions.json')
    test_caption_keys = list(test_captions.keys())
    random.shuffle(test_caption_keys)
    # choose a random key using a randomly generated index
    random_key = test_caption_keys[random.randint(0,len(test_captions))]
    captions = test_captions[random_key]
    video = features[random_key]
    return random_key, video, captions

# map an integet to a word
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

# generate a description for a videp
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

# data generator, intended to be used in a call to model.fit_generator()
def data_generator(captions, videos, tokenizer, max_length, vocab_size):
    # loop for ever over videos
    while 1:
        for key, captions_list in captions.items():
            # retrieve the video feature
            video = videos[key]
            in_video, in_seq, out_word = create_sequences(tokenizer, max_length, captions_list, video, vocab_size)
            yield [[in_video, in_seq], out_word]

def train(vocab_size, training_captions, training_features, tokenizer, max_length):
    # define the model
    model = captioning_model.get_model(vocab_size, max_length)

    # train the model, run epochs manually and save after each epoch
    epochs = 100
    steps = len(training_captions)
    for i in range(epochs):
        print('Running epoch %d' %i)
        # create the data generator
        generator = data_generator(training_captions, training_features, tokenizer, max_length, vocab_size)
        # fit for one epoch
        model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
        # save model
        model.save('model_' + str(i) + '.h5')

# evaluate the skill of the model
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


# test model against one video
def test(model, all_features, tokenizer, max_length):
    # pre-define the max sequence length (from training)
    # max_length = 50
    # load one random video
    video_name, video, captions = load_random_video_caption_pair(all_features)
    # generate description
    caption = generate_caption(model, tokenizer, video, max_length)
    print("===============================================")
    print('Generated caption: %s ----> %s \n' % (video_name, caption))
    print("===============================================")

    print("Ground truth captions are: \n %s" % captions)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--op', default='train')

    # NOTE: We load features for all videos here, but the train/validation/test caption maps
    # will ensure to only select the allocated videos according the 70/20/10 split
    all_features = load_features()

    training_captions = load_captions('training_captions.json')
    test_captions = load_captions('test_captions.json')
    tokenizer = preprocess_captions.create_tokenizer(training_captions)
    vocab_size = preprocess_captions.summarize_vocab(tokenizer)
    max_length = preprocess_captions.get_max_length(training_captions)
    pretrained_model = load_pretrained_model('model_99.h5')

    args = parser.parse_args()

    if args.op == 'train':
        print('ALL SET FOR TRAINING ...')
        train(vocab_size, training_captions, all_features, tokenizer, max_length)
    elif args.op == 'evaluate':
        print('ALL SET FOR EVALUATING ...')
        evaluate(pretrained_model, test_captions, all_features, tokenizer, max_length)
    elif args.op == 'test':
        print('ALL SET FOR TESTING ...')
        test(pretrained_model, all_features, tokenizer, max_length)
    else:
        raise Exception('Choose valid operation: \'train\', \'evaluate\' or \'test\'')
