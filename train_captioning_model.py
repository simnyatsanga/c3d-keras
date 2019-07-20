import os
import json
import argparse
import random
import time
import glob
import cv2
import numpy as np
from numpy import array
from numpy import argmax
from pickle import load, dump
from matplotlib import pyplot
import preprocess_captions
import captioning_model
from nltk.translate.bleu_score import corpus_bleu
from keras.utils import to_categorical
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.layers import Embedding

# Can be 50, 100, 200 or 300 dimensional according to the data
GLOVE_EMBEDDING_DIM = 100
CUSTOM_EMBEDDING_DIM = 256

def load_captions(input_file):
    with open(input_file, 'rb') as jsonfile:
        captions = json.loads(jsonfile.read())
    return captions

def load_video_features(filename):
    return load(open(filename, 'rb'))

def load_video_features_per_stride():
    # NOTE: We load features for all videos here, but the train/validation/test caption maps
    # will ensure to only select the allocated videos according the 70/20/10 split
    video_features_per_stride = []
    video_features_per_stride.append(load_video_features('video_features/video_features_first_16_frames.pkl'))
    video_features_per_stride.append(load_video_features('video_features/video_features_stride_2.pkl'))
    video_features_per_stride.append(load_video_features('video_features/video_features_stride_3.pkl'))
    video_features_per_stride.append(load_video_features('video_features/video_features_stride_4.pkl'))
    video_features_per_stride.append(load_video_features('video_features/video_features_stride_8.pkl'))
    video_features_per_stride.append(load_video_features('video_features/video_features_stride_10.pkl'))
    video_features_per_stride.append(load_video_features('video_features/video_features_stride_16.pkl'))
    return video_features_per_stride

def load_model_checkpoints_per_stride(checkpoint_dir):
    model_checkpoints_per_stride = []
    # Load model check points
    # NOTE: weights-improvement-stride_3 for Stride 3 was the best performing in BLEU using jointly
    # trained embedding layer
    model_checkpoints_per_stride.append(load_pretrained_model('model_checkpoints/%s/model_99.h5' % (checkpoint_dir)))
    model_checkpoints_per_stride.append(load_pretrained_model('model_checkpoints/%s/model_stride_2_99.h5' % (checkpoint_dir)))
    model_checkpoints_per_stride.append(load_pretrained_model('model_checkpoints/%s/weights-improvement-stride_3-10-0.35.hdf5' % (checkpoint_dir)))
    model_checkpoints_per_stride.append(load_pretrained_model('model_checkpoints/%s/weights-improvement-stride_4-17-0.35.hdf5' % (checkpoint_dir)))
    model_checkpoints_per_stride.append(load_pretrained_model('model_checkpoints/%s/weights-improvement-stride_8-07-0.35.hdf5' % (checkpoint_dir)))
    model_checkpoints_per_stride.append(load_pretrained_model('model_checkpoints/%s/weights-improvement-stride_10-21-0.35.hdf5' % (checkpoint_dir)))
    model_checkpoints_per_stride.append(load_pretrained_model('model_checkpoints/%s/weights-improvement-stride_16-17-0.35.hdf5' % (checkpoint_dir)))
    return model_checkpoints_per_stride

def load_pretrained_model(filename):
    return load_model(filename)

def load_random_video_caption_pair(features):
    # load validation caption keys
    validation_captions = load_captions('captions/validation_captions.json')
    validation_caption_keys = list(validation_captions.keys())
    random.shuffle(validation_caption_keys)
    # choose a random key using a randomly generated index
    random_key = validation_caption_keys[random.randint(0,len(validation_caption_keys))]
    captions = validation_captions[random_key]
    video = features[random_key]
    return random_key, video, captions

# Returns the full map of GloVE embeddings for 400k
def get_embeddings_index():
    embeddings_index = {}
    f = open(os.path.join('glove.6B', 'glove.6B.200d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

# Returns the embedding matrix
# For words in captions that exist in the embedding index, the embedding is picked
# For words in captions that don't exist in the embedding index, the embedding is np.zeros
def get_embedding_matrix(tokenizer):
    embeddings_index = get_embeddings_index()
    word_index = tokenizer.word_index
    not_found = []
    # NB: len(word_index) + 1 is equivalent to the vocab_size
    embedding_matrix = np.zeros((len(word_index) + 1, CUSTOM_EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Pad with zeros to match the 256 dimension of the embedding layer
            embedding_vector = np.concatenate([embedding_vector, np.zeros(56)])
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            not_found.append(word)
    print("No embeddings found for %s words" % (len(not_found)))
    return embedding_matrix

def get_embedding_layer(tokenizer, vocab_size):
    embedding_matrix = get_embedding_matrix(tokenizer)
    embedding_layer = Embedding(vocab_size,
                                CUSTOM_EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                trainable=False)
    return embedding_layer

def set_embedding_weights(model, embedding):
    model.layers[2].trainable=False
    # The glove embedding matrix needs to be passed in as a list because
    # model.layers[i].get_weights() returns the weights for layer i as a list
    model.layers[2].set_weights([embedding])

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
def data_generator(captions, video_features, tokenizer, max_length, vocab_size):
    # loop for ever over videos
    while 1:
        for key, captions_list in captions.items():
            # retrieve the video feature
            video_feature = video_features[key]
            in_video, in_seq, out_word = create_sequences(tokenizer, max_length, captions_list, video_feature, vocab_size)
            yield [[in_video, in_seq], out_word]

def train(video_features, training_captions, validation_captions, **kwargs):
    # define the model
    model = captioning_model.get_model(vocab_size, max_length)

    if kwargs['use_pretrained_emb'] == 'true':
        print('Model Training: Initialising model with pretrained embeddings and proceeding to finetune for captioning task')
        glove_embedding = get_embedding_matrix(kwargs['tokenizer'])
        set_embedding_weights(model, glove_embedding)
    elif kwargs['train_embedding'] == 'true':
        print('Model Training: Randomly initialising and training embedding layer')

    # load checkpoint
    # model = load_pretrained_model('weights-improvement-10-0.35.hdf5')

    # train the model, run epochs manually and save after each epoch
    epochs = 100
    training_steps = len(training_captions)
    validation_steps = len(validation_captions)

    # Simple implementation of training loop that saves all models with no validation accuracy check
    # training_generator = data_generator(training_captions, all_features, tokenizer, max_length, vocab_size)
    #
    # for i in range(epochs):
    #     print('Running epoch %d' %i)
    #     # create the data generator
    #     model.fit_generator(training_generator, epochs=1, steps_per_epoch=training_steps, verbose=1)
    #     # save model
    #     model.save('model_checkpoints/model_stride_1_glove_' + str(i) + '.h5')

    # Using training and validation generator so that the model is only saved if the validation accuracy improves
    training_generator = data_generator(training_captions,
                                        video_features,
                                        kwargs['tokenizer'],
                                        kwargs['max_length'],
                                        kwargs['vocab_size'])

    validation_generator = data_generator(validation_captions,
                                          video_features,
                                          kwargs['tokenizer'],
                                          kwargs['max_length'],
                                          kwargs['vocab_size'])
    if kwargs['use_pretrained_emb'] == 'true':
        checkpoint_filepath = "fine_tuned_embedding_layer/weights-improvement-stride_%s-glove-{epoch:02d}-{val_acc:.2f}.hdf5" % (kwargs['stride'])
    elif kwargs['train_embedding'] == 'true':
        checkpoint_filepath = "trained_embedding_layer/exp2-weights-improvement-stride_%s-{epoch:02d}-{val_acc:.2f}.hdf5" % (kwargs['stride'])

    checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_acc', verbose=1, mode='max', patience=50)
    callbacks_list = [checkpoint, early_stopping]

    history = model.fit_generator(training_generator,
                        epochs=100,
                        steps_per_epoch=training_steps,
                        validation_data=validation_generator,
                        validation_steps=validation_steps,
                        callbacks=callbacks_list,
                        verbose=1)
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    plot_file_name = 'training_val_loss_plot_stride_%s' % (stride)
    pyplot.savefig(plot_file_name, bbox_inches='tight')


# Evaluate the skill of the model
def evaluate(model, captions, videos, tokenizer, max_length, vocab_size):
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

    # with open('results.txt', 'w+') as result_file:
    #     result_file.write(result_string)

# Play videos and generate captions using pre-selected videos
def demo(model, all_features, tokenizer, max_length):
    validation_captions = load_captions('captions/validation_captions.json')
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

def prepare_model_checkpoints(args, **kwargs):
    if args.use_pretrained_embedding == 'true' and args.use_finetuned_embedding == 'true':
        raise ValueError('Can only use one of these options [use_pretrained_embedding, use_finetuned_embedding]')
    if args.use_pretrained_embedding == 'true':
        print('INFO: Initialising checkpoints with pretrained embeddings')
        model_checkpoints_per_stride = load_model_checkpoints_per_stride('trained_embedding_layer')
        glove_embedding = get_embedding_matrix(kwargs['tokenizer'])
        for checkpoint in model_checkpoints_per_stride:
            set_embedding_weights(checkpoint, glove_embedding)
    elif args.use_finetuned_embedding == 'true':
        print('INFO: Using checkpoints with finetuned embeddings')
        model_checkpoints_per_stride = load_model_checkpoints_per_stride('finetuned_embedding_layer')
    elif args.use_finetuned_embedding == 'false' and args.use_pretrained_embedding == 'false':
        print('INFO: Using checkpoints with trained embedding layer')
        model_checkpoints_per_stride = load_model_checkpoints_per_stride('trained_embedding_layer')
    return model_checkpoints_per_stride

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--op', default='train')
    parser.add_argument('--train_embedding', default='true')
    parser.add_argument('--use_pretrained_embedding', default='false')
    parser.add_argument('--use_finetuned_embedding', default='false')

    video_features_per_stride = load_video_features_per_stride()
    training_captions = load_captions('captions/training_captions.json')
    test_captions = load_captions('captions/test_captions.json')
    validation_captions = load_captions('captions/validation_captions.json')
    tokenizer = preprocess_captions.create_tokenizer(training_captions)
    vocab_size = preprocess_captions.summarize_vocab(tokenizer)
    max_length = preprocess_captions.get_max_length(training_captions)

    args = parser.parse_args()

    if args.op == 'train':
        print('ALL SET FOR TRAINING ...')
        i = 0
        strides = [1, 2, 3, 4, 8, 10, 16]
        for video_features in video_features_per_stride:
            print("Training for Stride %s " % (strides[i]))
            # train(vocab_size, training_captions, validation_captions, video_features, tokenizer, max_length)
            train(video_features,
                  training_captions,
                  validation_captions,
                  vocab_size=vocab_size,
                  tokenizer=tokenizer,
                  max_length=max_length,
                  stride=strides[i],
                  train_embedding=args.train_embedding,
                  use_pretrained_emb=args.use_pretrained_embedding)
            print("="*20)
            i += 1
    elif args.op == 'evaluate':
        # NOTE: Use validation set for fine-tuning and evaluation. Only use test set for final inference! - Willie Brink
        model_checkpoints_per_stride = prepare_model_checkpoints(args, tokenizer=tokenizer)

        print('Model Evaluation: ALL SET FOR EVALUATING ...')
        i = 0
        strides = [1, 2, 3, 4, 8, 10, 16]
        for checkpoint in model_checkpoints_per_stride:
            print('Stride %s' % (strides[i]))
            evaluate(checkpoint,
                     validation_captions,
                     video_features_per_stride[i],
                     tokenizer, max_length,
                     vocab_size)
            print("="*20)
            i += 1
    elif args.op == 'test':
        model_checkpoints_per_stride = prepare_model_checkpoints(args, tokenizer=tokenizer)
        print('ALL SET FOR TESTING ...')
        test(model_checkpoints_per_stride[2], video_features_per_stride[2], tokenizer, max_length)
    elif args.op == 'demo':
        model_checkpoints_per_stride = prepare_model_checkpoints(args, tokenizer=tokenizer)
        print('ALL SET FOR QUICK DEMO ...')
        demo(model_checkpoints_per_stride[2], video_features_per_stride[2], tokenizer, max_length)
    else:
        raise Exception('Choose valid operation: \'train\', \'evaluate\', \'test\' or \'demo\'')
