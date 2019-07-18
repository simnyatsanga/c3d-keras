import string
import copy
import glob
import json
import random
import pandas as pd
from keras.preprocessing.text import Tokenizer

# NOTE: There's a discrepancy between captions in the video_corpus.csv (found here: https://github.com/chenxinpeng/S2VT/blob/master/data/video_corpus.csv)
# This is because some of the entries in the csv are for videos that aren't on Youtube anymore. So after filtering the available videos, the total = 1970 videos
NUM_TRAIN = 1379 # 70% of total videos
NUM_VAL = 394 # 30% of total videos
NUM_TEST = 194 # 10% of total videos

# select only captioned videos that are available
# this is because some of the captioned videos are no longer available on youtube
def get_captions_for_available_videos():
    captions_df = pd.read_csv('data/video_corpus.csv')
    captions_df = captions_df[captions_df['Language'] == 'English']
    captions_df = captions_df[['VideoID', 'Start', 'End', 'Description']].dropna()
    captions_tuples = list(zip(captions_df['VideoID'].values.tolist(), captions_df['Start'].values.tolist(), captions_df['End'].values.tolist(), captions_df['Description'].values.tolist()))
    avail_videos = [vid.split("/")[-1].split(".")[0] for vid in glob.glob('data/videos/*')]
    avail_videos_with_captions = list()
    for vid_cap_tup in captions_tuples:
        video_id =  "%s_%s_%s" % (vid_cap_tup[0], vid_cap_tup[1], vid_cap_tup[2])
        if video_id in avail_videos:
            # append the concat'd video_id and caption eg. ('Bdfrwewjk_1_10', 'a man walking on the beach')
            avail_videos_with_captions.append((video_id, vid_cap_tup[3]))
    return avail_videos_with_captions

# maps every video to all its possible captions
def map_captions(captions_tuples):
    mapping = dict()
    for video_id, caption in captions_tuples:
        if video_id not in mapping:
        	mapping[video_id] = list()
    	# store caption
        mapping[video_id].append(caption)
    return mapping

def split_captions(captions):
    all_keys = list(captions.keys())
    random.shuffle(all_keys)
    training_caption_keys = all_keys[:1379]
    validation_caption_keys = all_keys[1379:1773]
    test_caption_keys = all_keys[1773:]

    training_captions = {key: captions[key] for key in training_caption_keys}
    validation_captions = {key: captions[key] for key in validation_caption_keys}
    test_captions = {key: captions[key] for key in test_caption_keys}

    return training_captions, validation_captions, test_captions

def save_captions(captions, output_file):
    with open(output_file, 'w') as jsonfile:
        jsonfile.write(json.dumps(captions))

def clean_captions(mapped_captions):
    captions_map = copy.deepcopy(mapped_captions)
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for key, captions_list in captions_map.items():
        for i in range(len(captions_list)):
            caption = captions_list[i]
            # tokenize
            caption = caption.split()
            # convert to lower case
            caption = [word.lower() for word in caption]
            # remove punctuation from each token
            caption = [w.translate(table) for w in caption]
            # remove hanging 's' and 'a'
            caption = [word for word in caption if len(word)>1]
            # remove tokens with numbers in them
            caption = [word for word in caption if word.isalpha()]
            # join back into string
            caption = ' '.join(caption)
            # append start and end token
            caption = 'startseq ' + caption + ' endseq'
            # store as string
            captions_list[i] = caption
    return captions_map

# convert the captions into a vocabulary of all unique words
def build_vocabulary(captions):
    vocab = set()
    for key in captions.keys():
    	[vocab.update(caption.split()) for caption in captions[key]]
    return vocab

def summarize_vocab(tokenizer):
	vocab_size = len(tokenizer.word_index) + 1
	print('[Info] Vocabulary Size: %d' % vocab_size)
	return vocab_size

# convert a dictionary of clean captions to a list of captions
def get_full_captions_list(captions):
    all_captions = list()
    for key in captions.keys():
        [all_captions.append(caption) for caption in captions[key]]
    return all_captions

# fit a tokenizer given caption descriptions
# every string extracted from the list of captions is mappped to a unique integer
# the map is useful when mapping from integer to word in vocabulary during predictions
def create_tokenizer(captions):
    lines = get_full_captions_list(captions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# calculate the length of the caption with the most words
def get_max_length(captions):
    captions_list = get_full_captions_list(captions)
    return max(len(caption.split()) for caption in captions_list)

def main():
    captions = get_captions_for_available_videos()
    mapped_captions = map_captions(captions)
    cleaned_captions = clean_captions(mapped_captions)
    training_captions, validation_captions, test_captions = split_captions(cleaned_captions)
    save_captions(training_captions, 'captions/training_captions.json')
    save_captions(validation_captions, 'captions/validation_captions.json')
    save_captions(test_captions, 'captions/test_captions.json')


if __name__ == '__main__':
    main()
