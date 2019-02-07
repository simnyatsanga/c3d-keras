import string
import copy
import glob
import pandas as pd
from keras.preprocessing.text import Tokenizer


def load_captions():
    captions_df = pd.read_csv('data/video_corpus.csv')
    captions_df = captions_df[captions_df['Language'] == 'English']
    return captions_df[['VideoID', 'Description']].dropna()

# select only videos that have captions in the video corpus
def get_videos_with_captions():
    captions_df = pd.read_csv('data/video_corpus.csv')
    captions_df = captions_df[captions_df['Language'] == 'English']
    videos_with_captions = list(set(captions_df['VideoID'].values.tolist()))
    all_videos = [vid.split("/")[-1].split(".")[0][0:11] for vid in glob.glob('data/videos/*')]
    vids = set(all_videos).intersection(set(videos_with_captions))
    import ipdb; ipdb.set_trace()


# maps every video to all its possible captions
def map_captions(captions_df):
    mapping = dict()
    video_caption_pairs = zip(captions_df['VideoID'].values.tolist(), captions_df['Description'].values.tolist())
    for vid_cap_pair in list(video_caption_pairs):
        if vid_cap_pair[0] not in mapping:
        	mapping[vid_cap_pair[0]] = list()
    	# store caption
        mapping[vid_cap_pair[0]].append(vid_cap_pair[1])
    return mapping

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
            # store as string
            captions_list[i] = ' '.join(caption)
    return captions_map

# convert the captions into a vocabulary of all unique words
def build_vocabulary(captions):
    vocab = set()
    for key in captions.keys():
    	[vocab.update(caption.split()) for caption in captions[key]]
    return vocab

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

def create_sequences(tokenizer, max_length, caption_list, video):
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

# data generator, intended to be used in a call to model.fit_generator()
def data_generator(captions, videos, tokenizer, max_length):
	# loop for ever over videos
	while 1:
		for key, captions_list in captions.items():
			# retrieve the video feature
			video = videos[key][0]
			in_video, in_seq, out_word = create_sequences(tokenizer, max_length, captions_list, video)
			yield [[in_video, in_seq], out_word]

def main():
    captions = load_captions()
    mapped_captions = map_captions(captions)
    cleaned_captions = clean_captions(mapped_captions)
    build_vocabulary(cleaned_captions)
    create_tokenizer(cleaned_captions)
    get_max_length(cleaned_captions)
    get_videos_with_captions()


if __name__ == '__main__':
    main()
