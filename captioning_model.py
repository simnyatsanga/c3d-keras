from keras.models import model_from_json
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add

import c3d_model

# Define the captioning model
def get_model(vocab_size, max_length):
	# feature extractor model
	inputs_from_c3d = Input(shape=(4096,))
	fe1 = Dropout(0.5)(inputs_from_c3d)
	fe2 = Dense(256, activation='relu')(fe1)
	# sequence model
	inputs_from_caption = Input(shape=(max_length,))
	se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs_from_caption)
	se2 = Dropout(0.5)(se1)
	se3 = LSTM(256)(se2)
	# decoder model
	decoder1 = add([fe2, se3])
	decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(decoder2)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs_from_c3d, inputs_from_caption], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# summarize model
	print(model.summary())
	plot_model(model, to_file='model.png', show_shapes=True)
	return model


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--op', default='evaluate')



	if args.op == 'train':
		print('ALL SET FOR TRAINING ...')
	elif args.op == 'evaluate':
		print('ALL SET FOR EVALUATING ...')
	elif args.op == 'test':
		print('ALL SET FOR TESTING ...')
	else:
		raise Exception('Choose valid operation: \'train\', \'evaluate\' or \'test\'')
