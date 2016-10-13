import numpy as np
import csv
import progressbar

"""
Taken from http://archive.ics.uci.edu/ml/datasets/Poker+Hand
Format is:

1) S1 "Suit of card #1" 
Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs} 

2) C1 "Rank of card #1" 
Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King) 

3) S2 "Suit of card #2" 
Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs} 

4) C2 "Rank of card #2" 
Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King) 

5) S3 "Suit of card #3" 
Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs} 

6) C3 "Rank of card #3" 
Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King) 

7) S4 "Suit of card #4" 
Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs} 

8) C4 "Rank of card #4" 
Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King) 

9) S5 "Suit of card #5" 
Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs} 

10) C5 "Rank of card 5" 
Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King) 

11) CLASS "Poker Hand" 
Ordinal (0-9) 

0: Nothing in hand; not a recognized poker hand 
1: One pair; one pair of equal ranks within five cards 
2: Two pairs; two pairs of equal ranks within five cards 
3: Three of a kind; three equal ranks within five cards 
4: Straight; five cards, sequentially ranked with no gaps 
5: Flush; five cards with the same suit 
6: Full house; pair + different rank three of a kind 
7: Four of a kind; four equal ranks within five cards 
8: Straight flush; straight + flush 
9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush 
"""

def readdata(filename):

	with open(filename,'r') as f:
		data_iter = csv.reader(f, delimiter = ',', quotechar = '"')
		data = [row for row in data_iter]
		data_array = np.asarray(data, dtype = np.int)   

	return data_array

def augment_data(Z):
	"""
	generate more features as follows:
		- max number of cards that have the same suit
		- max number of cards that are in consecutive numerical order (assume that ace can be 1 or 14 effectively)
		- high card
		- low card
		- 
	"""
	new_data = np.zeros([Z.shape[0], Z.shape[1] + 4])
	new_data[:, :Z.shape[1]] = Z
	for j in xrange(Z.shape[0]):
		pass

	return new_data
	
def bruteforce_classifier_test():
	testfilename = 'poker-hand-testing.data'
	from pokerhandid import idhand, besthand
	Z = readdata(testfilename)
	X = Z[:,:-1]
	y = Z[:,-1]
	N = X.shape[0]
	print N
	
	accuracy = 0
	for j in xrange(N):
		accuracy += (besthand(idhand(X[j,:])) == y[j])
		progressbar.printProgress(j+1, N, prefix = "pokerclassify.bruteforce_classifier_test: iteration {}".format(j+1), barLength = 40)
	
	accuracy = accuracy*1./float(N)
	return accuracy


def compute_confusion_matrix(Y_test, Y_predicted, nb_classes, normalize = False):
	"""
	Given true class values Y_test (shape = (N, nb_classes)) with values lying in range(nb_classes)
	computes the confusion matrix (shape = (nb_classes, nb_classes)) associated with the 
	predicted class values Y_predicted (shape = (N, nb_classes))
	"""
	#convert data sets to flattened vectors
	y_test = np.argmax(Y_test, axis = 1)
	y_predicted = np.argmax(Y_predicted, axis = 1)

	confusion_matrix = np.zeros([nb_classes, nb_classes])
	
	for j in range(nb_classes):
		#slice y_test for indices where the true class is equal to j
		indices = [i for (i,v) in enumerate(y_test == j) if v == True]
		
		for k in range(nb_classes):
			confusion_matrix[j,k] = np.sum(y_predicted[indices] == k)

		if normalize:
			#normalize jth row
			confusion_matrix[j, :] /= float(len(indices))
	
	return confusion_matrix

def keras_nn_train():
	from keras.models import Sequential, Model
	from keras.callbacks import ModelCheckpoint
	from keras.layers import Embedding, Permute, Input, Dense, Dropout, Activation, Flatten, Reshape, merge
	from keras.utils import np_utils
	from keras import backend as K

	# parameters
	batch_size = 128 #number of examples per gradient descent step
	nb_classes = 10 #number of possible hand labels
	nb_epoch = 20 #number of learning epochs

	# input data dimensions
	ns_cols = 5
	nr_cols = 5
	n_cols = ns_cols + nr_cols

	s_indices = [0,2,4,6,8]
	r_indices = [1,3,5,7,9]

	#data filenames
	trainingfilename = 'poker-hand-training.data'
	testfilename = 'poker-hand-testing.data'

	#read data
	Z_train = readdata(trainingfilename)
	Z_test = readdata(testfilename)

	k1 = 8
	k2 = 9 #royal flush and straight flush data
	zero_batch_size = 10

	###
	#filter out only a select number of classes
	Z_train = Z_train[(Z_train[:,-1] == k2) + (Z_train[:,-1] == k1) + (Z_train[:,-1] == 0), :]
	Z_test = Z_test[(Z_test[:,-1] == k2) + (Z_test[:,-1] == k1) + (Z_test[:,-1] == 0), :]

	#Z_train, Z_test are fixed from now on
	class_indices_1 = [i for (i,v) in enumerate(Z_train[:,-1] == k1) if v == True]
	class_indices_2 = [i for (i,v) in enumerate(Z_train[:,-1] == k2) if v == True]
	no_hand_indices = [i for (i,v) in enumerate(Z_train[:,-1] == 0) if v == True]
	Z_train[class_indices_1, -1] = 1
	Z_train[class_indices_2, -1] = 2
	Z_test[Z_test[:,-1]==k1, -1] = 1
	Z_test[Z_test[:,-1]==k2, -1] = 2

	###
	nb_classes = 3

	X_train_suits = Z_train[:, s_indices] - 1
	X_train_ranks = Z_train[:, r_indices] - 1
	y_train = Z_train[:,-1]

	X_test_suits = Z_test[:, s_indices] - 1
	X_test_ranks = Z_test[:, r_indices] - 1
	y_test = Z_test[:,-1]

	X_train_suits = X_train_suits.reshape(X_train_suits.shape[0], ns_cols, 1)
	X_train_ranks = X_train_ranks.reshape(X_train_ranks.shape[0], nr_cols, 1)
	X_test_suits = X_test_suits.reshape(X_test_suits.shape[0], ns_cols, 1)
	X_test_ranks = X_test_ranks.reshape(X_test_ranks.shape[0], nr_cols, 1)
	
	# convert class vectors to binary class matrices
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)
	
	suits = Input(shape=(5,1), dtype='int32', name = 'suits') #integer suit values to be embedded (4 values)
	ranks = Input(shape=(5,1), dtype='int32', name = 'ranks') #integer card ranks to be embedded (13 values)
	
	output_dim = 200
	em1 = Embedding(output_dim=output_dim, input_dim=4, input_length=5)(suits) #4 classes for suits feature
	em2 = Embedding(output_dim=output_dim, input_dim=13, input_length=5)(ranks) #13 classes for ranks feature
	T = merge([em1, em2], mode='concat', concat_axis = 1)
	T = Reshape((output_dim*10,))(T)
	T = Dense(128, activation = 'relu')(T)
	T = Dense(128, activation = 'relu')(T)
	T = Dense(128, activation = 'relu')(T)
	main_output = Dense(nb_classes, activation = 'softmax')(T)
	model = Model(input=[suits, ranks], output=[main_output])
	model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

	# Callback for model saving:
	checkpointer = ModelCheckpoint(filepath="auto_save_weights.hdf5", verbose=0, save_best_only = True)
	q = 0.4 #parameter for choosing permutation of cards in hand

	for j in range(nb_epoch):
#		X_train_ranks_temp = np.copy(X_train_ranks)
#		X_train_suits_temp = np.copy(X_train_suits)
		i1 = np.random.choice(no_hand_indices, size = zero_batch_size, replace = False)
		Z_train_temp = np.vstack((Z_train[class_indices_1, :], Z_train[class_indices_2, :], Z_train[i1, :]))
		y_train_temp = np.concatenate((Z_train[class_indices_1, -1], Z_train[class_indices_2, -1], Z_train[i1, -1]))
		X_train_ranks_temp = Z_train_temp[:, r_indices] - 1
		X_train_suits_temp = Z_train_temp[:, s_indices] - 1
		
		#reshape stuff
		X_train_suits_temp = X_train_suits_temp.reshape(X_train_suits_temp.shape[0], ns_cols, 1)
		X_train_ranks_temp = X_train_ranks_temp.reshape(X_train_ranks_temp.shape[0], nr_cols, 1)
		Y_train_temp = np_utils.to_categorical(y_train_temp, nb_classes)

		print "Starting epoch {}...".format(j+1)

		# Add noise on later epochs
		if j > 0:
			for k in range(0, X_train_ranks_temp.shape[0]):
				if np.random.rand(1) > q:				
					p = np.random.choice(5, size = 5, replace = False)
					X_train_ranks_temp[k,:,0] = X_train_ranks_temp[k,p,0] #permute
					X_train_suits_temp[k,:,0] = X_train_suits_temp[k,p,0]

		model.fit([X_train_suits_temp, X_train_ranks_temp], Y_train_temp, batch_size=batch_size, nb_epoch=1,
				verbose=1, validation_data=([X_test_suits, X_test_ranks], Y_test), callbacks = [checkpointer])

	score = model.evaluate([X_test_suits, X_test_ranks], Y_test, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])

	Y_predicted = model.predict([X_test_suits, X_test_ranks])

	confusion_matrix = compute_confusion_matrix(Y_test, Y_predicted, nb_classes, normalize = False)
	print confusion_matrix

if __name__ == "__main__":
	keras_nn_train()
	#print bruteforce_classifier_test()
