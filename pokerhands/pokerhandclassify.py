import numpy as np
from sklearn import svm
from sklearn.lda import LDA
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
	

def keras_nn_train():
	from keras.models import Sequential
	from keras.layers import Dense, Dropout, Activation, Flatten
	from keras.layers import Convolution2D, MaxPooling2D
	from keras.utils import np_utils
	from keras import backend as K

	trainingfilename = 'poker-hand-training_true.data'
	testfilename = 'poker-hand-testing.data'
	Z_train = readdata(trainingfilename)
	X_train = Z_train[:,:-1]
	y_train = Z_train[:,-1]
	N = X_train.shape[0]
	print N

	Z_test = readdata(testfilename)
	X_test = Z_test[:,:-1]
	y_test = Z_test[:,-1]

	model = Sequential()
#####


	batch_size = 128
	n_classes = 10
	nb_epoch = 12

	# input image dimensions
	ns_cols = 5
	nr_cols = 5
	n_cols = ns_cols + nr_cols

	s_indices = [0,2,4,6,8]
	r_indices = [1,3,5,7,9]

	# number of convolutional filters to use
	nb_filters = 32
	# size of pooling area for max pooling
	pool_size = (2, 2)
	# convolution kernel size
	kernel_size = (3, 3)

X_train = X_train.reshape(X_train.shape[0], n_cols, 1)
X_test = X_test.reshape(X_test.shape[0], n_cols, 1)
input_shape = (n_cols, 1)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
			verbose=1, validation_data=(X_test, Y_test))
	

def main():

	trainingfilename = 'poker-hand-training-true.data'
	testfilename = 'poker-hand-testing.data'
	
	
if __name__ == "__main__":
	main()
	#print bruteforce_classifier_test()
