import numpy as np

def idhand(x):
	"""
	
	
	Parameters:
	-----------
	x : array
		length 10 array of individual card information for 5-hand.
		
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
		
	Returns:
	--------
	hands : int
		an array of 0 and 1 indicating all possible hands held
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
		
		
	>>> print idhand([1,4,2,4,3,4,4,4,3,7])
	[ 1.  1.  1.  1.  0.  0.  0.  1.  0.  0.]
	
	>>> print idhand([1,10,1,12,1,13,1,1,1,11])
	[ 1.  0.  0.  0.  0.  1.  0.  0.  1.  1.]
	
	>>> print idhand([1,7,1,2,1,4,1,10,1,12])
	[ 1.  0.  0.  0.  0.  1.  0.  0.  0.  0.]
	"""
	
	cardcount = np.zeros(13)
	suitcount = np.zeros(4)
	for (j,r) in enumerate(x):
		if j%2:
			cardcount[r-1] += 1
		else:
			suitcount[r-1] += 1
	
	handrank = np.zeros(10)
	handrank[0] = 1 #initially the hand is set at nothing
	
	if 4 in cardcount:
		#four of a kind, also contains a pair, two pair, and three of a kind
		handrank[1:4] = 1
		handrank[7] = 1
	
	elif 3 in cardcount:
		#contains three of a kind and a pair
		handrank[3] = 1
		handrank[1] = 1
		if 2 in cardcount:
			#contains two pair and fullhouse
			handrank[2] = 1	
			handrank[6] = 1
			
	elif 2 in cardcount:
		#contains a pair
		handrank[1] = 1
		if np.sum(cardcount == 2) == 2:
			#2 pairs but not a fullhouse
			handrank[2] = 1

	
	cards = range(1,14)*cardcount
	cards = cards[cards != 0]
	cardgaps = cards[1:] - cards[:-1]
	if 5 in suitcount:
		#has flush of some kind
		handrank[5] = 1
		
		#now check for consecutiveness
		#cards are all distinct in this case
		if np.sum(cardgaps==1) == 4:
			#straight flush but not royal flush
			handrank[8] = 1
		elif cardgaps[0] == 9:
			handrank[9] = 1
			handrank[8] = 1
	
	elif np.sum(cardgaps==1) == 4:
		#straight
		handrank[4] = 1
		
	elif np.sum(cardgaps==1) == 3 and cardgaps[0] == 9:
		#straight ace high
		handrank[4] = 1
	
	return handrank

def besthand(hands):
	"""
	returns index of largest occurence of a 1 in array hands
	
	Parameters:
	-----------
	hands : ndarray
	
	Returns:
	--------
	j : int
		largest index of hands where a 1 occurs (between 0 and 9)
		
	>>> besthand([1,0,0,0,0,0,0,0,0,0])
	0
	>>> besthand([1,1,1,1,1,0,0,0,0,0])
	4
	"""
	
	return max([j for j,x in enumerate(hands) if x == 1])
	
	
if __name__ == "__main__":
	import doctest
	doctest.testmod()
