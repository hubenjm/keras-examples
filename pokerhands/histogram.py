import numpy as np
import csv
import matplotlib.pyplot as plt
import sys

"""
Plots a histogram of a given attribute for the supplied data.
Assumes that the attribute is an integer between 1 and the number of columns of the data, including the class variable (y).
This can be used for any data set which has nonnegative integer classes and which is arranged by column: x1, x2, x2, ..., xn, y
However, it was originally written to use for the poker hand data set from the UCI Machine Learning Repository

Usage: $ python histogram.py <filename> <attribute>
"""

def histogram(x, bins):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	counts, bins, patches = ax.hist(x, bins = bins, color='blue', normed=True, alpha=0.8)
	ax.set_xticks(bins)
	plt.show()
	
def readdata(filename):

	with open(filename,'r') as f:
		data_iter = csv.reader(f, delimiter = ',', quotechar = '"')
		data = [data for data in data_iter]
		data_array = np.asarray(data, dtype = int)   

	return data_array
	
def main():
	filename = sys.argv[1]
	attribute = int(sys.argv[2])
	
	X = readdata(filename)
        N = X.shape[1]
        assert attribute <= N
	x = X[:,attribute-1]
	
	if attribute == N:
                #histogram for classes
		bins = np.arange(N)
	else:
                #histogram for one of the independent attribute variables
		bins = np.linspace(1,np.max(x) + 1, np.max(x)+1)
		
	histogram(x, bins)
	
if __name__ == "__main__":
	main()
	
