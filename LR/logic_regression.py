#--coding: utf8 ---
#===============================================================================
# File  : logic_regression.py
# Desc  : the code of simple logic regression.
#  
# Date  : 2016.09.01
# Author: jiuqian(81691)
# Email : jianhui.jjh@alibaba-inc.com
#===============================================================================

import matplotlib.pyplot as plt
from numpy import *
import matplotlib.animation as animation
import time

def sigmoid(X):
	return 1.0 / (1 + exp(-X) )

# train a logistic regression model using some optional optimize algorithm  
# input: train_x is a mat datatype, each row stands for one sample  
#        train_y is mat datatype too, each row is the corresponding label  
#        opts is optimize option include step and maximum number of iterations  

def trainLogicRegression(train_x, train_y, opts):
	startTime = time.time()

	alpha = opts['alpha']
	maxIter = opts['maxIter']

	# load the data 
	numSamples, numFeatures = shape(train_x)

	# default weight for each feature
	weights = ones( (numFeatures, 1) ) 

	# begin to train
	for k in range(maxIter):
		if opts['optimizeType'] == 'gradientDescent':
			# calculate the error and update the parameters
			output = sigmoid( train_x * weights )
			error = train_y - output
			weights = weights + alpha * train_x.transpose() * error

	print "[DEG] Train the Logic Regression end with time:%fs"%( time.time() -startTime)

	return weights


def testLogicRegression(weights, test_x, test_y):
	numSamples, numFeatures = shape(test_x)
	matchCount = 0 

	for i in xrange( numSamples ):
		predict = sigmoid( test_x[i, :] * weights)[0, 0] > 0.5
		if predict == bool(test_y[i, 0]):
			matchCount += 1
	accury = float(matchCount) / numSamples


def showLogicRegression(weights, train_x, train_y):
	numSamples, numFeatures = shape(train_x)

	# draw thw points
	for i in xrange(numSamples):
		if int(train_y[i,0]) == 0:
			plt.plot(train_x[i,1], train_x[i, 2], 'or')
		elif int(train_y[i,0]) == 1:
			plt.plot(train_x[i,1], train_x[i, 2], 'ob')
	
	# draw the classify line 
	min_x = min(train_x[:,1])[0,0]
	max_x = max(train_x[:,1])[0,0]

	weights = weights.getA()  # convert mat to array
	
	y_min_x = float(-weights[0] - weights[1] * min_x) / weights[2]  
	y_max_x = float(-weights[0] - weights[1] * max_x) / weights[2]  

	plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
	plt.xlabel('x1')
	plt.ylabel('x2')

	plt.show()



# test the Logic Regression 
def loadData():
	train_x = []
	train_y = []

	fileIn = open("./logic_regression_testSet.txt")
	for line in fileIn.readlines():
		lineArr = line.strip().split()
		train_x.append( [1.0, float(lineArr[0]), float(lineArr[1])] )
		train_y.append( float( lineArr[2]) )
	
	return mat(train_x), mat(train_y).transpose


print "step 1: load data ..."
train_x, train_y = loadData()
test_x = train_x
test_y = train_y

print "step 2: training ..."
opts = {'alpha':0.01, 'maxIter':20, 'optimizeType':'gradientDescent'}
optimalweights = trainLogicRegression(train_x, train_y, opts)

print "step 3: testing ..."
accury = testLogicRegression(optimalweights, test_x, test_y)

print "step 4: show the result ..."
print 'The classify accuracy is: %.3f%%' % (accuracy * 100)  
showLogicRegression(optimalweights, train_x, train_y)









