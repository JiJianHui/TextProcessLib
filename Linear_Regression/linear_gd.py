#===============================================================================
# File  : linear_gd.py
# Desc  : 
#  
# Date  : 2016.06.21
# Author: jiuqian(81691)
# Email : jianhui.jjh@alibaba-inc.com
#===============================================================================
#!/bin/bash

import numpy as np
import os
import matplotlib.pyplot as plt

# print the san dian tu
def drawScatterDiagram(points, b, m):

	xrecord = []
	yrecord = []
	for point in points:
		xrecord.append( float(point[0]) )
		yrecord.append( float(point[1]) )
	plt.scatter(xrecord, yrecord, c='blue', marker='p')
	plt.axis([20,80,20,140])

	# the best linear regression
	x = np.arange(25.0,75.0,0.1)
	y = m*x + b
	plt.plot(x, y, c='red')

	plt.show()

# calculate the all the cost y = b + mx
def compute_error_for_line_given_points(b, m, points):
	totalError = sum( ( ( (b+m*point[0] - point[1])**2 ) for point in points ))
	return totalError / float( len(points) )

# one step of the gd algothim, betch gd
def step_gradient(b_current, m_current, points, learningRate):
	b_gradient = 0
	m_gradient = 0

	N = float(len(points))

	for i in range(0, len(points)):
		x = points[i,0]
		y = points[i,1]
		b_gradient += (2/N) * (b_current + m_current * x - y ) * 1 
		m_gradient += (2/N) * (b_current + m_current * x - y ) * x

	new_b = b_current - ( learningRate * b_gradient )
	new_m = m_current - ( learningRate * m_gradient )
	
	return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learningRate, num_iterations):
	b = starting_b
	m = starting_m

	for i in range(num_iterations):
		b, m = step_gradient(b, m, points, learningRate)
	return [b, m]

def run():
	points = np.genfromtxt("./data.csv", delimiter=",")
	learningRate = 0.0001
	initial_b = 0
	initial_m = 0
	num_iterations = 100

	print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points))
	print "Runing"

	[b,m] = gradient_descent_runner(points, initial_b, initial_m, learningRate, num_iterations)

	print "After gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points))

	drawScatterDiagram(points, b, m)

if __name__ == "__main__":
	run()










