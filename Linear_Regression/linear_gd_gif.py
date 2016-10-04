#===============================================================================
# File  : linear_gd.py
# Desc  : 
# Date  : 2016.06.21
# Author: jiuqian(81691)
# Email : jianhui.jjh@alibaba-inc.com
#===============================================================================
#!/bin/bash

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

	err = 0
	for i in range(0, len(points)):
		x = points[i,0]
		y = points[i,1]
		b_gradient += (2/N) * (b_current + m_current * x - y ) * 1 
		m_gradient += (2/N) * (b_current + m_current * x - y ) * x

		# count the error
		err = (b_current + m_current * x - y ) * ( b_current + m_current * x - y  ) 

	new_b = b_current - ( learningRate * b_gradient )
	new_m = m_current - ( learningRate * m_gradient )

	global errs
	errs.append(err)	
	return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learningRate, num_iterations):
	b = starting_b
	m = starting_m

	for i in range(num_iterations):
		b, m = step_gradient(b, m, points, learningRate)
	return [b, m]

def init():
	global xrecord, points, linear_gd, line_best, line_err

	return line_best, line_gd, line_err

def animate(i):
	global xrecord, points, linear_gd, line_best, learningRate, b, m, line_err, errs
	b, m = step_gradient(b, m, points, learningRate)
	line_gd.set_data(xrecord, b + m * xrecord)

	ssrecord = [i for i in range(len(errs))]
	line_err.set_data(ssrecord, errs)

	return line_gd, line_err

fig = plt.figure()
axes1 = fig.add_subplot(211)
axes2 = fig.add_subplot(212,xlim=(0,100), ylim=(0,6000))

points = np.genfromtxt("./data.csv", delimiter=",")
axes1.scatter(points[:,0], points[:,1], c='blue', marker="o")

xrecord = points[:,0]
line_gd, = axes1.plot(xrecord, [0 for i in range(len(xrecord))], lw=2)
line_best, = axes1.plot(xrecord, 7.99102098227 + 1.32243102276 * xrecord, lw=2, c='red')
line_err, = axes2.plot([], [], lw=2, c='green')

b = 0
m = 0
errs = []
learningRate = 0.0001
num_iterations = 10

anim1=animation.FuncAnimation(fig, animate, init_func=init,  frames=num_iterations, interval=600)#, blit=True)


plt.show()



#def run():
#	points = np.genfromtxt("./data.csv", delimiter=",")
#	learningRate = 0.0001
#	initial_b = 0
#	initial_m = 0
#	num_iterations = 100
#
#	print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points))
#	print "Runing"
#
#	[b,m] = gradient_descent_runner(points, initial_b, initial_m, learningRate, num_iterations)
#
#	print "After gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points))
#
#	drawScatterDiagram(points, b, m)








