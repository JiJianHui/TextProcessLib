# -*- coding: utf-8 -*
#===============================================================================
# File  : linear_regression.py
# Desc  : 
#  
# Date  : 2016.06.19
# Author: jiuqian(81691)
# Email : jianhui.jjh@alibaba-inc.com
#===============================================================================

import numpy as np
import os
import matplotlib.pyplot as plt

# print the san dian tu
def drawScatterDiagram(fName):
	xrecord = []
	yrecord = []

	fr = open(fName)
	for line in fr.readlines():
		lineArr = line.strip().split()
		xrecord.append( float(lineArr[1]) )
		yrecord.append( float(lineArr[2]) )
	plt.scatter(xrecord, yrecord, c='red', marker='s')
	plt.axis([80,260,5,35])

	# the best linear regression
	a = 0.1612; b = -8.6394
	x = np.arange(90.0,250.0,0.1)
	y = a*x + b
	plt.plot(x,y)

	plt.show()

if __name__ == "__main__":
	fName = "./linear_regression.txt"
	drawScatterDiagram(fName)
