# coding: utf8
#===============================================================================
# File  : linear_regression_child.py
# Desc  : 
#本文是多元线性回归的练习，这里练习的是最简单的二元线性回归，
#参考斯坦福大学的教学网: http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=DeepLearning&doc=exercises/ex2/ex2.html。
#本题给出的是50个数据样本点，
#	X为这50个小朋友到的年龄，年龄为2岁到8岁，年龄可有小数形式呈现。
#	Y为这50个小朋友对应的身高，当然也是小数形式表示的。
#现在的问题是要根据这50个训练样本，估计出3.5岁和7岁时小孩子的身高。
#通过画出训练样本点的分布凭直觉可以发现这是一个典型的线性回归问题。
#  
# Date  : 2016.07.19
# Author: jiuqian(81691)
# Email : jianhui.jjh@alibaba-inc.com
#===============================================================================
#!/bin/bash

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation 

def getGradientDescent(xrecord, yrecord, curB, curW):
	gd_b = 0
	gd_w = 0
	for i in range(0, len(xrecord)):
		gd_b = gd_b + (curB + curW * xrecord[i] - yrecord[i]) * 1
		gd_w = gd_w + (curB + curW * xrecord[i] - yrecord[i]) * xrecord[i]

	return [gd_b, gd_w]

def gd_iteration(xrecord, yrecord, iter_max_num, learningRate):
	curB = 0
	curW = 0  # init the curW
	m = len(xrecord)

	for j in range(0, iter_max_num):
		print "[DEG] iter_num:{0}, curB = {1}, curW:{2}".format(j, curB, curW)
		gd_b, gd_w  = getGradientDescent(xrecord, yrecord, curB, curW)
		curB = curB - learningRate * gd_b / m 
		curW = curW - learningRate * gd_w / m 

	return [curB, curW]



# ----------------------------------------------------------------------
# load the data 
xrecord = np.loadtxt('./ex2Data/ex2x.dat', dtype='float')
yrecord = np.loadtxt('./ex2Data/ex2y.dat', dtype='float')

print xrecord
print yrecord

# plot the scooter picture

#fig = plt.figure( figsize=(200,200) )
fig = plt.figure()
axes1 = fig.add_subplot(111)

axes1.scatter(xrecord, yrecord, c='blue', marker="o")
#axes1.xlabel("Age in Years")
#axes1.ylabel("Height in meters")

line_best, = axes1.plot([], [], lw=2)
line_gd,   = axes1.plot([], [], lw=2)

b = 0
w = 0
m = len(xrecord)
iter_index = 0
iter_max_num = 500
learningRate = 0.07

def init():
	
	# show the best line regression result 
	#plt.plot(xrecord, 0.0639*xrecord+0.7502, c='red', label='best result')
	
	# use the gd to get the best lr result
	#curB, curW = gd_iteration(xrecord, yrecord, iter_max_num, learningRate)
	#plt.plot(xrecord, curB+curW*xrecord, 'b--', label='gd result')

	line_best.set_data(xrecord, 0.0639*xrecord+0.7502)
	line_gd.set_data(xrecord, [])

	return line_best, line_gd

def animate(i):
	global b, w, learningRate, m, xrecord, yrecord
	gd_b, gd_w = getGradientDescent(xrecord, yrecord, b, w)	
	b = b - learningRate * gd_b / m
	w = w - learningRate * gd_w / m

	#line_best.set_data(xrecord, 0.0639*xrecord+0.7502)
	line_gd.set_data(xrecord, b+w*xrecord)

	return line_gd



anim1=animation.FuncAnimation(fig, animate, init_func=init,  frames=1, interval=600)#, blit=True)
#plt.legend()
plt.show()



