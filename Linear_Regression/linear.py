#===============================================================================
# File  : script1.py
# Desc  : 
#  
# Date  : 2016.06.11
# Author: jiuqian(81691)
# Email : jianhui.jjh@alibaba-inc.com
#===============================================================================
#!/bin/bash

import numpy as np
import matplotlib.pyplot as plt

N = 100
x = np.random.rand(N, 1) * 10
y = 5 * x + 10 + 5 * np.random.rand(N, 1)
#Sample = [x y]
#save('data.mat', 'Sample')
plt.plot(x, y, '.')
plt.show()
 



#xData = np.arange(0,10,1)
#yData1 = xData.__pow__(2.0)
#yData2 = np.arange(15,61,5)

## the basic plot
#plt.figure(num=1, figsize=(8,6))
#plt.title('Plot 1', size=14)
#plt.xlabel('x-axis', size=14)
#plt.ylabel('y-axis', size=14)
#
#plt.plot(xData, yData1, color='b', linestyle='--', marker='o', label='y1 data')
#plt.plot(xData, yData2, 'o', color='r', linestyle='-', label='y2 data')
#
#plt.legend(loc='upper left')
##plt.savefig('images/plot1.png', format='png')
#plt.show()
