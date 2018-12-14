# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 16:53:59 2018

@author: karan.verma
"""

import matplotlib.pyplot as plt
import numpy as np

y = np.random.random(100)
x = np.arange(100)

d = zip(x, y)

fig, ax = plt.subplots(1)

plt.subplot(111)

for i,j in d:
    plt.subplot(111).annotate(str(j), xy=(i,j))
    
    
plt.subplot(111).plot(x, y)


