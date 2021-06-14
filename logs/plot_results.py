# -*- coding: utf-8 -*-
"""
@author: bishwo

Usages: experiments results and visualization 
Run as: python3 plot_results.py
"""

import matplotlib
import matplotlib.pyplot as plt 
import numpy as np

# to get rid of type 3 font 
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Confidence plot 
plt.figure(1, figsize = (6,3))
yhat = [x*0.01 for x in range(0, 100)]
positive_samples = [(-((1 - x)**8)*np.log(x)) for x in yhat]
np.seterr(divide = 'ignore') 
negative_samples = positive_samples[::-1]
lines = plt.plot(yhat, positive_samples, '-', yhat,  negative_samples, '-')
plt.setp(lines[0], linewidth=3.5)
plt.setp(lines[1], linewidth=3.5)
plt.legend(('FL for positive samples', 'FL for negative samples'), loc='upper center')
plt.xlabel(r' Model confidence $\hat y $' )
plt.ylabel('Loss')
plt.grid(True)
plt.grid(color='0.80')
plt.tight_layout()
plt.savefig('FL_plot.pdf')
plt.show()



def normalize(x):
    new_list = []
    for i in x:
        ii = ((i - min(x))/(max(x)-min(x)))
        new_list.append(ii)
    return new_list


# noise range starting from 0 to 50 %
noise = range(0, 60, 10)
# Indoor
ce_Indoor = [0.9742, 0.9508, 0.9340, 0.9163, 0.8982, 0.8652] 
fl_Indoor = [0.9684, 0.9604, 0.9570, 0.9111, 0.8480, 0.8442]

#  PASCAL VOC 
ce_VOC = [0.4363, 0.4151, 0.4001, 0.3572, 0.3463, 0.3154] 
fl_VOC = [0.5324, 0.5059, 0.4688, 0.4433, 0.3978, 0.3312]

# FDDB Face dataset
ce_faces  = [0.9502, 0.9374, 0.9317, 0.9109, 0.8820, 0.8672] 
fl_faces  = [0.9602, 0.9526, 0.9328, 0.9149, 0.8646, 0.7774]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (6,3))
ax1.plot(noise, fl_Indoor, 's-', noise, ce_Indoor, 'o-')
ax2.plot(noise, fl_VOC, 's-', noise, ce_VOC, 'o-')
ax3.plot(noise, fl_faces, 's-', noise, ce_faces, 'o-')
ax1.set_title('(a) Indoor')
ax2.set_title('(b) PASCAL VOC')
ax3.set_title('(c) FDDB')
ax3.legend(('FL loss', 'CE loss'), loc='lower left')
ax1.set(ylabel='mAP')
ax2.set(xlabel='Noise level [%]')
plt.tight_layout()
# plt.grid(axis='x', color='0.90')
plt.savefig('losses_mAP.pdf')
plt.show()
# plt.imshow(X, alpha=)


# gammas from 0 to 8
gammas = range(9)

# Indoor gammas
zero_Indoor =  [0.9673, 0.9701, 0.9684, 0.9676, 0.9805, 0.9780, 0.9683, 0.9637, 0.9402]
ten_Indoor  =  [0.9601, 0.9649, 0.9604, 0.9766, 0.9788, 0.9663, 0.9760, 0.9521, 0.9335]
fifty_Indoor = [0.7821, 0.7899, 0.8442, 0.8540, 0.8548, 0.9094, 0.8940, 0.9227, 0.8879]

# PASCAL VOC gammas
zero_VOC =  [0.4098, 0.5146, 0.5324, 0.5380, 0.5446, 0.5314, 0.5206, 0.5027, 0.4275]
ten_VOC  =  [0.4440, 0.4769, 0.5059, 0.5071, 0.5191, 0.5206, 0.4996, 0.4735, 0.4047]
fifty_VOC = [0.2845, 0.2980, 0.3349, 0.3822, 0.4233, 0.4345, 0.4128, 0.4075, 0.4343]

# FDDB Face gammas 
zero_Face =  [0.9535, 0.9565, 0.9602, 0.9620, 0.9631, 0.9647, 0.9647, 0.9607, 0.9606]
ten_Face  =  [0.9406, 0.9447, 0.9526, 0.9517, 0.9587, 0.9583, 0.9614, 0.9599, 0.9579]
fifty_Face = [0.7646, 0.7720, 0.7774, 0.8145, 0.8546, 0.8926, 0.9248, 0.9487, 0.9487]


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (6,3))
ax1.plot(gammas, zero_Indoor, 's-', gammas, ten_Indoor, 'o-', gammas, fifty_Indoor, '^-')
ax2.plot(gammas, zero_VOC, 's-', gammas, ten_VOC, 'o-', gammas, fifty_VOC, '^-')
ax3.plot(gammas, zero_Face, 's-', gammas, ten_Face, 'o-', gammas, fifty_Face, '^-')
ax1.set_title('(a) Indoor')
ax2.set_title('(b) PASCAL VOC')
ax3.set_title('(c) FDDB')
ax3.legend(('0% ', '10% ', '50% '), loc='lower right')
ax1.set(ylabel='mAP')
# ax3.set(ylabel='AP')
ax2.set(xlabel=r' Parameter $\gamma $')
plt.tight_layout()
# plt.grid(axis='x', color='0.90')
plt.savefig('gammas_plot.pdf')
plt.show()

    
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (6,3))
ax1.plot(noise, normalize(fl_Indoor), 's-', noise, normalize(ce_Indoor), 'o-')
ax2.plot(noise, normalize(fl_VOC), 's-', noise, normalize(ce_VOC), 'o-')
ax3.plot(noise, normalize(fl_faces), 's-', noise, normalize(ce_faces), 'o-')
ax1.set_title('(a) Indoor')
ax2.set_title('(b) PASCAL VOC')
ax3.set_title('(c) FDDB')
ax3.legend(('FL loss', 'CE loss'), loc='lower left')
ax1.set(ylabel='Relative decrease in mAP [%]')
ax2.set(xlabel='Noise level [%]')
plt.tight_layout()
plt.savefig('losses_relative.pdf')
plt.show()
