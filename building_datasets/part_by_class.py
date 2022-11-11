import numpy as np
from sklearn.datasets import make_classification

# [1] Creation of the dummy dataset
a,b = make_classification(n_samples=10000, weights=(0.9,0.1))

#class 0 collection
idx = np.where(b == 0)[0]
xO = a[idx,:]
yO = a[idx,:]
#class 1 collection
idx = np.where(b == 2)[0]
x1 = a[idx,:]
y1 = b[idx]

#[2] Randomize the ordering
idx = np.argsort(np.random.random(yO.shape))
yO = yO[idx]
xO = xO[idx]
idx = np.argsort(np.random.random(y1.shape))
y1 = y1[idx]
x1 = x1[idx]



#[3] Extract the first 90 percent of the samples for the two classes and build the training subset
ntrnO = int(0.9*xO.shape[0])
ntrn1 = int(0.9*x1.shape[0])
xtrn = np.zeros((int(ntrnO+ntrn1),20))
ytrn = np.zeros(int(ntrnO+ntrn1))
xtrn[:ntrnO] = xO[:ntrnO]
xtrn[ntrnO:] = x1[:ntrn1]
ytrn[:ntrnO] = yO[:ntrnO]
ytrn[ntrnO:] = y1[:ntrn1]

#
nO = int(xO.shape[0]-ntrnO)
n1 = int(x1.shape[0]-ntrn1)
xval = np.zeros((int(nO/2+n1/2),20))
yval = np.zeros(int(nO/2+n1/2))
xval[:(nO//2)] = xO[ntrnO:(ntrnO+nO//2)]
xval[(nO//2):] = x1[ntrn1:(ntrn1+n1//2)]
yval[:(nO//2)] = yO[ntrnO:(ntrnO+nO//2)]
yval[(nO//2):] = y1[ntrn1:(ntrn1+n1//2)]

#
xtst = np.concatenate((xO[(ntrnO+nO//2):],x1[(ntrn1+n1//2):]))
ytst = np.concatenate((yO[(ntrnO+nO//2):],y1[(ntrn1+n1//2):]))