import numpy as np 

b = np.zeros((3,4),dtype='uint8')
print(b)

b[0,1] = 1
b[1,0] = 2

print(b)