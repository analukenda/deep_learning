import numpy as np
from matplotlib import pyplot as plt
from lineary_separable_2D import plot,trainlms_p,initp

A=np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
C=np.array([[0, 1, 1, 0]])

w=initp(A,C)
plot(w,A,C)
w,err=trainlms_p(0.1,A,C,w,1000)
plot(w,A,C)
plt.plot(range(1,len(err)+1),err)
plt.xlabel('Iteracija')
plt.ylabel('Greska')
plt.show()

# Perceptron nije uspio riješiti XOR problem, zato što se radi o klasifikaciji linearno neodvojivih primjera (vidljivo
# sa slike da je nemoguće povući takav pravac da ispravno odvaja 4 zadane točke), a perceptron je linearan model koji kao
# rješenje daje linearnu granicu između primjera te stoga nije pogodan za rješavanje ovog problema.