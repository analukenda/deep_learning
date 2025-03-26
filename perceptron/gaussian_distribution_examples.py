import numpy as np
from matplotlib import pyplot as plt

from lineary_separable_2D import plot,initp,trainlms_p

A1=np.random.normal((10, 10), (2.5, 2.5), size=(100,2)).T
A2=np.random.normal((20, 5), (2, 2), size=(100,2)).T

A=np.hstack([A1, A2])
C=np.hstack([np.zeros((1,100)), np.ones((1, 100))]).astype(int)

w=initp(A,C)
w,err=trainlms_p(0.1,A,C,w,1000)
plot(w,A,C)

# Krivo su klasificirana 2 primjera (1 crveni i 1 plavi).
# Bio bi klasificiran kao crveni primjer, ali je jako blizu granice te klasifikacija ovisi o stohastičnosti procesa
# odnosno konačnoj dobivenoj granici koja ovisi o inicijalnim vrijednostima težina.

# Dva perceptrona
A=np.array([[0.1, 0.7, 0.8, 0.8, 1.0, 0.3, 0.0, -0.3, -0.5, -1.5], [1.2, 1.8, 1.6, 0.6, 0.8, 0.5, 0.2, 0.8, -1.5, -1.3]])

C=np.array([[1, 1, 1, 0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]);

w=initp(A,C)
w,err=trainlms_p(0.1,A,C,w,1000)
plt.plot(range(1,len(err)+1),err)
plt.xlabel('Iteracija')
plt.ylabel('Greska')
plt.show()
