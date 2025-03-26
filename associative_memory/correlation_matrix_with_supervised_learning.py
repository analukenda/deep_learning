import numpy as np
from matplotlib import pyplot as plt

from forming_correlation_matrix_directly import real

a1=real("ruka")
a2=real("kset")
a3=real("more")
a4=real("mama")

b1=real("vrat")
b2=real("kraj")
b3=real("cres")
b4=real("otac")

A=np.hstack([a1, a2, a3, a4])
B=np.hstack([b1, b2, b3, b4])

M=np.random.rand(4, 4)-0.5

def trainlms(A, B, M, ni, max_num_iter, min_err=0.02):
    d=B
    x=A
    w=M

    n=0
    err=[]
    while (n<max_num_iter):
        n+=1
        e=d-w@x
        w+=ni*np.dot(e, x.T)
        err.append(np.sum(np.sum(np.multiply(e, e))))
        if (err[-1]<min_err):
            break
    return w, err

ni=0.9999/np.linalg.eig(A @ A.T)[0].max()

M, e=trainlms(A, B, M, ni, 100000)

np.round(M@A)==B

plt.plot(e)
plt.yscale("log")
plt.xscale("log")
plt.xlabel("Number of iterations")
plt.ylabel("Error")
plt.show()

iter=0
err=[1.0]
M=np.random.rand(4, 4)-0.5
iter_memorized=[]
while err[-1]>0.02:
    iter+=1
    M, err=trainlms(A,B,M,ni,1)
    memorized=np.round(M@A)==B
    sum=0
    for i in memorized:
        for j in i:
            if j==True:
                sum+=1
    iter_memorized.append(sum)

plt.plot(range(1,iter+1),iter_memorized)
plt.xlabel('Broj iteracija')
plt.ylabel('Broj ispravno zapamcenih znakova')
plt.title('Promjena broja ispravno zapamcenih znakova kroz iteracije')
plt.show()

a5=real("auto")
b5=real("mrak")
A=np.hstack([a1, a2, a3, a4,a5])
B=np.hstack([b1, b2, b3, b4,b5])

M=np.random.rand(4, 4)-0.5

ni=0.9999/max(np.linalg.eig(np.dot(A, A.T))[0])
M, e=trainlms(A, B, M, ni, 100000)
print(np.sum(np.round(np.dot(M, A))==B))

# Koristeno je 100000 iteracija, odnosno maksimalan posto nisu svi znakovi ispravno zapamceni i nije postignuta
# zadovoljavajuca tocnost.
# Tocno su zapamcena 2 znaka.
# SSE greska je 219.8 (zadnji element liste e)
# Ako pokrenemo funkciju iz početka, njena početna matrica M bit će konačna matrica M iz prethodnog pokretanja,
# sto bi trebalo znacit da ce se stvari sada bolje memeorizirati jer ce proci dodatne iteracije treniranja. Ali to nije slucaj
# i opet je zapamceno samo 2 znaka.
# Tocno su zapamcena 2 znaka i greska je i dalje 219.8, nema razlike jer je ocito postignuto najbolje moguce stanje naucenosti.
# Nije moguce, zato sto je dimenzija ulaznih vektora 4 te je moguce memorizirati 4 para, a ne 5.