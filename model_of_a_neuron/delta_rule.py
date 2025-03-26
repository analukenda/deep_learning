import matplotlib.pyplot as plt
import random
from neuron import neuron, sigmoid_function

x1 = [-1, 0, 0]
x2 = [-1, 0, 1]
x3 = [-1, 1, 0]
x4 = [-1, 1, 1]

# Write your code here
x=[x1,x2,x3,x4]
y=[0,0,0,1]

etas=[0.05,0.1,0.15,0.25,0.3,0.5,0.7]
colors=['red','blue','green','yellow','orange','purple','brown']
w=[[random.random() for i in range(3)] for j in range(3)]

def learning(x,y,w,eta):
    iter=0
    err=1000
    y_axis=[]
    w_pom=w.copy()
    while(err>0.001):
        iter+=1
        err=0.0
        for i in range(len(x)):
            rez=neuron(x[i][1:],w_pom,sigmoid_function)
            e=y[i]-rez
            err+=(0.5*e*e)
            for j in range(len(w_pom)):
                w_pom[j]+=(eta*x[i][j]*e)
        y_axis.append(err)
    return w_pom, y_axis, iter

for i in range(len(etas)):
    rez,y_axis,iter=learning(x,y,w[0],etas[i])
    plt.plot(range(1,iter+1),y_axis,color=colors[i])

plt.xlabel('Broj iteracija')
plt.ylabel('Greska')
plt.legend(['Eta = '+str(i) for i in etas])
plt.show()

for different_w in w[1:]:
    rez, y_axis, iter = learning(x, y, different_w, 0.15)
    print('S pocetnim vrijednostima tezina: '+str(different_w)+'i eta = 0.15 u '
          +str(iter)+' iteracija dobivena greska '+str(y_axis[-1]))

# U ovom slucaju najbolja je najveca stopa ucenja, odnosno eta = 0.7, iz razloga sto sustav nije nestabilan, sam zadatak
# nije prekompliciran te se sa vecom stopom ucenja dolazi do zeljenog rezultata u manje iteracija odnosno u manje vremena.
# S malom stopom ucenja ucenje traje duze (sto vidimo iz grafa, najvise iteracija, njih preko 4000 potrebno je sa stopom
# ucenja 0.05), ali s prevelikom stopom sustav moze biti nestabilan te stoga treba traziti neku zlatnu sredinu (npr. eta=0.15).

# Za dovoljno malu gresku sam stavila 0.001, naravno uvijek je bolje da je greska sto manja (osim kad nas brine prenaucenost,
# ali u ovom slucaju to ne razmatramo), te sam stoga stavila 0.001. Prvo sam kao kriterij postavila gresku manju od 0.01, ali
# kad sam vidjela da ne treba puno vise vremena za dostizanje greske od 0.001 postavila sam tako jer je ipak manja greska tj.
# tocniji sustav.

# Iz grafa je vidljivo da broj iteracija ovisi o stopi ucenja, posto u zadatku nije postavljeno da kriterij prestanka rada
# bude dostignut broj iteracija. Za eta = 0.05 potrebno je preko 4000 iteracija, a za eta = 0.7 oko 500, tj. broj itreacija
# je obrnuto proporcionalan stopi ucenja.
