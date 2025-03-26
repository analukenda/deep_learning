import numpy as np


def predict(W, A):
    return W@np.vstack([-np.ones((1, A.shape[1])), A])

def trainlms(ni, x, d, W, max_epoch):
    w=W.copy()

    n=0
    errors=[]
    while (n<max_epoch):
        n+=1
        y=predict(w, x)
        e=d-y
        w+=ni*e@np.vstack([-np.ones((1, x.shape[1])), x]).T
        error=np.sum(np.square(e))
        errors.append(error)
        if (error<0.02):
            break
    return w, errors

#Funkcija trainlms trenira model tj. pronalazi optimalne vrijednosti tezina w,
#na nacin da za svaki primjer prvo racuna predikciju modela sa trenutnim tezinama,
#zatim racuna gubitak modela na nacin da izracuna razliku stvarnih i izracunatih predikcija.
#Zatim azurira vrijednosti tezina na nacin da postojecim vrijednostima doda umnozak stope ucenja,
#vrijednosti znacajke i gubitka. To ce biti tezine za iducu iteraciju. Racuna gresku kao zbroj kvadrata
#gubitaka svih primjera, ako je greska manja od 0.02 to je zadovoljavajuca tocnost te zavrsava s postupkom
#i vraca izracunate tezine, ako ne nastavlja sa sljedecom iteracijom s alteriranim tezinama. Radi se o
#algoritmu least mean squarred error.

# Loading data from a file for a local notebook
data=np.loadtxt("stock.txt", delimiter=",")

# Loading data from a file for a Colab notebook
#data=np.loadtxt("/content/gdrive/My Drive/Notebooks/stock.txt", delimiter=",")

import matplotlib.pyplot as plt
plt.plot(range(1, data.shape[0]+1), data)
plt.show()

def memory(i,data,N):
    a=[]
    for j in range(i-N,i):
        a.append(data[j])
    return np.array(a).T

def memorize(data,day,N,i):
    matr=[]
    for k in range(day-i,day):
        a=memory(k,data,N)
        matr.append(a)
    return np.array(matr).T

day=151
N=70
i=50
A=memorize(data, day, N, i)

y=np.array([data[day-i+1:day+1]])

def initp(data, labels):
    return -0.5+np.random.rand(labels.shape[0], data.shape[0]+1)

W=initp(A, y)

ni=1e-8
max_num_iter=10000

W1, errors=trainlms(ni, A, y, W, max_num_iter)

p=predict(W1, A)

plt.plot(range(1, y[0].shape[0]+1), y[0])
plt.plot(range(1, p[0].shape[0]+1), p[0], c="red")
plt.show()

combinations = [[30, 20, 10000], [50, 20, 10000], [100, 50, 10000], [20, 80, 50000], [30, 80, 50000], [50, 20, 500000]]

for combination in combinations:
    i = combination[0]
    N1 = combination[1]
    max_num_iter = combination[2]
    A = memorize(data, day, N1, i)
    y1 = np.array([data[day - i + 1:day + 1]])
    W = initp(A, y1)
    W, errors = trainlms(ni, A, y1, W, max_num_iter)
    p = predict(W, A)

    plt.plot(range(1, y1[0].shape[0] + 1), y1[0])
    plt.plot(range(1, p[0].shape[0] + 1), p[0], c="red")
    print('Graf za parametre i=' + str(i) + ', N=' + str(N1) + ' i max_num_iter=' + str(max_num_iter))
    plt.show()

    # Sa grafova vidimo da su najbolji modeli 4 i 5 s parametrima i=20,N=80,max_num_iter=50000 te i=30,N=80 i max_num_iter=
    # 50000. Najgori su modeli 1 i 2 s parametrima i=30,N=20 i max_num_iter=10000 te i=50,N=20 i max_num_iter=10000. Zaključak
    # je da je model veći što je veći broj N odnosno što se više dana unazad gleda za svaki pojedinačni dan tj. da je bolje
    # imati velik N nego i. Modelu može pomoći veći broj iteracija.

a=data[day:day+1]          # price today - we assume the same price tomorrow
y=data[day+1:day+2]        # the real price tomorrow
err_oo=np.sum(np.abs(y-a)) # error

p=predict(W1, memorize(data, day+1, N, 1))
err_nn=np.sum(np.abs(y-p))

profit=err_oo-err_nn
print(profit)

print('N    i    max_num_iter    Profit')
print('------------------------------------')
for combination in combinations[:-1]:
    i = combination[0]
    N1 = combination[1]
    max_num_iter = combination[2]
    if i >= N1:
        granica = i
    else:
        granica = N1
    profit = 0.0
    for day in range(granica, len(data)):
        A = memorize(data, day, N1, i)
        y1 = np.array([data[day - i + 1:day + 1]])
        W = initp(A, y1)
        W, errors = trainlms(ni, A, y1, W, max_num_iter)
        p = predict(W, A)
        b = data[day:day + 1]  # price today - we assume the same price tomorrow
        y = data[day + 1:day + 2]  # the real price tomorrow
        err_oo = np.sum(np.abs(y - b))
        p = predict(W, memorize(data, day + 1, N1, 1))
        err_nn = np.sum(np.abs(y - p))
        profit += err_oo - err_nn
    print('{:<5}'.format(N1) + '{:<5}'.format(i) + '{:<16}'.format(max_num_iter) + str(profit))

# Bolji nacin bi bio trenirati mrezu ispocetka s podacima i za jucerasnji dan, iako je to racunalno i vremenski zahtjevnije,
# ali ce dati bolju predikciju s obzirom na dodatak vrijednosti prethodnog dana.