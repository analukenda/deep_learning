import numpy as np
from matplotlib import pyplot as plt
from lineary_separable_2D import trainlms_p, initp

a1=np.array([[0, 0, 0]]).T
a2=np.array([[0, 0, 1]]).T
a3=np.array([[0, 1, 0]]).T
a4=np.array([[0, 1, 1]]).T
a5=np.array([[1, 0, 0]]).T

A=np.hstack([a1, a2, a3, a4, a5])
C=np.array([[0, 1, 0, 0, 1]])

import math

w = initp(A, C)
w, err = trainlms_p(0.1, A, C, w, 1000)
plt.plot(range(1, len(err) + 1), err)
plt.xlabel('Iteracija')
plt.ylabel('Greska')
plt.show()

N = len(C)
w_init = initp(A, C)
found = False
while (True):
    # Mijenjamo klasu za po jedan primjer
    for i in range(N):
        c_alter = C.copy()
        c_alter[0][i] = math.fabs(C[0][i] - 1)
        w, err = trainlms_p(0.1, A, c_alter, w_init, 1000)
        if err[-1] >= 0.02:
            print('Linearno neodvojiv za klase: ' + str(c_alter))
            found = True
            break
    if found is True:
        break

    # Mijenjamo klasu za po dva primjera
    for i in range(N):
        for j in range(i + 1, N):
            c_alter = C.copy()
            c_alter[0][i] = math.fabs(C[0][i] - 1)
            c_alter[0][j] = math.fabs(C[0][j] - 1)
            w, err = trainlms_p(0.1, A, c_alter, w_init, 1000)
            if err[-1] >= 0.02:
                print('Linearno neodvojiv za klase: ' + str(c_alter))
                found = True
                break
        if found is True:
            break
    # Mijenjamo klasu za po tri primjera.
    trojke = [[1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 3, 4], [1, 3, 5], [1, 4, 5], [2, 3, 4], [2, 3, 5], [2, 4, 5],
              [3, 4, 5]]
    for trojka in trojke:
        c_alter = C.copy()
        c_alter[0][trojka[0] - 1] = math.fabs(C[0][trojka[0] - 1] - 1)
        c_alter[0][trojka[1] - 1] = math.fabs(C[0][trojka[1] - 1] - 1)
        c_alter[0][trojka[2] - 1] = math.fabs(C[0][trojka[2] - 1] - 1)
        w, err = trainlms_p(0.1, A, c_alter, w_init, 1000)
        if err[-1] >= 0.02:
            print('Linearno neodvojiv za klase: ' + str(c_alter))
            found = True
            break
    if found is True:
        break

    # Mijenjamo klasu za po 4 primjera.
    for i in range(N):
        c_alter = C.copy()
        for j in range(N):
            if i == j:
                continue
            c_alter[0][j] = math.fabs(C[0][j] - 1)

        w, err = trainlms_p(0.1, A, c_alter, w_init, 1000)
        if err[-1] >= 0.02:
            print('Linearno neodvojiv za klase: ' + str(c_alter))
            found = True
            break
    if found is True:
        break
    # Mijenjamo klasu za svih 5 primjera.
    c_alter = C.copy()
    for i in range(N):
        c_alter[0][i] = math.fabs(C[0][i] - 1)
    w, err = trainlms_p(0.1, A, c_alter, w_init, 1000)
    if err[-1] >= 0.02:
        print('Linearno neodvojiv za klase: ' + str(c_alter))
        found = True
        break
    else:
        print('Primjer je uvijek linearno odvojiv.')
        break

# Zadatak postaje linearno neodvojiv kada od 5 zadanih primjera bar trima promijenimo klasu, u nasem slucaju kada
# umjesto oznaka [0, 1, 0, 0, 1] zadamo oznake [1, 0, 0, 1, 1] tj. promijenimo klase prvom, drugom i cetvrtom primjeru.