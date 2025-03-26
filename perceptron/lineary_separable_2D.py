import numpy as np
import matplotlib.pyplot as plt

def initp(data, labels):
    return -0.5+np.random.rand(labels.shape[0], data.shape[0]+1)

def predict(W, A):
    return (W@np.vstack([-np.ones((1, A.shape[1])), A])>=0).astype(int)

def plot(W, A, C):
    x_start, x_end=A[0, :].min()-1, A[0, :].max()+1
    y_start, y_end=A[1, :].min()-1, A[1, :].max()+1

    xx, yy=np.meshgrid(np.arange(x_start, x_end, 0.01), np.arange(y_start, y_end, 0.01))
    grid=np.vstack([xx.ravel(), yy.ravel()])

    Z=predict(W, grid).reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.scatter(A[0, :], A[1, :])

    plt.scatter(A[0, :], A[1, :], color=[["red", "blue"][C[0, i]] for i in range(A.shape[1])])
    plt.show()

def trainlms_p(ni, x, d, W, max_epoch):
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

if __name__=='__main__':
    a1=np.array([[1, 1]]).T
    a2=np.array([[1, 1]]).T
    a3=np.array([[2, 0]]).T
    a4=np.array([[1, 2]]).T
    a5=np.array([[2, 1]]).T

    x=[i[0][0] for i in [a1,a2,a3,a4,a5]]
    y=[i[1][0] for i in [a1,a2,a3,a4,a5]]
    A=np.array([x,y])
    plt.scatter(A[0][:3],A[1][:3],color='red')
    plt.scatter(A[0][3:],A[1][3:],color='blue')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['C0','C1'])
    plt.show()

    C = np.array([[0, 0, 0, 1, 1]])
    W=initp(A, C)
    plot(W,A,C)
    plot(W,A,C)

    W,err=trainlms_p(0.1,A,C,W,1000)
    plot(W,A,C)
    plt.plot(range(1,len(err)+1),err)
    plt.xlabel('Iteracija')
    plt.ylabel('Greska')
    plt.show()

# Granica prije treniranja ne klasificira dobro, a granica poslije treniranja klasificira dobro.
# 2D: podjela nogometnih igrača na iskusne (C0) i neiskusne (C1) ovisno o godinama treniranja (x) i broju odigranih utakmica (y)
# 3D: podjela životinja na opasne (C0) i neopasne (C1) ovisno o veličini (x), broju prijavljenih incidenata (y) i starosti (z)