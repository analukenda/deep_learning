import numpy as np
import math
import random

def step_function(x):
    if x >= 0:
        return 1
    return 0

def ramp_function(x):
    if x >= 0:
        return x
    return 0

def sigmoid_function(x, a=1):
    return 1 / (1+math.pow(math.e,-a*x))

def neuron(x, w, activation):
    sum =0.0
    sum -= w[0]
    for i in range(len(x)):
        sum+=(x[i]*w[i+1])
    return activation(sum)

if __name__=='__main__':
    x1=[0.5,1,0.7]
    x2=[0,0.8,0.2]

    w=[random.random() for i in range(4)]
    print('Generirane tezine: ')
    for i in range(4):
        print('w'+str(i)+': '+str(w[i]))

    for x in [x1,x2]:
        print('Aktivacija za ulaz '+str(x)+':')
        for activation in [step_function,ramp_function,sigmoid_function]:
            print('Funkcija: '+str(activation.__name__)+', rezultat: '+str(neuron(x,w,activation)))

