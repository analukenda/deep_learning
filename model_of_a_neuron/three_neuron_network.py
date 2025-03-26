from neuron import neuron, sigmoid_function

def three_neuron_network(x):
    w1=[1,0.5,1,-0.4]
    w2=[0.5,0.6,-1.5,-0.7]
    w3=[-0.5,-1.5,0.6]
    w=[w1,w2]
    first_layer_out=[]
    for i in range(2):
        first_layer_out.append(neuron(x,w[i],sigmoid_function))
    return neuron(first_layer_out,w3,sigmoid_function)

print('Rezultat mreze: '+str(three_neuron_network([0.3,0.7,0.9])))

# Naravno da rezultat mreže ovisi o težinama neurona, ovisno o vrijednosti težina će
# izlazna suma biti drukčija te će ovisno o tome izlaz aktivacijske funkcije biti drukčiji te cjeloukupni ilaz neurona,
# koji utječe na rezultate daljnjih slojeva mreže.