import numpy as np

real=lambda x: np.array([[ord(character) for character in x]]).T

if __name__=='__main__':
    b1=real("vrat")
    b2=real("kraj")
    b3=real("cres")
    b4=real("otac")

    # Ortogonalni ulazni vektori
    a1 = np.array([[1, 0, 0, 0]]).T
    a2 = np.array([[0, 1, 0, 0]]).T
    a3 = np.array([[0, 0, 1, 0]]).T
    a4 = np.array([[0, 0, 0, 1]]).T

    M = b1 * a1.T + b2 * a2.T + b3 * a3.T + b4 * a4.T
    print(M)

    char=lambda x:"".join(map(chr, map(int, list(x))))

    word=char(M@a1)
    print(word)

    for key in [a1, a2, a3, a4]:
        print('Kljuc: ' + str(key) + ', odziv: ' + char(M @ key))

    # Svi parovi su ispravno memorizirani.

    # Kada kljucevi ne bi bili normirani vektori, onda skup vektora kljuceva ne bi bio ortonomiran i sum ne bi bio jednak nuli,
    # odnosno postojala bi mogucnost da su neki ulazi krivo memorizirani.

    # Svojstva korelacijske matrice
    a5 = (a1 + a3) / np.sqrt(2)

    b5 = real("mrak")
    M_five = b1 * a1.T + b2 * a2.T + b3 * a3.T + b4 * a4.T + b5 * a5.T

    for key in [a1, a2, a3, a4, a5]:
        print('Kljuc: ' + str(key) + ', odziv: ' + char(M_five @ key))

    # Nova rijec nije uspjesno memorizirana

    # Od prethodnih rijeci, neke su ostale dobro memorizirane, a neke ne. Dobro su memorizirane druga i cetvrta rijec: kraj i otac,
    # a prva i treca, vrat i cres, nisu.

    # Neke rijeci nisu ispravno memorizirane zato sto novi ulazni kljuc, a5, sa prethodnim ulaznim kljucevima ne cini
    # skup ortonomiranih vektora, stoga sum vise nije jednak nuli, a i dimenzija ulaznih i izlaznih vektora je 4 sto znaci
    # da je kapacitet memorije 4 (moze ispravno zapamtiti 4 para), a ne 5 koliko ih sad imamo. Pretpostavljam da prva i treca
    # rijec nisu dobro memorizirane upravo zato sto se njihovi kljucevi koriste za racunaje kljuca nove rijeci.

    # Parovi riječi kao asocijacije
    a1 = real("ruka")
    a2 = real("kset")
    a3 = real("more")
    a4 = real("mama")
    M = b1 * a1.T + b2 * a2.T + b3 * a3.T + b4 * a4.T
    print(M)

    # for key in [a1, a2, a3, a4]:
    #    print('Kljuc: ' + char(key) + ', odziv: ' + char(M @ key))

    # Nemoguce je uopce ispisati ijedan par ulaz-izlaz, sto znaci da nijedan par nije dobro memoriziran.
    # Razlog je sto skup ulaznih vektora kljuceva nije ortonomiran te je sum prevelik.
    # Problem mozemo rijesiti ortonormalizacijom ulaznih vektora npr. Gram-Schmidt procedurom.

    # Ortogonalizacija ulaznih vektora
    A=np.hstack([a1, a2, a3, a4])

    from scipy.linalg import orth
    C=orth(A.T)

    c1=np.array([C[0]]).T
    c2=np.array([C[1]]).T
    c3=np.array([C[2]]).T
    c4=np.array([C[3]]).T

    M = b1 * c1.T + b2 * c2.T + b3 * c3.T + b4 * c4.T

    for key in [c1, c2, c3, c4]:
        print('Kljuc: ' + str(key) + ', odziv: ' + char(M @ key))

    # Efekt ortnormalizacije vektora je da skup ulaznih vektora postaje ortogonalan, sto znaci da u memoriji nema vise suma
    # i u ovom slucaju ispravno se mogu zapamtiti cetiri para.
    # Svi su parovi ispravno zapamceni.
    # Kada su vektori normirani onda odziv mozemo pisati kao b = bj +vj, pri cemu je bj zeljeni odziv, a vj preslusavanje
    # izmedu kljuca aj i ostalih kljuceva u memoriji.
    # Ukoliko skup napravimo ortogonalnih, i ako je uz to i normiran tj. skupa ortnormiran, mozemo ocekivati da ce
    # memorija ispravno zapamtiti broj parova jednak dimenziji vektora.
    # Ako je broj linearno nezavisnih vektora dimenzije p jednak p onda je kapacitet memorije jednak p, ukoliko sustav
    # napravimo ortogonalnim.

    # Inverz matrice
    B=np.hstack([b1, b2, b3, b4])
    M=B@np.linalg.inv(A)

    for key in [a1, a2, a3, a4]:
        print('Kljuc: ' + str(key) + ', odziv: ' + char(np.round(M @ key)))

    # Svi su parovi ispravno memorizirani.

    # Pseudoinverz matrice
    a1 = np.array([[1, 0, 0, 0]]).T
    a2 = np.array([[0, 1, 0, 0]]).T
    a3 = np.array([[0, 0, 1, 0]]).T
    a4 = np.array([[0, 0, 0, 1]]).T

    a5 = (a1 + a3) / np.sqrt(2)

    A=np.hstack([a1, a2, a3, a4, a5])
    B=np.hstack([b1, b2, b3, b4, b5])
    A_pseudo=A.T@np.linalg.inv(A@A.T)
    M=B@A_pseudo

    keys=[a1,a2,a3,a4,a5]
    outputs=[b1,b2,b3,b4,b5]
    for i in range(len(keys)):
        if char(outputs[i]) == char(M@keys[i]):
            print('Ispravno zapamcen par - kljuc: '+str(keys[i])+', odziv: '+char(M@keys[i]))
        else:
            print('Neispravno zapamcen par - kljuc: '+str(keys[i])+', odziv: '+char(M@keys[i]))

    print('Greska memorije: '+str(np.linalg.norm(B-M@A)))
