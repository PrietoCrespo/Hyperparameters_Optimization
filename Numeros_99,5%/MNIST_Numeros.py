import numpy as np
import generacionPoblacion as gP
import funcionesGenetico as fG
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # La BD 1 es la de Zalando, mientras que la 0 es la de los números de MNIST

    '''listaEv, listaPun, reinicios, listaCromosomas = fG.genetico_CHC(baseDatos=0, nPoblacion=30, numeroElite=2, distanciaHinicial=5,
                                                                    prueba=False)
    # Pinto en el fichero
    dirFichero = './Numeros.txt'
    mejorCromosoma = fG.arrayLimpio(listaCromosomas[-1])
    fG.escribeFichero(dirFichero=dirFichero, listaEv=listaEv, listaPun=listaPun, reinicios=reinicios,
                      mejorCromosoma=mejorCromosoma)

    
'''

    individuo = [2, 3, 2, 2, 1, 1, 2, 4, 2, 1, 2, 2, 3, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 2, 4, 3, 1, 5, 2, 1, 3, 7, 3, 2, 1, 2, 1, 2, 6]


    print(fG.traducirConfiguracion(individuo))

    listaEv = [34, 53, 143, 201, 231, 259, 569, 599, 895]
    listaPun = [0.9926999807357788, 0.9930999875068665, 0.9933000206947327, 0.9939000010490417, 0.9940999746322632, 0.994700014591217, 0.9947999715805054, 0.99481000047683716, 0.99481000047683716]
    fG.pintarGrafica(listaEv, listaPun, True, "Genetico CHC MNIST_Zalando")
