import numpy as np
import generacionPoblacion as gP
import funcionesGenetico as fG
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # La BD 1 es la de Zalando, mientras que la 0 es la de los números de MNIST

    listaEv, listaPun, reinicios, listaCromosomas = fG.genetico_CHC(baseDatos=0, nPoblacion=30, numeroElite=2, distanciaHinicial=5,
                                                                    prueba=False)
    # Pinto en el fichero
    dirFichero = './Numeros.txt'
    mejorCromosoma = fG.arrayLimpio(listaCromosomas[-1])
    fG.escribeFichero(dirFichero=dirFichero, listaEv=listaEv, listaPun=listaPun, reinicios=reinicios,
                      mejorCromosoma=mejorCromosoma)

    fG.pintarGrafica(listaEv, listaPun, True, "Genetico CHC Numeros")
