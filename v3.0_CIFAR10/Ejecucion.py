import numpy as np
import generacionPoblacion as gP
import funcionesGenetico as fG
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    inicio = time.time()
    listaEv, listaPun, reinicios, listaCromosomas = fG.genetico_CHC( nPoblacion=30, numeroElite=2,
                                                                    distanciaHinicial=5, prueba=False)
    fin = time.time()
    tiempo = fin - inicio
    print(f"Tiempo ejecucion: {tiempo}")
    # Pinto en el fichero
    dirFichero = './CIFAR10.txt'

    mejorCromosoma = fG.arrayLimpio(listaCromosomas[-1])
    fG.escribeFichero(dirFichero=dirFichero, listaEv=listaEv, listaPun=listaPun, reinicios=reinicios,
                      mejorCromosoma=mejorCromosoma)

    fG.pintarGrafica(listaEv, listaPun, True, "Genetico CHC CIFAR10")
