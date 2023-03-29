import numpy as np
import funcionesGenetico as fG
import matplotlib.pyplot as plt


def puntuacion():
    f = open('Historial2.txt')
    puntuaciones = []
    vGeneraciones = []
    vPuntuaciones = []
    generacionMejor = -1
    mejorPuntuacion = 0
    for linea in f:
        separacion = linea.split()
        print(separacion)
        generacion = int(separacion[0].split(':')[-1])
        print(f"Generacion: {generacion}")
        puntuacion = float(separacion[-1].split(':')[-1])
        print(f"Puntuacion: {puntuacion}")
        puntuaciones.append(puntuacion)

        if puntuacion > mejorPuntuacion:
            if generacionMejor == generacion:  # Si es la misma generación
                mejorPuntuacion = puntuacion
                print(f"Vpunt: {vPuntuaciones}")
                vPuntuaciones[-1] = puntuacion
            else:  # Si es otra generacion
                generacionMejor = generacion
                mejorPuntuacion = puntuacion

                vGeneraciones.append(generacionMejor)
                vPuntuaciones.append(mejorPuntuacion)
    vGeneraciones.append(generacion)
    vPuntuaciones.append(mejorPuntuacion)
    indice = np.argmax(puntuaciones)
    print(f"Posicion: {indice}")
    print(puntuaciones[indice])
    print(f"Vector generaciones: {vGeneraciones}")
    print(f"Vector puntuaciones: {vPuntuaciones}")

    return vPuntuaciones, vGeneraciones

def traduce(vector):
    vectorTraducido = fG.traducirConfiguracion(vector)
    print(vectorTraducido)


def pintarGrafica(x, y, pinta, titulo):
    print(x)
    print(y)
    y = np.array(y)
    y = np.round(y, 6)
    maximoValor = y[y.argmax()]
    minimoValor = y[y.argmin()]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim(minimoValor, maximoValor)
    plt.plot(x, y)

    for i, j in zip(x, y):
        ax.annotate(str(j), xy=(i, j))
    if pinta:
        plt.title(titulo)
        plt.xlabel("Generación")
        plt.ylabel("Evaluacion")
        plt.show()
    print("Termino de pintar")


if __name__ == '__main__':

    vPunt, vGen = puntuacion()
    pintarGrafica(vGen, vPunt, True, 'Genetico CHC CIFAR10 ' )

