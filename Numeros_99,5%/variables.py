def getValores():
    numCapasConv = [1, 2, 3]  # 0
    numCapasPooling = [1, 2, 3, 4, 5]  # 1
    numCapasDensas = [1, 2, 3]  # 2
    tamFiltros = [1, 2, 3]  # 3
    numFiltros = [1, 2, 3, 4, 5]  # 4
    numFiltrosDensas = [1, 2, 3, 4, 5]  # 5
    stride = [1, 2]  # Pasos en que se mueve la ventana de los filtros #6
    padding = [1, 2]  # 7
    Factivacion = [1, 2, 3, 4, 5, 6, 7]  # 8
    optimizadores = [1, 2, 3]  # 9
    dropout = [1, 2, 3]  # 10
    batchNorm = [1, 2]  # 11

    return [numCapasConv, numCapasPooling, numCapasDensas, tamFiltros, numFiltros, numFiltrosDensas, stride, padding,
            Factivacion, optimizadores, dropout, batchNorm]


def getValoresReales():
    numCapasConv = [3, 4, 5]  # 0
    numCapasPooling = [1, 2, 3, 4, 5]  # 1
    numCapasDensas = [0, 1, 2]  # 2
    tamFiltros = [3, 5, 7]  # 3
    numFiltros = [32, 64, 128, 256, 512]  # 4
    numFiltrosDensas = [10, 32, 64, 128, 256]  # 5
    stride = [1, 2]  # Pasos en que se mueve la ventana de los filtros
    padding = ['valid', 'same']
    Factivacion = ['elu', 'exponential', 'tanh', 'sigmoid', 'softmax', 'relu', 'selu']
    optimizadores = ['SGD', 'RMSprop', 'Adam']
    dropout = [0, 0.25, 0.5]
    batchNorm = [0, 1]

    return [numCapasConv, numCapasPooling, numCapasDensas, tamFiltros, numFiltros, numFiltrosDensas, stride, padding,
            Factivacion, optimizadores, dropout, batchNorm]


def getNumCapasConv():
    return getValores()[0]


def getNumCapasPooling():
    return getValores()[1]


def getNumCapasDensas():
    return getValores()[2]


def getTamFiltros():
    return getValores()[3]


def getNumFiltros():
    return getValores()[4]


def getNumFiltrosDensas():
    return getValores()[5]


def getStride():
    return getValores()[6]


def getPadding():
    return getValores()[7]


def getFactivacion():
    return getValores()[8]


def getOptimizadores():
    return getValores()[9]


def getDropout():
    return getValores()[10]


def getBatchNorm():
    return getValores()[11]

# Variables reales

def getNumCapasConvReal():
    return getValoresReales()[0]


def getNumCapasPoolingReal():
    return getValoresReales()[1]


def getNumCapasDensasReal():
    return getValoresReales()[2]


def getTamFiltrosReal():
    return getValoresReales()[3]


def getNumFiltrosReal():
    return getValoresReales()[4]


def getNumFiltrosDensasReal():
    return getValoresReales()[5]


def getStrideReal():
    return getValoresReales()[6]


def getPaddingReal():
    return getValoresReales()[7]


def getFactivacionReal():
    return getValoresReales()[8]


def getOptimizadoresReal():
    return getValoresReales()[9]


def getDropoutReal():
    return getValoresReales()[10]


def getBatchNormReal():
    return getValoresReales()[11]
