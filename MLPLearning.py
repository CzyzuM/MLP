import pickle
import numpy as np
from matplotlib import pyplot as plt
from random import uniform

with open('train.pkl','rb') as file:
    x, y = pickle.load(file)

def preprocessing(X):
    """
    Wycina obrazki 28x28 z kwadratu 36x36, innymi słowy wycina szum z obrazków
    param: X - surowe obrazki z szumem 36x36
           noise_threshold - dopuszczalny poziom szumu. Wektor o wartości średniej mniejszej niż ten próg traktujemy,
                             jako zawierający obrazek
    return: X_img - obrazki 28x28, które nas interere
    """
    X = X.reshape((x.shape[0],36,36))
    X_img = []
    for image in X:
        frame_brightness_sum = np.zeros(shape = (8,8))
        for i in range(8):
            for j in range(8):
                frame_brightness_sum[i,j] = np.sum(image[i,j:j+28]) + np.sum(image[i+28,j:j+28]) + np.sum(image[i+1:i+27,j]) + np.sum(image[i+1:i+27,j+28])
        min_frame_args = np.unravel_index(np.argmin(frame_brightness_sum, axis=None), frame_brightness_sum.shape)
        X_img.append(image[min_frame_args[0]:min_frame_args[0]+28 , min_frame_args[1]:min_frame_args[1]+28])
    return np.array(X_img).reshape((x.shape[0],28*28))

def initLayerWeights(neuronCount, prevLayerNeuronCount):
    """
    Funkcja inicjalizująca macierz wag połączeń między neuronami kolejnych warstw sieci oraz wektor bias-ów
    :param neuronCount: ilość neuronów w bieżącej warstwie
    :param prevLayerNeuronCount: ilość neuronów w poprzedniej warstwie
    :return: krotka (W,B), gdzie W = macierz o wymiarach neuronCount x prevLayerNeuronCount 
                            oraz B = wektor o wymiarach neuronCount x 1
    """
    W = np.zeros(shape=(neuronCount,prevLayerNeuronCount))
    B = np.zeros(shape=(neuronCount,1))
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i,j] = uniform(-1.0,1.0)
        B[i] = uniform(-1.0,1.0)
    return (W,B)

def sigmoid(x):
    """
    Wylicz wartość funkcji sigmoidalnej dla punktów *x*.

    :param x: wektor wartości *x* do zaaplikowania funkcji sigmoidalnej Nx1
    :return: wektor wartości funkcji sigmoidalnej dla wartości *x* Nx1
    """
    return [ 1/(1+np.exp(0-xi)) for xi in x ]

def layer_response(prevLayerValues, weights):
    """
    Funkcja zwracająca obliczoną odpowiedź warstwy znając wartości neuronów z poprzedniej warstwy i wagi połączeń
    :param prevLayerValues: wektor neuronów poprzedniej warstwy
    :param weights: macierz wag połączeń między obecną a poprzednią warstwą (kolumna nr 1 to bias)
    :return: wektor odpowiedzi neuronów
    """
    prevLayerAndImaginaryBiasNeuron = np.append([1],prevLayerValues,axis = 0)
    sum = [w@prevLayerAndImaginaryBiasNeuron for w in weights]
    return sigmoid(sum)

def SSE(layerValues, expectedCategory):
    expectedValues = np.zeros(shape=np.shape(layerValues))
    expectedValues[expectedCategory] = 1
    errors = layerValues - expectedValues
    return np.sum(errors*errors)

def predict_one_with_state(x, W_in_hidden, W_hidden_out):
    """
    Funkcja predykcji zwracająca wraz z wynikiem stan sieci 
    w celu umożliwnienia przeprowadzenia propagacji wstecznej
    """
    hidden_layer_response = layer_response(x,W_in_hidden)
    output_layer_response = layer_response(hidden_layer_response,W_hidden_out)
    prediction = np.argmax(output_layer_response)
    return (prediction, hidden_layer_response, output_layer_response)

#main:
X = preprocessing(x)
#W_in_hidden = initLayerWeights(28,X.shape[1])
#W_hidden_out = initLayerWeights(10,28)

