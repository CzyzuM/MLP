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
    :return: macierz wag o wymiarach neuronCount x prevLayerNeuronCount 
    """
    W = np.zeros(shape=(neuronCount,prevLayerNeuronCount))
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i,j] = uniform(-1.0,1.0)
    return W

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
    return sigmoid(weights@prevLayerValues)

def SSE(layerValues, expectedCategory):
    expectedValues = np.zeros(shape=np.shape(layerValues))
    expectedValues[expectedCategory] = 1
    errors = layerValues - expectedValues
    return np.sum(errors * errors)

def MLP_response(x, W_in_hidden, W_hidden_out):
    """
    Funkcja obliczająca stan każdej warstwy sieci po przerabianiu jednego przypadku testowego
    :param x: - przypadek testowy - obrazek 28x28 w postaci wektora
    :param W_in_hidden: - macierz wag połączeń między warstwą wejściową a warstwą ukrytą
    :param W_hidden_out: - macierz wag połączeń między warstwą ukrytą a warstwą wyjściową
    :return: krotka warstw sieci: (wejściowa, ukryta, wyjściowa)
    """
    hidden_layer_response = layer_response(x,W_in_hidden)
    output_layer_response = layer_response(hidden_layer_response,W_hidden_out)
    return (x, hidden_layer_response, output_layer_response)

def gradient_descent_single_image(MLP_state, W_in_hidden, W_hidden_out, expectedCategory):
    """
    Funkcja obliczająca przybliżenie ujemnego gradientu SSE dla jednego przypadku testowego
    :param MLP_state: obecny stan warstw: ukrytej oraz wyjściowej sieci
    :param W_in_hidden: - macierz wag połączeń między warstwą wejściową a warstwą ukrytą
    :param W_hidden_out: - macierz wag połączeń między warstwą ukrytą a warstwą wyjściową
    :return: para macierzy będących wartościami o jakie trzeba zmienić wagi poszczególnych połączeń
    """
    (input, hidden, output) = MLP_state
    W_in_hidden_changes = np.ones( shape = np.shape(W_in_hidden) )
    W_hidden_out_changes = np.ones( shape = np.shape(W_hidden_out) )

    for j in range(W_hidden_out_changes.shape[0]):
        aj = output[j]
        neuronError = 0
        if(expectedCategory == j):
            neuronError = aj - 1
        W_hidden_out_changes[j] *= np.full(shape = W_hidden_out_changes.shape[1], fill_value = (neuronError * aj * (1-aj))) * hidden

    for i in range((W_in_hidden_changes).shape[0]):
        ai = hidden[i]
        expectedValues = np.zeros(shape=np.shape(output))
        expectedValues[expectedCategory] = 1
        neuronErrors = output - expectedValues
        sumOfHigherLevel = np.transpose(W_hidden_out)[i] @ (neuronErrors * output * (np.ones(shape=np.shape(output))-output))
        W_in_hidden_changes[i] *= np.full(shape = W_in_hidden_changes[1], fill_value = (sumOfHigherLevel)*ai*(1-ai)) * input
    return(W_in_hidden_changes,W_hidden_out_changes)

def stochastic_gradient_descent( x_train, mini_batch_size, W_in_hidden, W_hidden_out ):
    for i in range(np.shape(x_train)[0] / mini_batch_size):
        batch = x_train[ mini_batch_size*i : mini_batch_size*(i+1)-1 ]
        W_in_hidden_gradient_accumulator = np.zeros( shape = np.shape(W_in_hidden) )
        W_hidden_out_gradient_accumulator = np.ones( shape = np.shape(W_hidden_out) )
        for image in batch:
            (a,b) = gradient_descent_single_image(MLP_response(image, W_in_hidden, W_hidden_out),W_in_hidden, W_hidden_out, expectedCategory)
            W_in_hidden_gradient_accumulator += a
            W_hidden_out_gradient_accumulator += b
        W_in_hidden -= W_in_hidden_gradient_accumulator / mini_batch_size
        W_hidden_out-= W_hidden_out_gradient_accumulator / mini_batch_size

#main:
X = preprocessing(x)
x_train_test = X[:1000]
#x_val = X[55000:]

#print(x_train.shape)
#print(x_val.shape)
#W_in_hidden = initLayerWeights(28,X.shape[1])
#W_hidden_out = initLayerWeights(10,28)
