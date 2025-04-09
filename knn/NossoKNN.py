import numpy as np

class NossoKNN:
    def __init__(self, k):
        self.k = k
        self.X_treino = None
        self.y_treino = None

    def fit(self, X, y):
        self.X_treino = X
        self.y_treino = y

    def predict(self, X):
        previsoes = []

        for amostra in X:
            distancias = []

            for indice, amostra_treino in enumerate(self.X_treino):
                distancia = np.sqrt(np.sum((amostra - amostra_treino)**2))
                distancias.append((distancia, self.y_treino[indice]))
            
            distancias.sort(key=lambda par: par[0])
            vizinhos = distancias[:self.k]
            
            contagens = {}
            for _, rotulo in vizinhos:
                contagens[rotulo] = contagens.get(rotulo, 0) + 1
            
            previsao = max(contagens, key=contagens.get)
            previsoes.append(previsao)

        return previsoes