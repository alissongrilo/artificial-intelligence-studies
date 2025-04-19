import random
import numpy as np

class NossoKMeans:
    def __init__(self, k):
        self.k = k
        self.centroides = []
        self.rotulos = []

    def fit(self, dados):
        self._inicializar_centroides(dados)

        for _ in range(100):
            grupos = self._atribuir_grupos(dados)
            novos_centroides = self._calcular_novos_centroides(grupos)

            if self._convergiram(novos_centroides):
                break

            self.centroides = novos_centroides

        self._atualizar_rotulos(dados)

    def _inicializar_centroides(self, dados):
        amostras = random.sample(list(dados), self.k)
        self.centroides = np.array(amostras)

    def _atribuir_grupos(self, dados):
        grupos = [[] for _ in range(self.k)]
        for ponto in dados:
            indice = self._indice_mais_proximo(ponto)
            grupos[indice].append(ponto)
        return grupos

    def _calcular_novos_centroides(self, grupos):
        novos = []
        for grupo in grupos:
            if grupo:
                novos.append(np.mean(grupo, axis=0))
            else:
                novos.append(random.choice(self.centroides))
        return np.array(novos)

    def _convergiram(self, novos_centroides):
        return np.allclose(self.centroides, novos_centroides)

    def _indice_mais_proximo(self, ponto):
        distancias = [self._distancia(ponto, centroide) for centroide in self.centroides]
        return np.argmin(distancias)

    def _atualizar_rotulos(self, dados):
        self.rotulos = [self._indice_mais_proximo(ponto) for ponto in dados]

    def _distancia(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))
