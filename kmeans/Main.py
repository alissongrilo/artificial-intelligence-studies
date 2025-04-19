import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans as SklearnKMeans
import matplotlib.pyplot as plt
import time

from NossoKMeans import NossoKMeans

def rodar_kmeans_hardcore(k, dados_array):
    inicio = time.time()
    modelo = NossoKMeans(k)
    modelo.fit(dados_array)
    fim = time.time()

    score = silhouette_score(dados_array, modelo.rotulos)
    tempo = fim - inicio

    print(f"KMeans Hardcore - K={k} | Silhouette Score: {score:.4f} | Tempo: {tempo:.4f} segundos")

    return modelo.rotulos, modelo.centroides, tempo, score

def rodar_kmeans_sklearn(k, dados_array):
    inicio = time.time()
    modelo = SklearnKMeans(n_clusters=k, n_init='auto')
    modelo.fit(dados_array)
    fim = time.time()

    score = silhouette_score(dados_array, modelo.labels_)
    tempo = fim - inicio

    print(f"KMeans Sklearn - K={k} | Silhouette Score: {score:.4f} | Tempo: {tempo:.4f} segundos")

    return modelo.labels_, modelo.cluster_centers_, tempo, score

def plotar_com_pca(rotulos, centroides, dados_array, titulo):
    for n_componentes in [1, 2]:
        pca = PCA(n_components=n_componentes)
        dados_reduzidos = pca.fit_transform(dados_array)
        centroides_reduzidos = pca.transform(centroides)

        plt.figure()
        if n_componentes == 1:
            plt.xlabel('Componente 1')
            plt.scatter(dados_reduzidos[:, 0], [0] * len(dados_reduzidos), c=rotulos, cmap='viridis', s=40)
            plt.scatter(centroides_reduzidos[:, 0], [0] * len(centroides_reduzidos), c='red', s=100, marker='X')
            plt.title(f'{titulo} | PCA 1D')
        else:
            plt.ylabel('Componente 2')
            plt.scatter(dados_reduzidos[:, 0], dados_reduzidos[:, 1], c=rotulos, cmap='viridis', s=40)
            plt.scatter(centroides_reduzidos[:, 0], centroides_reduzidos[:, 1], c='red', s=100, marker='X')
            plt.title(f'{titulo} | PCA 2D')

        plt.tight_layout()
        plt.show()

def main():
    iris = load_iris()
    dados = pd.DataFrame(iris.data, columns=iris.feature_names)
    dados_array = dados.values

    for k in [3, 5]:
        print(f"\n--- Análise com K={k} ---")
        rotulos_hard, centroides_hard, tempo_hard, score_hard = rodar_kmeans_hardcore(k, dados_array)
        rotulos_sklearn, centroides_sklearn, tempo_sklearn, score_sklearn = rodar_kmeans_sklearn(k, dados_array)

        if k == 3:
            plotar_com_pca(rotulos_hard, centroides_hard, dados_array, "Hardcore")
            plotar_com_pca(rotulos_sklearn, centroides_sklearn, dados_array, "Sklearn")

        print("\nComparação de desempenho:")
        print(f"Hardcore:  Tempo = {tempo_hard:.4f}s | Silhouette = {score_hard:.4f}")
        print(f"Sklearn:   Tempo = {tempo_sklearn:.4f}s | Silhouette = {score_sklearn:.4f}")
        print("--------------------------")

if __name__ == "__main__":
    main()
