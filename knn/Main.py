from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from NossoKNN import NossoKNN
from Metricas import (
    matriz_confusao,
    acuracia,
    precisao_macro,
    revocacao_macro)
import time

def main():
    iris = load_iris()
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        iris.data, iris.target, test_size=0.7, random_state=42
    )

    valores_k = [1, 3, 5, 7]

    for k in valores_k:
        print(f"\n=== k = {k} ===")

        # Nosso KNN 
        inicio = time.time()
        classificador = NossoKNN(k=k)
        classificador.fit(X_treino, y_treino)
        previsoes_nosso = classificador.predict(X_teste)
        tempo_nosso = time.time() - inicio

        acuracia_nosso = acuracia(y_teste, previsoes_nosso)
        precisao_nosso = precisao_macro(y_teste, previsoes_nosso)
        recall_nosso = revocacao_macro(y_teste, previsoes_nosso)
        matriz_confusao_nosso = matriz_confusao(y_teste, previsoes_nosso)

        # Sklearn KNN
        inicio = time.time()
        knn_sklearn = KNeighborsClassifier(n_neighbors=k)
        knn_sklearn.fit(X_treino, y_treino)
        previsoes_sklearn = knn_sklearn.predict(X_teste)
        tempo_sklearn = time.time() - inicio

        acuracia_sklearn = accuracy_score(y_teste, previsoes_sklearn)
        precisao_sklearn = precision_score(y_teste, previsoes_sklearn, average='macro')
        recall_sklearn = recall_score(y_teste, previsoes_sklearn, average='macro')
        matriz_confusao_sklearn = confusion_matrix(y_teste, previsoes_sklearn)

        print("\nNosso KNN:")
        print(f"Tempo de execução: {tempo_nosso:.4f}s")
        print(f"Acurácia: {acuracia_nosso:.4f}")
        print(f"Precisão: {precisao_nosso:.4f}")
        print(f"Recall: {recall_nosso:.4f}")
        print("Matriz de Confusão:")
        print(matriz_confusao_nosso)

        print("\nSklearn KNN:")
        print(f"Tempo de execução: {tempo_sklearn:.4f}s")
        print(f"Acurácia: {acuracia_sklearn:.4f}")
        print(f"Precisão: {precisao_sklearn:.4f}")
        print(f"Recall: {recall_sklearn:.4f}")
        print("Matriz de Confusão:")
        print(matriz_confusao_sklearn)

if __name__ == "__main__":
    main()