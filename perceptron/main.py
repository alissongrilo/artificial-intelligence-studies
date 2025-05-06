import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate(data, target, dataset_name):
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.3, random_state=42
    )

    # Normalizar os dados
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Inicializar o MLPClassifier como um perceptron
    model = MLPClassifier(hidden_layer_sizes=(), max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Previsões
    y_pred = model.predict(X_test)

    # Avaliação
    print(f"\n=== Avaliação para {dataset_name} ===")
    print("Acurácia:", accuracy_score(y_test, y_pred))
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f'Matriz de Confusão - {dataset_name}')
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.show()

# Treinando com o dataset Iris
iris = load_iris()
train_and_evaluate(iris.data, iris.target, "Iris")

# Treinando com o dataset Wine
wine = load_wine()
train_and_evaluate(wine.data, wine.target, "Wine")
