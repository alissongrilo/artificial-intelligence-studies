import numpy as np

def matriz_confusao(y_real, y_previsto):
    classes = np.unique(y_real)
    n_classes = len(classes)
    matriz = np.zeros((n_classes, n_classes), dtype=int)
    
    for real, previsto in zip(y_real, y_previsto):
        matriz[real][previsto] += 1
    
    return matriz

def acuracia(y_real, y_previsto):
    corretos = sum(r == p for r, p in zip(y_real, y_previsto))
    return corretos / len(y_real)

def precisao(y_real, y_previsto, classe):
    matriz = matriz_confusao(y_real, y_previsto)
    tp = matriz[classe, classe]
    fp = np.sum(matriz[:, classe]) - tp
    return tp / (tp + fp) if (tp + fp) != 0 else 0

def revocacao(y_real, y_previsto, classe):
    matriz = matriz_confusao(y_real, y_previsto)
    tp = matriz[classe, classe]
    fn = np.sum(matriz[classe, :]) - tp
    return tp / (tp + fn) if (tp + fn) != 0 else 0

def precisao_macro(y_real, y_previsto):
    classes = np.unique(y_real)
    return np.mean([precisao(y_real, y_previsto, c) for c in classes])

def revocacao_macro(y_real, y_previsto):
    classes = np.unique(y_real)
    return np.mean([revocacao(y_real, y_previsto, c) for c in classes])