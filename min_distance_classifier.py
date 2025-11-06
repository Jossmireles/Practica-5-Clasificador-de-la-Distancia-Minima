import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# ==============================================================================
# 1. IMPLEMENTACIÓN DEL CLASIFICADOR DE DISTANCIA MÍNIMA
# ==============================================================================

class MinDistanceClassifier:
    """Implementa el clasificador de Distancia Mínima (Euclidiana)."""
    
    def fit(self, X_train, y_train):
        self.classes_ = np.unique(y_train)
        self.centroids = {}
        for c in self.classes_:
            self.centroids[c] = X_train[y_train == c].mean(axis=0)
        return self

    def predict(self, X_test):
        predictions = []
        for sample in X_test:
            distances = {}
            for c in self.classes_:
                distances[c] = np.linalg.norm(sample - self.centroids[c])
            prediction = min(distances, key=distances.get)
            predictions.append(prediction)
        return np.array(predictions)

# ==============================================================================
# 2. CARGA Y PREPROCESAMIENTO DE DATOS (SOLUCIÓN FINAL Y ROBUSTA)
# ==============================================================================

def load_and_preprocess(file_path="heart.csv"):
    """
    Carga heart.csv, aplica One-Hot Encoding a todas las variables NO NUMÉRICAS, 
    y estandariza los datos.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {file_path}. Asegúrate de que esté en la carpeta P5_IA.")
        exit()

    # 1. Normalizar nombres de columnas a minúsculas
    df.columns = df.columns.str.lower() 

    # 2. BÚSQUEDA DINÁMICA DE LA COLUMNA OBJETIVO
    target_column = None
    if 'heartdisease' in df.columns:
        target_column = 'heartdisease'
    elif 'target' in df.columns:
        target_column = 'target'
    
    if target_column is None:
        print("\nFATAL ERROR: No se pudo identificar la columna objetivo ('heartdisease' o 'target').")
        print("Columnas encontradas:", df.columns.tolist())
        exit()
    
    print(f"✅ Columna objetivo identificada como: '{target_column}'")

    # 3. IDENTIFICAR Y CODIFICAR AUTOMÁTICAMENTE COLUMNAS NO NUMÉRICAS
    # Identifica todas las columnas que no son ni números enteros ni flotantes.
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_cols:
        # En caso de que no detecte ninguna columna object/category, revisamos por las más comunes.
        # Esto sucede si pandas lee las columnas como números/strings mixtos.
        categorical_cols = [col for col in ['sex', 'chestpaintype', 'restingecg', 'exerciseangina', 'slope', 'thal'] if col in df.columns]

    print(f"✅ Columnas a codificar (contienen texto/object): {categorical_cols}")

    # Aplicar One-Hot Encoding a las columnas detectadas (o la lista por defecto)
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # 4. Separar X y Y
    X = df.drop(target_column, axis=1).values
    y = df[target_column].values
    
    # 5. Estandarización de Datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

if __name__ == '__main__':
    X, y = load_and_preprocess()
    print("Dataset 'heart.csv' cargado, codificado y estandarizado.")
    print(f"Forma de X (Características): {X.shape}, Forma de Y (Etiquetas): {y.shape}")

    # ==============================================================================
    # 3. VALIDACIÓN HOLD-OUT 70/30 (El resto del código es el mismo)
    # ==============================================================================

    print("\n--- RESULTADOS HOLD-OUT 70/30 ---")
    
    X_train_ho, X_test_ho, y_train_ho, y_test_ho = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    mdc_ho = MinDistanceClassifier()
    mdc_ho.fit(X_train_ho, y_train_ho)
    y_pred_ho = mdc_ho.predict(X_test_ho)

    accuracy_ho = accuracy_score(y_test_ho, y_pred_ho)
    conf_matrix_ho = confusion_matrix(y_test_ho, y_pred_ho)

    print(f"Accuracy (Hold-Out): {accuracy_ho:.4f}")
    print("Matriz de Confusión:\n", conf_matrix_ho)
    print("------------------------------------------")

    # ==============================================================================
    # 4. VALIDACIÓN 10-FOLD CROSS-VALIDATION
    # ==============================================================================

    print("\n--- RESULTADOS 10-FOLD CROSS-VALIDATION ---")

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    mdc_cv = MinDistanceClassifier()

    accuracies_cv = []
    total_conf_matrix_cv = np.zeros((2, 2), dtype=int) 

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train_cv, X_test_cv = X[train_index], X[test_index]
        y_train_cv, y_test_cv = y[train_index], y[test_index]
        
        mdc_cv.fit(X_train_cv, y_train_cv)
        y_pred_cv = mdc_cv.predict(X_test_cv)
        
        accuracies_cv.append(accuracy_score(y_test_cv, y_pred_cv))
        total_conf_matrix_cv += confusion_matrix(y_test_cv, y_pred_cv)

    mean_accuracy_cv = np.mean(accuracies_cv)
    std_accuracy_cv = np.std(accuracies_cv)

    print(f"Accuracy Promedio (10-Fold CV): {mean_accuracy_cv:.4f} (+/- {std_accuracy_cv:.4f})")
    print("Matriz de Confusión Total (Acumulada de 10 Folds):\n", total_conf_matrix_cv)
    print("-------------------------------------------------")