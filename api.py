import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template  # Importar render_template
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from joblib import dump, load
import os
from flask_cors import CORS

# Crear la aplicación Flask
app = Flask(__name__)
CORS(app)

# Cargar los datos y preprocesar
file_path = "recursos/Crop_recommendation2esp.xls"
datos_cultivos = pd.read_csv(file_path, sep=",", encoding='latin1')

# Codificar las etiquetas de cultivos a valores numéricos
label_encoder = LabelEncoder()
datos_cultivos['label_codificada'] = label_encoder.fit_transform(datos_cultivos['label'])

# Dividir los datos en entrenamiento y prueba (80% - 20%)
train_data, test_data = train_test_split(datos_cultivos, test_size=0.2, random_state=42)

# Seleccionar columnas numéricas para normalizar
columnas_numericas = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
caracteristicas_train = train_data[columnas_numericas]

# Escalar los datos de entrada
scaler = StandardScaler()
scaler.fit(caracteristicas_train)
caracteristicas_train_normalizadas = scaler.transform(caracteristicas_train)

# Guardar perfiles normalizados de entrenamiento
perfiles_cultivos = caracteristicas_train_normalizadas
cultivos_nombres = train_data['label_codificada']

# Función para recomendar cultivo
def recomendar_cultivo(entrada_usuario):
    entrada_normalizada = scaler.transform([entrada_usuario])
    similitudes = cosine_similarity(entrada_normalizada, perfiles_cultivos)
    indice_recomendado = np.argmax(similitudes)
    cultivo_recomendado = label_encoder.inverse_transform([cultivos_nombres.iloc[indice_recomendado]])[0]
    return cultivo_recomendado

# Función para entrenar el modelo
def entrenar_modelo():
    X_train = train_data[columnas_numericas]
    y_train = train_data['label_codificada']

    # Definir el rango de hiperparámetros
    parametros = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Crear el modelo y la búsqueda
    random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=parametros,
        n_iter=10,
        cv=3,
        scoring='accuracy',
        verbose=0,
        n_jobs=-1
    )

    # Ajustar el modelo
    random_search.fit(X_train, y_train)

    # Mejor modelo
    mejor_modelo = random_search.best_estimator_

    # Guardar el modelo entrenado
    dump(mejor_modelo, 'modelo_entrenado.joblib')

    return mejor_modelo

# Verificar si el modelo ya está entrenado, si no, entrenarlo
def cargar_o_entrenar_modelo():
    if os.path.exists('modelo_entrenado.joblib'):
        return load('modelo_entrenado.joblib')
    else:
        return entrenar_modelo()

# Ruta para recomendar cultivo
@app.route('/recomendar', methods=['POST'])
def recomendar():
    data = request.get_json()

    # Obtener condiciones del usuario
    entrada_usuario = [
        data['N'], data['P'], data['K'], data['temperature'],
        data['humidity'], data['ph'], data['rainfall']
    ]

    # Cargar el modelo y hacer la recomendación
    modelo = cargar_o_entrenar_modelo()
    cultivo_recomendado = recomendar_cultivo(entrada_usuario)

    return jsonify({"cultivo_recomendado": cultivo_recomendado})

# Ruta para evaluar el modelo
@app.route('/evaluar_modelo', methods=['POST'])
def evaluar_modelo():
    modelo = cargar_o_entrenar_modelo()

    # Evaluación del modelo
    X_test = test_data[columnas_numericas]
    y_verdadero = test_data['label_codificada'].values
    y_predicho = modelo.predict(X_test)

    mae = mean_absolute_error(y_verdadero, y_predicho)
    mse = mean_squared_error(y_verdadero, y_predicho)
    rmse = np.sqrt(mse)

    reporte = classification_report(y_verdadero, y_predicho)

    return jsonify({
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "reporte": reporte
    })

# Ruta principal para servir el archivo HTML
@app.route('/')
def index():
    return render_template('index.html')  # Esto servirá el archivo index.html desde la carpeta templates

# Iniciar el servidor Flask
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
