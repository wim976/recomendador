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

# Configuración de CORS
CORS(app, resources={r"/recomendar": {"origins": "https://dashboard.render.com"}})


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
# Función para recomendar cultivo (versión mejorada sin duplicados)
def recomendar_cultivo(entrada_usuario):
    entrada_normalizada = scaler.transform([entrada_usuario])
    similitudes = cosine_similarity(entrada_normalizada, perfiles_cultivos)
    
    # Diccionario para evitar duplicados (cultivo: máxima similitud encontrada)
    cultivos_unicos = {}
    for indice, similitud in enumerate(similitudes[0]):
        cultivo = label_encoder.inverse_transform([cultivos_nombres.iloc[indice]])[0]
        if cultivo not in cultivos_unicos or similitud > cultivos_unicos[cultivo]:
            cultivos_unicos[cultivo] = similitud
    
    # Ordenar por similitud (mayor a menor) y seleccionar top 5
    top_cultivos = sorted(cultivos_unicos.items(), 
                         key=lambda item: item[1], 
                         reverse=True)[:5]
    
    # Formatear resultado
    return [{"cultivo": cultivo, "similitud": float(similitud)} 
            for cultivo, similitud in top_cultivos]

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
    cultivos_recomendados = recomendar_cultivo(entrada_usuario)

    return jsonify({"cultivos_recomendados": cultivos_recomendados})

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

# Ruta principal para servir el archivo HTML grupo6
@app.route('/grup6')
def grupo6():
    return render_template('grup6.html')

# Iniciar el servidor Flask
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Usa el puerto que Render le asigna
    app.run(debug=True, host='0.0.0.0', port=port)
