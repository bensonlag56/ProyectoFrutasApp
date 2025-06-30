from flask import Flask, render_template, request, url_for
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Crear directorio de uploads si no existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cargar modelo entrenado
modelo = load_model("modelo/modelo_fold_1.keras")
clases = ["🍏 Buen estado", "💩 Mal estado"]

@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None
    imagen_url = None

    if request.method == 'POST':
        archivo = request.files['imagen']
        if archivo and archivo.filename != '':
            # Guardar archivo con nombre seguro
            filename = secure_filename(archivo.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            archivo.save(filepath)

            # Generar URL para mostrar la imagen
            imagen_url = url_for('static', filename=f'uploads/{filename}')

            try:
                # Procesar imagen
                imagen = cv2.imread(filepath)
                if imagen is not None:
                    imagen_redim = cv2.resize(imagen, (128, 128))
                    imagen_norm = imagen_redim / 255.0
                    imagen_input = np.expand_dims(imagen_norm, axis=0)

                    # Predicción
                    prediccion = modelo.predict(imagen_input)[0]
                    indice_predicho = np.argmax(prediccion)
                    confianza = prediccion[indice_predicho] * 100
                    resultado = f"{clases[indice_predicho]} ({confianza:.1f}%)"
                else:
                    resultado = "❌ Error al procesar la imagen"
            
            except Exception as e:
                resultado = f"❌ Error: {str(e)}"

    return render_template("index.html", resultado=resultado, imagen=imagen_url)

if __name__ == '__main__':
    app.run(debug=True)