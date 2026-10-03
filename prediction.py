from google.colab import drive
drive.mount('/content/drive')

!pip install rasterio

import rasterio
import numpy as np
import os

# 1. Función para cargar y cortar en parches de 256x256
def procesar_imagen_tif(ruta_archivo, tamano=256):
    with rasterio.open(ruta_archivo) as src:
        img = src.read(1).astype('float32') # Float32 es obligatorio para redes neuronales
    
    alto, ancho = img.shape
    parches = []
    
    # Recorrer la imagen y extraer recortes exactos de 256x256
    for y in range(0, alto - tamano + 1, tamano):
        for x in range(0, ancho - tamano + 1, tamano):
            parches.append(img[y:y+tamano, x:x+tamano])
            
    # Añadir la dimensión del canal al final -> (N, 256, 256, 1)
    return np.expand_dims(np.array(parches), axis=-1)

# 2. Cargar todos los años
ruta_base = '/content/drive/MyDrive/Dataset_ConvLSTM/Mancha_Urbana_SJL_'
datos_anuales = []

for anio in range(2018, 2024): # 2018 a 2023
    ruta = f"{ruta_base}{anio}.tif"
    parches_anio = procesar_imagen_tif(ruta)
    datos_anuales.append(parches_anio)

# datos_anuales es una lista de 6 arrays. Cada array contiene los parches de ese año.


# Secuencia 1: Usamos 2018, 2019, 2020 para predecir 2021
X1 = np.stack((datos_anuales[0], datos_anuales[1], datos_anuales[2]), axis=1)
Y1 = datos_anuales[3]

# Secuencia 2: Usamos 2019, 2020, 2021 para predecir 2022
X2 = np.stack((datos_anuales[1], datos_anuales[2], datos_anuales[3]), axis=1)
Y2 = datos_anuales[4]

# Secuencia 3: Usamos 2020, 2021, 2022 para predecir 2023 (Para Validación)
X3 = np.stack((datos_anuales[2], datos_anuales[3], datos_anuales[4]), axis=1)
Y3 = datos_anuales[5]

# Unificar datos de entrenamiento (Secuencias 1 y 2)
X_train = np.concatenate((X1, X2), axis=0)
Y_train = np.concatenate((Y1, Y2), axis=0)

# Unificar datos de validación (Secuencia 3)
X_val = X3
Y_val = Y3

print("Forma de X_train:", X_train.shape) 
# Esperado: (N_total_parches, 3, 256, 256, 1)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Conv2D, TimeDistributed, MaxPooling2D, UpSampling2D, Input

# 1. Definir la arquitectura
modelo = Sequential([
    Input(shape=(3, 256, 256, 1)),
    
    # Módulo Codificador (Reduce la resolución para ahorrar memoria)
    TimeDistributed(Conv2D(32, (3,3), padding='same', activation='relu')),
    TimeDistributed(MaxPooling2D((2,2))), # Baja a 128x128
    
    # Módulo Predictor ConvLSTM (Extrae patrones espacio-temporales)
    ConvLSTM2D(filters=64, kernel_size=(3,3), padding='same', return_sequences=False),
    
    # Módulo Decodificador (Restaura la resolución original)
    UpSampling2D((2,2)), # Sube a 256x256
    Conv2D(32, (3,3), padding='same', activation='relu'),
    
    # Capa de Salida (Sigmoide escupe probabilidades de 0 a 1 para área urbana)
    Conv2D(filters=1, kernel_size=(1,1), padding='same', activation='sigmoid')
])

# 2. Compilar (Tasa de aprendizaje 10^-4 según la literatura base)
from tensorflow.keras.optimizers import Adam
modelo.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
modelo.summary()

# 3. Entrenar el modelo (Batch_size=1 es obligatorio)
historial = modelo.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    batch_size=1, 
    epochs=10
)


# Los últimos 3 años disponibles
X_futuro = np.stack((datos_anuales[3], datos_anuales[4], datos_anuales[5]), axis=1)

# Predecir 2024 (Devuelve probabilidades continuas)
prediccion_2024_prob = modelo.predict(X_futuro, batch_size=1)

# Aplicar umbral: Todo píxel > 50% de probabilidad se convierte en ciudad (1)
mapa_2024 = (prediccion_2024_prob > 0.5).astype(np.uint8)

print("Forma de la predicción 2024:", mapa_2024.shape)


import rasterio
import numpy as np

# 1. Extraer la metadata espacial (coordenadas, proyección, dimensiones originales)
# Usamos el mapa de 2023 como referencia para copiar su "perfil" geográfico
ruta_referencia = '/content/drive/MyDrive/Dataset_ConvLSTM/Mancha_Urbana_SJL_2023.tif'

with rasterio.open(ruta_referencia) as src:
    perfil = src.profile
    alto = src.height
    ancho = src.width

# 2. Crear un lienzo en blanco del tamaño original exacto del distrito
# Usamos np.uint8 porque nuestro mapa final es binario (solo 0s y 1s)
imagen_reconstruida = np.zeros((alto, ancho), dtype=np.uint8)

# 3. Ensamblar los parches predichos (Armar el rompecabezas)
tamano = 256
indice_parche = 0

# mapa_2024 tiene forma (N, 256, 256, 1). Extraemos los parches eliminando el canal extra:
parches_predichos = mapa_2024[:, :, :, 0]

# Usamos EXACTAMENTE el mismo bucle de la Fase 3 para colocar cada parche en su sitio
for y in range(0, alto - tamano + 1, tamano):
    for x in range(0, ancho - tamano + 1, tamano):
        # Insertar el parche en las coordenadas correspondientes del lienzo
        imagen_reconstruida[y:y+tamano, x:x+tamano] = parches_predichos[indice_parche]
        indice_parche += 1

# 4. Actualizar el perfil geográfico para asegurar compatibilidad
perfil.update(
    dtype=rasterio.uint8,
    count=1,              # 1 sola banda (blanco y negro)
    compress='lzw',       # Compresión ligera para no saturar tu Drive
    nodata=0              # Los bordes vacíos se leerán como 0 (sin mancha urbana)
)

# 5. Exportar el archivo GeoTIFF final a Google Drive
ruta_salida = '/content/drive/MyDrive/Dataset_ConvLSTM/Prediccion_Mancha_Urbana_SJL_2024.tif'

with rasterio.open(ruta_salida, 'w', **perfil) as dst:
    dst.write(imagen_reconstruida, 1)

print(" ¡Exportación completada!")
print(f"El mapa predictivo ha sido guardado en: {ruta_salida}")
