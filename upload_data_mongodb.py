# ===========================================
# CREADOR DE BASE DE DATOS OPTIMIZADA CON ÍNDICES
# ===========================================

import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import json
from datetime import datetime

# --- CONFIGURACIÓN ---
MONGO_URI = "mongodb+srv://topicos_user:vt2GV4Q75YFJrVpR@puj-topicos-bd.m302xsg.mongodb.net/?retryWrites=true&w=majority&appName=puj-topicos-bd"
JSON_FILE = "restaurantes_bogota_completo.json"
DATABASE_NAME = "restaurantes_bogota_db"
COLLECTION_NAME = "bogota_data"

print("=" * 70)
print(" CREADOR DE BASE DE DATOS OPTIMIZADA")
print("=" * 70)

# --- 1. CONEXIÓN ---
print("\n[1/5]  Conectando a MongoDB Atlas...")

try:
    fixed_uri = MONGO_URI.replace('mongodb-srv://', 'mongodb+srv://')
    client = MongoClient(fixed_uri, server_api=ServerApi('1'))
    client.admin.command('ping')  # Verificar conexión
    print(" Conexión establecida")
    
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

except Exception as e:
    print(f" Error al conectar: {e}")
    exit()

# --- 2. BORRAR COLECCIÓN ANTERIOR ---
print("\n[2/5]   Borrando colección anterior (si existe)...")

try:
    if COLLECTION_NAME in db.list_collection_names():
        collection.drop()
        print(f" Colección '{COLLECTION_NAME}' eliminada")
    else:
        print(f"ℹ  La colección '{COLLECTION_NAME}' no existía")
    
    # Recrear colección vacía
    collection = db[COLLECTION_NAME]

except Exception as e:
    print(f" Error al borrar colección: {e}")
    exit()

# --- 3. CARGAR Y VALIDAR DATOS ---
print("\n[3/5]  Cargando datos desde JSON...")

try:
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f" {len(data)} documentos cargados desde '{JSON_FILE}'")
    
    # Validar estructura de ubicación
    docs_sin_ubicacion = 0
    docs_ubicacion_invalida = 0
    
    for doc in data:
        if 'ubicacion' not in doc:
            docs_sin_ubicacion += 1
        elif not isinstance(doc['ubicacion'], dict) or \
             'lat' not in doc['ubicacion'] or \
             'lng' not in doc['ubicacion']:
            docs_ubicacion_invalida += 1
    
    if docs_sin_ubicacion > 0:
        print(f"  {docs_sin_ubicacion} documentos sin campo 'ubicacion'")
    if docs_ubicacion_invalida > 0:
        print(f"  {docs_ubicacion_invalida} documentos con ubicación inválida")
    
    if docs_sin_ubicacion + docs_ubicacion_invalida > len(data) * 0.5:
        print(" MÁS DEL 50% de documentos tienen problemas de ubicación")
        print(" Sugerencia: Revisa el formato del JSON")
        exit()

except FileNotFoundError:
    print(f" Archivo '{JSON_FILE}' no encontrado")
    exit()
except json.JSONDecodeError:
    print(f" Error al parsear JSON: formato inválido")
    exit()
except Exception as e:
    print(f" Error inesperado: {e}")
    exit()

# --- 4. INSERTAR DATOS ---
print("\n[4/5] ⬆  Insertando datos en MongoDB...")

try:
    # Convertir ubicación a formato GeoJSON (requerido por 2dsphere)
    for doc in data:
        if 'ubicacion' in doc and isinstance(doc['ubicacion'], dict):
            lat = doc['ubicacion'].get('lat')
            lng = doc['ubicacion'].get('lng')
            
            if lat is not None and lng is not None:
                # Formato GeoJSON: { type: "Point", coordinates: [lng, lat] }
                doc['ubicacion_geo'] = {
                    "type": "Point",
                    "coordinates": [float(lng), float(lat)]
                }
                # Mantener ubicacion original para compatibilidad
    
    result = collection.insert_many(data)
    print(f" {len(result.inserted_ids)} documentos insertados")

except Exception as e:
    print(f" Error al insertar datos: {e}")
    client.close()
    exit()

# --- 5. CREAR ÍNDICES OPTIMIZADOS ---
print("\n[5/5]  Creando índices optimizados...")

indices_creados = []
indices_fallidos = []

# ========================================
# ÍNDICE 1: GEOESPACIAL (2dsphere) - CRÍTICO
# ========================================
try:
    collection.create_index([("ubicacion_geo", "2dsphere")], name="idx_ubicacion_geo")
    indices_creados.append(" idx_ubicacion_geo (2dsphere)")
    print("\n ÍNDICE GEOESPACIAL creado")
    print("   Campo: ubicacion_geo")
    print("   Tipo: 2dsphere (esfera geodésica)")
    print("   Beneficio: Búsquedas $near, $geoWithin en O(log n)")
    print("   Uso: Encontrar restaurantes cercanos a coordenadas")
except Exception as e:
    indices_fallidos.append(f" idx_ubicacion_geo: {e}")

# ========================================
# ÍNDICE 2: NOMBRE (B-tree) - IMPORTANTE
# ========================================
try:
    collection.create_index([("nombre", 1)], name="idx_nombre")
    indices_creados.append(" idx_nombre (B-tree)")
    print("\n ÍNDICE DE NOMBRE creado")
    print("   Campo: nombre")
    print("   Tipo: B-tree ascendente")
    print("   Beneficio: Búsquedas exactas y parciales rápidas")
    print("   Uso: Buscar restaurante por nombre específico")
except Exception as e:
    indices_fallidos.append(f" idx_nombre: {e}")

# ========================================
# ÍNDICE 3: RATING (B-tree) - ÚTIL
# ========================================
try:
    collection.create_index([("rating", -1)], name="idx_rating")
    indices_creados.append(" idx_rating (B-tree descendente)")
    print("\n ÍNDICE DE RATING creado")
    print("   Campo: rating")
    print("   Tipo: B-tree descendente (mayor a menor)")
    print("   Beneficio: Ordenamiento rápido por calificación")
    print("   Uso: Obtener top restaurantes mejor calificados")
except Exception as e:
    indices_fallidos.append(f" idx_rating: {e}")

# ========================================
# ÍNDICE 4: COMPUESTO UBICACION + RATING - AVANZADO
# ========================================
try:
    collection.create_index(
        [("ubicacion_geo", "2dsphere"), ("rating", -1)],
        name="idx_geo_rating"
    )
    indices_creados.append(" idx_geo_rating (compuesto)")
    print("\n ÍNDICE COMPUESTO creado")
    print("   Campos: ubicacion_geo + rating")
    print("   Beneficio: Búsquedas geográficas pre-ordenadas por rating")
    print("   Uso: Mejores restaurantes cercanos en una sola query")
except Exception as e:
    indices_fallidos.append(f" idx_geo_rating: {e}")


# --- RESUMEN FINAL ---
print("\n" + "=" * 70)
print(" RESUMEN DE CREACIÓN")
print("=" * 70)

print(f"\n ÍNDICES CREADOS ({len(indices_creados)}):")
for idx in indices_creados:
    print(f"   {idx}")

if indices_fallidos:
    print(f"\n ÍNDICES FALLIDOS ({len(indices_fallidos)}):")
    for idx in indices_fallidos:
        print(f"   {idx}")

# Verificación final
stats = db.command("collStats", COLLECTION_NAME)
print(f"\n Total de documentos: {stats.get('count', 0):,}")
print(f" Total de índices: {stats.get('nindexes', 0)}")
print(f" Tamaño de índices: {stats.get('totalIndexSize', 0) / 1024:.2f} KB")

print("\n" + "=" * 70)
print(" BASE DE DATOS OPTIMIZADA CREADA EXITOSAMENTE")
print("=" * 70)

# Cerrar conexión
client.close()
print("\n Conexión cerrada")

