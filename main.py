# ===========================================
# BUSCADOR INTELIGENTE DE RESTAURANTES CON FAISS
# ===========================================

import os
import json
import pandas as pd
import requests
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from geopy.distance import geodesic
import streamlit as st
import folium
from streamlit_folium import st_folium
import openai
import numpy as np
import faiss
from tqdm import tqdm
from typing import List, Dict

# ======================
# CONFIGURACIÓN
# ======================
MONGO_URI = "mongodb+srv://topicos_user:vt2GV4Q75YFJrVpR@puj-topicos-bd.m302xsg.mongodb.net/?retryWrites=true&w=majority&appName=puj-topicos-bd"
DATABASE_NAME = "restaurantes_db"
COLLECTION_NAME = "chapinero_data"

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GOOGLE_KEY = st.secrets["GOOGLE_KEY"]
MONGO_URI = st.secrets["MONGO_URI"]

openai.api_key = OPENAI_API_KEY

# FAISS DB
DB_PATH = "./faiss_db"
INDEX_FILE = os.path.join(DB_PATH, "resenas.index")
META_FILE = os.path.join(DB_PATH, "metadata.json")

# ======================
# FUNCIONES AUXILIARES
# ======================

# MongoDB
def connect_mongo():
    try:
        client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
        return client[DATABASE_NAME][COLLECTION_NAME]
    except Exception as e:
        st.error(f"Error conectando a MongoDB: {e}")
        return None

def read_mongo_data(col):
    data = list(col.find({}))
    for o in data:
        o["_id"] = str(o["_id"])
    df = pd.DataFrame(data)
    
    # ELIMINAR DUPLICADOS
    # Prioriza el registro con más información (menos valores nulos)
    df = df.sort_values(by=df.columns.tolist(), 
                        key=lambda x: x.isna().sum(), 
                        ascending=True)
    
    # Elimina duplicados basándote en el nombre del restaurante
    nombre_cols = ['nombre', 'Nombre', 'name']
    for col_name in nombre_cols:
        if col_name in df.columns:
            df = df.drop_duplicates(subset=[col_name], keep='first')
            break
    
    return df

# Google Maps
def get_coordinates(address):
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    r = requests.get(base_url, params={"address": address + ", Bogotá, Colombia", "key": GOOGLE_KEY}).json()
    if r["status"] == "OK":
        loc = r["results"][0]["geometry"]["location"]
        return loc["lat"], loc["lng"]
    return None, None

def calculate_distance(a, b):
    return geodesic(a, b).meters

def extract_coordinates(row):
    try:
        if 'ubicacion' in row and isinstance(row['ubicacion'], dict):
            lat = row['ubicacion'].get('lat')
            lng = row['ubicacion'].get('lng')
            if lat is not None and lng is not None:
                return float(lat), float(lng)
        if 'lat' in row and 'lng' in row:
            return float(row['lat']), float(row['lng'])
        if 'Latitud' in row and 'Longitud' in row:
            return float(row['Latitud']), float(row['Longitud'])
        return None, None
    except:
        return None, None

def filter_nearby(lat, lng, df, max_m=1000):
    distances = []
    valid_coords = []
    for idx, row in df.iterrows():
        rest_lat, rest_lng = extract_coordinates(row)
        if rest_lat is not None and rest_lng is not None:
            dist = calculate_distance((lat, lng), (rest_lat, rest_lng))
            distances.append(dist)
            valid_coords.append(True)
        else:
            distances.append(float('inf'))
            valid_coords.append(False)
    df["dist"] = distances
    df["valid_coords"] = valid_coords
    nearby = df[(df["dist"] <= max_m) & (df["valid_coords"] == True)].copy()
    return nearby.sort_values("dist")

def get_restaurant_name(r):
    return r.get('nombre') or r.get('Nombre') or r.get('name') or 'Sin nombre'

def get_restaurant_rating(r):
    return r.get('rating') or r.get('Rating') or 'N/A'

# ======================
# FUNCIONES FAISS
# ======================
def obtener_embeddings_openai(textos: List[str]) -> np.ndarray:
    response = openai.embeddings.create(model="text-embedding-ada-002", input=textos)
    return np.array([d.embedding for d in response.data], dtype="float32")

def cargar_faiss():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        st.error(" No se encontró la base FAISS. Crea la base primero con tu script FAISS.")
        return None, None
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

def buscar_resenas(index, metadata, pregunta: str, n_resultados=5, restaurante_especifico=None):
    q_emb = obtener_embeddings_openai([pregunta])
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, n_resultados)
    resultados = []
    for idx, dist in zip(I[0], D[0]):
        if idx >= len(metadata):
            continue
        meta = metadata[idx]
        if restaurante_especifico and meta["restaurante"] != restaurante_especifico:
            continue
        resultados.append({
            "texto": meta.get("texto", ""),
            "restaurante": meta["restaurante"],
            "fecha": meta["fecha"],
            "score": float(dist)
        })
    return resultados

def generar_resumen_faiss(restaurantes: List[Dict], index, metadata):
    texto_resenas = ""
    for r in restaurantes:
        nombre = get_restaurant_name(r)
        res = buscar_resenas(index, metadata, nombre, n_resultados=1)
        if res:
            texto_resenas += f"- {nombre} (distancia {round(r.get('dist',0))}m, rating {get_restaurant_rating(r)})\n"
            texto_resenas += f"  Reseña: {res[0]['texto'][:200]}...\n"
    if not texto_resenas:
        return "No se encontraron reseñas relevantes en FAISS."

    prompt = f"""
Analiza estos restaurantes cercanos y proporciona un resumen claro y útil:

{texto_resenas}

Devuelve:
1. Resumen general de tipos de comida disponibles
2. Top 3 restaurantes recomendados con sus puntos fuertes
3. Comentarios destacados de clientes

Mantén el resumen conciso y útil (máximo 300 palabras).
"""
    try:
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error al generar resumen con IA: {str(e)}\n\nRestaurantes:\n{texto_resenas}"

# ======================
# STREAMLIT
# ======================
st.set_page_config(page_title="Buscador de Restaurantes", layout="wide")
st.title(" Buscador Inteligente de Restaurantes")

# Conexión Mongo
col = connect_mongo()
if col is None:
    st.error("No se pudo conectar a la base de datos")
    st.stop()
df = read_mongo_data(col)
if df.empty:
    st.error("No se encontraron datos de restaurantes")
    st.stop()

# Cargar FAISS
index, metadata = cargar_faiss()
if index is None:
    st.stop()

# Input usuario
addr = st.text_input(" Ingresa tu ubicación:", placeholder="Ej: Calle 72 Carrera 5")
if addr:
    with st.spinner("Buscando tu ubicación..."):
        user_lat, user_lng = get_coordinates(addr)
    if user_lat and user_lng:
        st.success(f" Ubicación encontrada")
        with st.spinner("Buscando restaurantes cercanos..."):
            nearby = filter_nearby(user_lat, user_lng, df, max_m=1000)
        if nearby.empty:
            st.warning(" No se encontraron restaurantes cercanos")
        else:
            st.info(f" Se encontraron {len(nearby)} restaurantes en un radio de 1 km")

            # Mapa
            m = folium.Map(location=[user_lat, user_lng], zoom_start=14)
            folium.Marker([user_lat,user_lng], tooltip=" Tu ubicación",
                          icon=folium.Icon(color="red", icon="home", prefix='fa')).add_to(m)
            for idx,row in nearby.iterrows():
                lat,lng = extract_coordinates(row)
                if lat and lng:
                    folium.Marker([lat,lng],
                                  tooltip=f"{get_restaurant_name(row)}\nRating: {get_restaurant_rating(row)}\nDistancia: {round(row.get('dist',0))}m",
                                  popup=f"<b>{get_restaurant_name(row)}</b><br>Rating: {get_restaurant_rating(row)}<br>Distancia: {round(row.get('dist',0))}m",
                                  icon=folium.Icon(color="blue", icon="cutlery", prefix='fa')).add_to(m)
            st.subheader(" Mapa de Restaurantes")
            st_folium(m, width=900, height=500)

            # Resumen FAISS
            st.subheader(" Resumen Inteligente con FAISS")
            with st.spinner("Generando resumen con IA y FAISS..."):
                summary = generar_resumen_faiss(nearby.to_dict(orient="records"), index, metadata)
            st.write(summary)

            # Tabla
            st.subheader("Lista de Restaurantes Cercanos")
            display_data = []
            for idx,row in nearby.iterrows():
                display_data.append({
                    'Nombre': get_restaurant_name(row),
                    'Rating': get_restaurant_rating(row),
                    'Distancia (m)': round(row.get('dist',0)),
                    'Dirección': row.get('direccion') or row.get('Dirección') or row.get('address') or 'N/A'
                })
            st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)
    else:
        st.error(" No se pudo encontrar la ubicación. Intenta con una dirección más específica.")





