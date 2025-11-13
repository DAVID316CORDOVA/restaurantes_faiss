 ┌─────────────────────────────────────────────────────────────┐
    │              STREAMLIT CLOUD WEB APPLICATION                │
    │                                                             │
    │  ┌──────────────┐         ┌───────────────────────────┐   │
    │  │   Text Input │         │   Folium Map Component     │   │
    │  │   (Address)  │────────▶│   - User marker (red)      │   │
    │  │              │         │   - Restaurant markers     │   │
    │  └──────┬───────┘         │     (blue, clickable)      │   │
    │         │                 └───────────────────────────┘   │
    │         │                                                  │
    │         │                 ┌───────────────────────────┐   │
    │         │                 │  GPT-Generated Summary     │   │
    │         │                 │  (Contextual AI text)      │   │
    │         │                 └───────────────────────────┘   │
    │         │                                                  │
    │         │                 ┌───────────────────────────┐   │
    │         │                 │   Results DataFrame        │   │
    │         │                 │   (Name|Rating|Distance)   │   │
    │         │                 └───────────────────────────┘   │
    └─────────┼──────────────────────────────────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │  Google Maps API    │
    │   (Geocoding)       │────┐
    │  Address → (lat,lng)│    │
    └─────────────────────┘    │
                               │
                               ▼
              ┌────────────────────────────────────┐
              │      MongoDB Atlas (M0/M10)        │
              │  Collection: restaurantes          │
              │                                    │
              │  Query: db.restaurantes.find({     │
              │    ubicacion: {                    │
              │      $near: {                      │
              │        $geometry: {                │
              │          type: "Point",            │
              │          coordinates: [lng, lat]   │
              │        },                          │
              │        $maxDistance: 1000  // m    │
              │      }                             │
              │    }                               │
              │  }).limit(10)                      │
              │                                    │
              │  Result: [                         │
              │    {nombre, rating, direccion,     │
              │     ubicacion, distancia, ...}     │
              │  ]                                 │
              └───────────┬────────────────────────┘
                          │
                          │ For each restaurant
                          ▼
              ┌────────────────────────────────────┐
              │      FAISS Index (in-memory)       │
              │  File: resenas.index (vectors)     │
              │  File: metadata.json (text+meta)   │
              │                                    │
              │  1. Generate query embedding:      │
              │     embedding = openai.embed(      │
              │       model="text-embedding-ada",  │
              │       input=restaurant_name        │
              │     )                              │
              │                                    │
              │  2. Normalize vector:              │
              │     faiss.normalize_L2(embedding)  │
              │                                    │
              │  3. Search index:                  │
              │     D, I = index.search(           │
              │       embedding, k=1               │
              │     )                              │
              │                                    │
              │  4. Retrieve metadata:             │
              │     review_text = metadata[I[0]]   │
              │                                    │
              │  Result: Most relevant review text │
              └───────────┬────────────────────────┘
                          │
                          │ Consolidate all reviews
                          ▼
              ┌────────────────────────────────────┐
              │         OpenAI API                 │
              │  Model: gpt-3.5-turbo              │
              │                                    │
              │  Prompt:                           │
              │  "Analiza estos restaurantes:      │
              │   - Trattoria (450m, 4.7★)         │
              │     Review: 'Pasta increíble...'   │
              │   - El Azteca (780m, 4.5★)         │
              │     Review: 'Tacos auténticos...'  │
              │                                    │
              │   Devuelve:                        │
              │   1. Resumen de tipos de comida    │
              │   2. Top 3 recomendados            │
              │   3. Comentarios destacados"       │
              │                                    │
              │  Response:                         │
              │  "En la zona encontrarás...        │
              │   Top 3: 1) Trattoria..."          │
              └────────────────────────────────────┘
                          │
                          │ Return to Streamlit
                          ▼
                   [Display Results]
