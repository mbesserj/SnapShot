#!/usr/bin/env python3
"""
GENERADOR DE CATÁLOGO DE AVES DE CHILE (STANDALONE / INDEPENDIENTE)
Este script conecta directo a la API de iNaturalist sin depender de otros módulos.
"""
import requests
import json
import time

# --- CONFIGURACIÓN ---
CHILE_PLACE_ID = 7170
MIN_OBSERVATIONS = 150  # Umbral alto para evitar rarezas

# Lista de especies a ignorar (mascotas, erróneas, introducidas no deseadas)
BLACKLIST = [
    "serinus serinus", "carduelis carduelis", "anas platyrhynchos",
    "columba livia domestica", "gallus gallus", "canary",
    "melopsittacus undulatus"  # Periquito australiano
]


def fetch_chilean_birds():
    print("=" * 60)
    print(f"📡 CONECTANDO A INATURALIST (ID: {CHILE_PLACE_ID})")
    print("=" * 60)

    # URL oficial de la API v1 de iNaturalist
    url = "https://api.inaturalist.org/v1/observations/species_counts"

    # Parámetros para obtener SOLO Aves de Chile
    params = {
        "place_id": CHILE_PLACE_ID,
        "iconic_taxa": "Aves",
        "verifiable": "true",
        "spam": "false",
        "locale": "es-CL",  # Pedimos nombres en Español de Chile
        "preferred_place_id": CHILE_PLACE_ID,
        "per_page": 500  # Traer las 500 más comunes (suficiente para Chile)
    }

    try:
        print("⏳ Realizando petición a la API...")
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()  # Lanza error si hay fallo HTTP (404, 500, etc)

        data = response.json()
        results = data.get("results", [])

        print(f"✅ Conexión exitosa. Se recibieron {len(results)} especies crudas.")
        return results

    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR CRÍTICO DE CONEXIÓN: {e}")
        return []
    except Exception as e:
        print(f"❌ ERROR INESPERADO: {e}")
        return []


def process_and_save(raw_results):
    print("\n⚙️ Procesando y Filtrando lista...")

    clean_species = []

    for item in raw_results:
        count = item.get("count", 0)
        taxon = item.get("taxon", {})

        scientific_name = taxon.get("name", "").lower()
        common_name = taxon.get("preferred_common_name", taxon.get("name", "")).title()

        # --- FILTROS ---

        # 1. Filtro de Cantidad (Popularidad)
        if count < MIN_OBSERVATIONS:
            continue  # Ignoramos aves raras

        # 2. Filtro de Lista Negra
        if scientific_name in BLACKLIST:
            print(f"   🚫 Ignorando (Blacklist): {common_name}")
            continue

        # 3. Filtro de Nivel (Solo Especies, no Géneros ni Familias)
        if taxon.get("rank") != "species":
            continue

        # Crear objeto limpio
        species_obj = {
            "scientific_name": taxon.get("name"),
            "common_name": common_name,
            "image_url": taxon.get("default_photo", {}).get("medium_url", ""),
            "observation_count": count,
            "id": taxon.get("id")
        }

        clean_species.append(species_obj)

    # Ordenar por cantidad de observaciones (los más comunes primero)
    clean_species.sort(key=lambda x: x["observation_count"], reverse=True)

    # Estructura final compatible con tu App
    final_json = {
        "countries": {
            "Chile": {
                "species": clean_species
            }
        }
    }

    # Guardar
    filename = "bird_species.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"🎉 ÉXITO: Se guardaron {len(clean_species)} especies en '{filename}'")
    print("=" * 60)

    print("Top 10 Especies detectadas:")
    for i, sp in enumerate(clean_species[:10], 1):
        print(f"  {i}. {sp['common_name']} ({sp['observation_count']} obs)")


def main():
    raw_data = fetch_chilean_birds()
    if raw_data:
        process_and_save(raw_data)
    else:
        print("\n❌ No se pudo generar el archivo debido a errores de conexión.")


if __name__ == "__main__":
    main()