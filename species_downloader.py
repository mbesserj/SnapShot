import requests
import json
import time
from pathlib import Path
from typing import Dict, List


class SpeciesDownloader:
    def __init__(self, log_callback=None, progress_callback=None):
        self.headers = {
            "User-Agent": "SnapShotBirdIdentifier/1.0 (Student Project)"
        }
        # Lista Negra Integrada (Mascotas y aves que no corresponden)
        self.blacklist = [
            "serinus serinus",
            "carduelis carduelis",
            "anas platyrhynchos",
            "columba livia domestica",
            "gallus gallus",
            "canary",
            "melopsittacus undulatus",
            "anser anser"
        ]

    def get_place_id(self, country_name):
        """Busca dinámicamente el ID numérico del país en iNaturalist."""
        url = "https://api.inaturalist.org/v1/places/autocomplete"
        try:
            r = requests.get(url, params={"q": country_name}, headers=self.headers, timeout=10)
            if r.ok:
                results = r.json().get("results", [])
                if results:
                    return results[0]["id"]
        except Exception as e:
            print(f"⚠️ Advertencia: No se pudo autodetectar el ID de {country_name}: {e}")
        return None

    def download_and_save(self, country, place_id=None, min_observations=50, filepath="bird_species.json"):
        # 1. Autodetectar ID
        detected_id = self.get_place_id(country)
        final_place_id = detected_id if detected_id else (place_id if place_id else 7170)

        print(f"📡 Conectando a iNaturalist para '{country}' (ID: {final_place_id})...")

        locale = "es-CL" if country.lower() == "chile" else "es"
        url = "https://api.inaturalist.org/v1/observations/species_counts"

        params = {
            "place_id": final_place_id,
            "taxon_id": 3,  # ID 3 = Clase Aves
            "verifiable": "true",
            "spam": "false",
            "locale": locale,
            "per_page": 500
        }

        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=20)

            if not response.ok:
                print(f"❌ Error API: {response.status_code} - {response.text}")
                return False

            raw_results = response.json().get("results", [])
            print(f"✅ Descarga exitosa: {len(raw_results)} registros crudos.")

            # 4. Procesamiento y Limpieza
            clean_species = []
            for item in raw_results:
                count = item.get("count", 0)
                taxon = item.get("taxon", {})

                # --- FILTROS ---
                if count < min_observations: continue  # Muy raras

                sc_name = taxon.get("name", "").lower()
                if sc_name in self.blacklist: continue  # Lista negra

                if taxon.get("rank") != "species": continue  # Solo especies puras

                # Construcción del objeto
                clean_species.append({
                    "scientific_name": taxon.get("name"),
                    "common_name": taxon.get("preferred_common_name", taxon.get("name", "")).title(),
                    "image_url": taxon.get("default_photo", {}).get("medium_url", ""),
                    "observation_count": count,
                    # --- LA CORRECCIÓN CLAVE ESTÁ AQUÍ ABAJO ---
                    "taxon_id": taxon.get("id")  # Antes decía "id", ahora dice "taxon_id"
                    # -------------------------------------------
                })

            # Ordenar por popularidad
            clean_species.sort(key=lambda x: x["observation_count"], reverse=True)

            # 5. Guardar JSON
            data = {
                "countries": {
                    country: {
                        "species": clean_species
                    }
                }
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

            print(f"💾 Guardado: {len(clean_species)} especies válidas en {filepath}")
            return True

        except Exception as e:
            print(f"❌ Error crítico en descarga: {e}")
            return False