import requests
import json
from pathlib import Path


class SpeciesDownloader:
    def __init__(self):
        # Cabeceras para evitar bloqueos (Error 422/403)
        self.headers = {
            "User-Agent": "SnapShotBirdIdentifier/1.0 (Student Project)"
        }

        # Lista Negra Integrada (Mascotas, Vagantes, Errores comunes)
        self.blacklist = [
            "serinus serinus",  # Verdecillo
            "carduelis carduelis",  # Jilguero Europeo
            "anas platyrhynchos",  # Pato doméstico
            "columba livia domestica",  # Palomas de plaza (variedades raras)
            "gallus gallus",  # Gallina
            "canary",  # Canarios
            "melopsittacus undulatus",  # Catita australiana
            "anser anser"  # Ganso doméstico
        ]

    def get_place_id(self, country_name):
        """Busca dinámicamente el ID numérico del país en iNaturalist."""
        url = "https://api.inaturalist.org/v1/places/autocomplete"
        try:
            r = requests.get(url, params={"q": country_name}, headers=self.headers, timeout=10)
            if r.ok:
                results = r.json().get("results", [])
                if results:
                    # Retornamos el primer resultado (suele ser el más relevante)
                    return results[0]["id"]
        except Exception as e:
            print(f"⚠️ Advertencia: No se pudo autodetectar el ID de {country_name}: {e}")
        return None

    def download_and_save(self, country, place_id=None, min_observations=50, filepath="bird_species.json"):
        """
        Descarga la lista de especies filtrada y la guarda en JSON.
        Ahora usa lógica robusta con taxon_id=3.
        """
        # 1. Autodetectar ID si no se provee o si el provisto falló antes
        detected_id = self.get_place_id(country)

        # Prioridad: ID detectado > ID manual > Fallback Chile (7170)
        final_place_id = detected_id if detected_id else (place_id if place_id else 7170)

        print(f"📡 Conectando a iNaturalist para '{country}' (ID: {final_place_id})...")

        # 2. Configurar idioma según país (Mapeo básico)
        locale = "es-CL" if country.lower() == "chile" else "es"

        # 3. URL y Parámetros (La configuración que SÍ funciona)
        url = "https://api.inaturalist.org/v1/observations/species_counts"
        params = {
            "place_id": final_place_id,
            "taxon_id": 3,  # ID 3 = Clase Aves (Evita error 422)
            "verifiable": "true",
            "spam": "false",
            "locale": locale,  # Nombres comunes locales
            "per_page": 500  # Traer las 500 más comunes
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
                    "id": taxon.get("id")
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