# species_manager.py
"""
Gestor de especies de aves.
Lee, escribe y modifica el archivo bird_species.json
"""
import json
from pathlib import Path
from typing import Dict, List, Optional


class SpeciesManager:
    """Gestor de especies desde archivo JSON."""

    DEFAULT_FILE = "bird_species.json"

    def __init__(self, filepath: str = None):
        self.filepath = Path(filepath) if filepath else Path(self.DEFAULT_FILE)
        self.data = self._load()

    def _load(self) -> Dict:
        """Carga el archivo JSON."""
        if not self.filepath.exists():
            return {"metadata": {}, "countries": {}}

        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error cargando {self.filepath}: {e}")
            return {"metadata": {}, "countries": {}}

    def save(self):
        """Guarda el archivo JSON."""
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)

    def reload(self):
        """Recarga el archivo desde disco."""
        self.data = self._load()

    def get_countries(self) -> List[str]:
        countries = list(self.data.get("countries", {}).keys())
        return sorted(countries) if countries else ["Chile"]

    def get_species(self, country: str) -> List[Dict]:
        if country in self.data.get("countries", {}):
            return self.data["countries"][country].get("species", [])
        return []

    def get_country_place_id(self, country: str) -> int:
        if country in self.data.get("countries", {}):
            return self.data["countries"][country].get("place_id", 0)
        return 7170 if country == "Chile" else 0

    # --- NUEVAS FUNCIONES PARA EDICIÓN MANUAL ---

    def delete_species(self, country: str, scientific_name: str):
        """Elimina una especie de la lista por nombre científico."""
        if country not in self.data["countries"]: return

        original_list = self.data["countries"][country]["species"]
        # Filtramos para quitar la que coincida
        new_list = [sp for sp in original_list if sp["scientific_name"].lower() != scientific_name.lower()]

        self.data["countries"][country]["species"] = new_list
        self.save()  # Guardamos cambios en disco

    def add_species(self, country: str, common_name: str, scientific_name: str):
        """Agrega una especie manualmente."""
        if country not in self.data["countries"]:
            # Si el país no existe, lo creamos
            self.data["countries"][country] = {"species": []}

        species_list = self.data["countries"][country]["species"]

        # Evitar duplicados
        for sp in species_list:
            if sp["scientific_name"].lower() == scientific_name.lower():
                return False  # Ya existe

        # Crear nueva entrada (usamos ID 0 porque es manual)
        new_bird = {
            "scientific_name": scientific_name,
            "common_name": common_name.title(),
            "image_url": "",
            "observation_count": 0,
            "taxon_id": 0  # 0 indica manual
        }

        # Insertar al principio de la lista
        species_list.insert(0, new_bird)
        self.save()
        return True


# =========================================================================
# FUNCIONES GLOBALES
# =========================================================================

_manager: Optional[SpeciesManager] = None


def get_species_manager() -> SpeciesManager:
    global _manager
    if _manager is None:
        _manager = SpeciesManager()
    return _manager


def reload_species():
    global _manager
    if _manager:
        _manager.reload()
    else:
        _manager = SpeciesManager()