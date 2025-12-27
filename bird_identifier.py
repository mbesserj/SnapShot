# bird_identifier.py
"""
Módulo para identificar especies de aves chilenas usando el modelo YOLO entrenado.
Integrable con SnapShot AI.

Autor: SnapShot AI
Fecha: 2025
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class BirdDetection:
    """Resultado de detección de un ave."""
    species: str  # Nombre común
    scientific_name: str  # Nombre científico
    confidence: float  # Confianza (0-1)
    bbox: Tuple[int, int, int, int]  # Bounding box (x1, y1, x2, y2)
    class_id: int  # ID de clase


class ChileanBirdIdentifier:
    """
    Identificador de aves chilenas usando YOLO.
    """

    def __init__(self, model_path: str, class_mapping_path: Optional[str] = None):
        """
        Inicializa el identificador.

        Args:
            model_path: Ruta al modelo .pt entrenado
            class_mapping_path: Ruta al archivo class_mapping.json
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Instala ultralytics: pip install ultralytics")

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

        # Cargar modelo
        self.model = YOLO(str(self.model_path))

        # Cargar mapeo de clases
        self.class_mapping = {}
        self.species_info = {}

        if class_mapping_path:
            mapping_path = Path(class_mapping_path)
            if mapping_path.exists():
                with open(mapping_path, 'r') as f:
                    data = json.load(f)
                    self.class_mapping = {
                        int(v): k for k, v in data.get("class_to_id", {}).items()
                    }
                    self.species_info = data.get("species_info", {})

        # Si no hay mapeo, usar nombres del modelo
        if not self.class_mapping:
            self.class_mapping = self.model.names

    def identify(
            self,
            image_path: str,
            confidence_threshold: float = 0.5,
            max_detections: int = 10
    ) -> List[BirdDetection]:
        """
        Identifica aves en una imagen.

        Args:
            image_path: Ruta a la imagen
            confidence_threshold: Umbral mínimo de confianza
            max_detections: Máximo número de detecciones a retornar

        Returns:
            Lista de BirdDetection ordenada por confianza
        """
        # Ejecutar inferencia
        results = self.model(image_path, verbose=False)

        detections = []

        for result in results:
            boxes = result.boxes

            if boxes is None:
                continue

            for i in range(len(boxes)):
                conf = float(boxes.conf[i])

                if conf < confidence_threshold:
                    continue

                class_id = int(boxes.cls[i])
                species = self.class_mapping.get(class_id, f"unknown_{class_id}")

                # Obtener nombre científico
                scientific = species
                if species in self.species_info:
                    scientific = self.species_info[species].get("scientific_name", species)

                # Bounding box
                bbox = tuple(map(int, boxes.xyxy[i].tolist()))

                detections.append(BirdDetection(
                    species=species,
                    scientific_name=scientific,
                    confidence=conf,
                    bbox=bbox,
                    class_id=class_id
                ))

        # Ordenar por confianza y limitar
        detections.sort(key=lambda x: x.confidence, reverse=True)
        return detections[:max_detections]

    def identify_batch(
            self,
            image_paths: List[str],
            confidence_threshold: float = 0.5
    ) -> Dict[str, List[BirdDetection]]:
        """
        Identifica aves en múltiples imágenes.

        Args:
            image_paths: Lista de rutas a imágenes
            confidence_threshold: Umbral mínimo de confianza

        Returns:
            Diccionario {ruta_imagen: lista_detecciones}
        """
        results = {}

        for path in image_paths:
            results[path] = self.identify(path, confidence_threshold)

        return results

    def get_species_list(self) -> List[Dict[str, str]]:
        """
        Retorna la lista de especies que el modelo puede identificar.

        Returns:
            Lista de diccionarios con nombre común y científico
        """
        species_list = []

        for class_id, species_name in self.class_mapping.items():
            info = self.species_info.get(species_name, {})
            species_list.append({
                "id": class_id,
                "common_name": species_name,
                "scientific_name": info.get("scientific_name", species_name),
                "taxon_id": info.get("taxon_id")
            })

        return sorted(species_list, key=lambda x: x["common_name"])

    def draw_detections(
            self,
            image_path: str,
            output_path: str,
            confidence_threshold: float = 0.5
    ) -> str:
        """
        Dibuja las detecciones en la imagen y la guarda.

        Args:
            image_path: Ruta a la imagen original
            output_path: Ruta donde guardar la imagen anotada
            confidence_threshold: Umbral mínimo de confianza

        Returns:
            Ruta a la imagen guardada
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("Instala opencv: pip install opencv-python")

        # Cargar imagen
        img = cv2.imread(image_path)

        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")

        # Obtener detecciones
        detections = self.identify(image_path, confidence_threshold)

        # Colores por especie (generar automáticamente)
        import hashlib

        def get_color(species: str) -> Tuple[int, int, int]:
            hash_val = int(hashlib.md5(species.encode()).hexdigest()[:6], 16)
            return (
                (hash_val >> 16) & 255,
                (hash_val >> 8) & 255,
                hash_val & 255
            )

        # Dibujar cada detección
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = get_color(det.species)

            # Rectángulo
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Etiqueta
            label = f"{det.species} ({det.confidence:.2f})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            # Fondo de etiqueta
            cv2.rectangle(
                img,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )

            # Texto
            cv2.putText(
                img,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # Guardar
        cv2.imwrite(output_path, img)
        return output_path


# ============================================================================
# INTEGRACIÓN CON SNAPSHOT AI
# ============================================================================

class BirdIdentifierService:
    """
    Servicio de identificación de aves para integrar con SnapShot AI.
    Sigue el patrón de servicios de la aplicación.
    """

    _instance = None
    _identifier = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self, model_path: str, class_mapping_path: Optional[str] = None):
        """
        Inicializa el servicio con el modelo.

        Args:
            model_path: Ruta al modelo entrenado
            class_mapping_path: Ruta al mapeo de clases
        """
        self._identifier = ChileanBirdIdentifier(model_path, class_mapping_path)

    @property
    def is_initialized(self) -> bool:
        return self._identifier is not None

    def identify_bird(self, image_path: str, min_confidence: float = 0.5) -> Optional[Dict]:
        """
        Identifica el ave principal en una imagen.

        Args:
            image_path: Ruta a la imagen
            min_confidence: Confianza mínima

        Returns:
            Diccionario con información del ave o None
        """
        if not self.is_initialized:
            return None

        detections = self._identifier.identify(image_path, min_confidence, max_detections=1)

        if not detections:
            return None

        det = detections[0]
        return {
            "species": det.species,
            "scientific_name": det.scientific_name,
            "confidence": det.confidence,
            "bbox": det.bbox
        }

    def identify_all_birds(self, image_path: str, min_confidence: float = 0.3) -> List[Dict]:
        """
        Identifica todas las aves en una imagen.

        Args:
            image_path: Ruta a la imagen
            min_confidence: Confianza mínima

        Returns:
            Lista de diccionarios con información de cada ave
        """
        if not self.is_initialized:
            return []

        detections = self._identifier.identify(image_path, min_confidence)

        return [
            {
                "species": det.species,
                "scientific_name": det.scientific_name,
                "confidence": det.confidence,
                "bbox": det.bbox
            }
            for det in detections
        ]

    def get_available_species(self) -> List[Dict]:
        """
        Retorna la lista de especies disponibles.
        """
        if not self.is_initialized:
            return []

        return self._identifier.get_species_list()


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # Ejemplo de uso standalone
    import sys

    if len(sys.argv) < 3:
        print("Uso: python bird_identifier.py <modelo.pt> <imagen.jpg>")
        print("Ejemplo: python bird_identifier.py models/chilean_birds_best.pt foto.jpg")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    # Crear identificador
    identifier = ChileanBirdIdentifier(
        model_path=model_path,
        class_mapping_path="dataset_chilean_birds/class_mapping.json"
    )

    # Identificar
    print(f"\n🔍 Analizando: {image_path}")
    detections = identifier.identify(image_path)

    if not detections:
        print("❌ No se detectaron aves")
    else:
        print(f"\n✅ Se detectaron {len(detections)} ave(s):\n")
        for i, det in enumerate(detections, 1):
            print(f"  {i}. {det.species}")
            print(f"     Nombre científico: {det.scientific_name}")
            print(f"     Confianza: {det.confidence:.1%}")
            print()

    # Guardar imagen anotada
    output_path = image_path.replace(".jpg", "_detected.jpg")
    identifier.draw_detections(image_path, output_path)
    print(f"📷 Imagen guardada: {output_path}")