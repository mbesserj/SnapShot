# services/ai.py
"""
Servicio de análisis de imágenes con YOLO y OpenCV.
Motor IA v6.0 - Con identificación de especies de aves chilenas.
"""

import os
import sys
import contextlib
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging

import rawpy
import numpy as np
import cv2
import torch
from ultralytics import YOLO

# Configuración de logging
logger = logging.getLogger(__name__)


def resource_path(relative_path: str) -> str:
    """
    Obtiene la ruta absoluta al recurso, compatible con PyInstaller.
    """
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class AIService:
    """
    Servicio de análisis de calidad de imágenes usando YOLO para detección
    de sujetos (principalmente aves) y OpenCV para análisis de nitidez.

    Incluye identificación de especies de aves chilenas.
    """

    # Clases COCO relevantes: bird, cat, dog, horse, sheep, cow, bear
    TARGET_CLASSES = [14, 15, 16, 17, 18, 19, 21]
    BIRD_CLASS_ID = 14  # ID de "bird" en COCO

    # Configuración de umbrales
    SUBJECT_CONFIDENCE_THRESHOLD = 0.15
    IOU_THRESHOLD = 0.45
    YOLO_IMAGE_SIZE = 1280

    # Pesos para scoring
    SHARPNESS_WEIGHT_SUBJECT = 1.5
    SHARPNESS_WEIGHT_CENTER = 0.8
    SIZE_WEIGHT = 5.0

    # Umbrales de exposición
    HIGHLIGHT_THRESHOLD = 0.05
    HIGHLIGHT_PENALTY = 500
    LOW_BRIGHTNESS_THRESHOLD = 15
    LOW_BRIGHTNESS_PENALTY = 2000

    def __init__(
            self,
            model_name: str = "yolov8l.pt",
            bird_model_path: Optional[str] = None,
            bird_class_mapping_path: Optional[str] = None
    ):
        """
        Inicializa el servicio de IA.

        Args:
            model_name: Nombre del modelo YOLO para detección general
            bird_model_path: Ruta al modelo de identificación de aves chilenas
            bird_class_mapping_path: Ruta al mapeo de clases de aves
        """
        self.device = self._setup_device()
        self.model = self._load_model(model_name)
        self.target_classes = self.TARGET_CLASSES

        # Inicializar identificador de aves (opcional)
        self.bird_identifier = None
        self.bird_identification_enabled = False

        if bird_model_path:
            self._init_bird_identifier(bird_model_path, bird_class_mapping_path)

        logger.info(f"AIService inicializado en dispositivo: {self.device}")

    def _init_bird_identifier(
            self,
            model_path: str,
            class_mapping_path: Optional[str] = None
    ) -> None:
        """
        Inicializa el identificador de aves chilenas.

        Args:
            model_path: Ruta al modelo entrenado
            class_mapping_path: Ruta al archivo de mapeo de clases
        """
        try:
            from bird_identifier import ChileanBirdIdentifier

            full_model_path = resource_path(model_path)
            full_mapping_path = resource_path(class_mapping_path) if class_mapping_path else None

            if os.path.exists(full_model_path):
                self.bird_identifier = ChileanBirdIdentifier(
                    model_path=full_model_path,
                    class_mapping_path=full_mapping_path
                )
                self.bird_identification_enabled = True
                logger.info("✅ Identificador de aves chilenas cargado")
            else:
                logger.warning(f"Modelo de aves no encontrado: {full_model_path}")

        except ImportError as e:
            logger.warning(f"No se pudo cargar bird_identifier: {e}")
        except Exception as e:
            logger.error(f"Error inicializando identificador de aves: {e}")

    def enable_bird_identification(
            self,
            model_path: str,
            class_mapping_path: Optional[str] = None
    ) -> bool:
        """
        Habilita la identificación de especies de aves.

        Args:
            model_path: Ruta al modelo de aves
            class_mapping_path: Ruta al mapeo de clases

        Returns:
            True si se habilitó correctamente
        """
        self._init_bird_identifier(model_path, class_mapping_path)
        return self.bird_identification_enabled

    def _setup_device(self) -> str:
        """Configura el dispositivo de cómputo óptimo."""
        if torch.backends.mps.is_available():
            device = 'mps'
            logger.info("🚀 ACELERACIÓN ACTIVADA: Usando GPU Apple Silicon (Metal)")
        else:
            device = 'cpu'
            logger.info("🐢 MODO CPU: Configurando threads óptimos")
            torch.set_num_threads(1)
        return device

    def _load_model(self, model_name: str) -> YOLO:
        """Carga el modelo YOLO desde el sistema de archivos."""
        model_path = resource_path(model_name)

        if not os.path.exists(model_path):
            error_msg = f"Modelo no encontrado: {model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            logger.info(f"Cargando modelo desde: {model_path}")
            model = YOLO(model_path)
            logger.info("✅ Modelo YOLO cargado exitosamente")
            return model
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise RuntimeError(f"No se pudo cargar el modelo YOLO: {e}")

    def analyze(self, file_path: str, debug: bool = False) -> Tuple[float, str]:
        """
        Analiza una imagen RAW y retorna su score de calidad.
        Incluye identificación de especie si está habilitada.

        Args:
            file_path: Ruta al archivo RAW
            debug: Si True, genera imágenes de debug con visualización

        Returns:
            Tuple de (score, detalles)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

        try:
            # 1. Extraer thumbnail del RAW
            img_color, img_gray = self._extract_thumbnail(file_path)

            if img_color is None or img_gray is None:
                return 0, "Error: No se pudo extraer thumbnail"

            # 2. Preparar imagen de debug si es necesario
            debug_img = img_color.copy() if debug else None

            # 3. Detectar sujetos con YOLO
            subject_found, best_box, max_conf, is_bird = self._detect_subjects(img_color)

            # 4. Identificar especie si es un ave
            species_info = None
            if is_bird and self.bird_identification_enabled:
                species_info = self._identify_bird_species(img_color)

            # 5. Analizar nitidez en región de interés
            sharpness, desc = self._analyze_sharpness(
                img_gray, subject_found, best_box, max_conf, debug_img
            )

            # 6. Añadir info de especie a la descripción
            if species_info:
                desc = f"{species_info['species']} ({species_info['confidence']:.0%}) 🦅"

            # 7. Analizar exposición
            exposure_penalty = self._analyze_exposure(img_gray)

            # 8. Calcular score final
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            final_score = sharpness + (size_mb * self.SIZE_WEIGHT) - exposure_penalty

            # 9. Guardar debug si está activado
            if debug and debug_img is not None:
                self._save_debug_image(
                    file_path, debug_img, sharpness, desc, img_gray, species_info
                )

            # 10. Generar detalles
            mean_brightness = np.mean(img_gray)
            details = f"{desc} | Foco:{int(sharpness)} | Brillo:{int(mean_brightness)}"

            logger.debug(f"{Path(file_path).name}: Score={final_score:.0f}, {details}")

            return final_score, details

        except Exception as e:
            error_msg = f"Error en análisis: {str(e)}"
            logger.error(f"{Path(file_path).name}: {error_msg}")
            return 0, error_msg

    def _extract_thumbnail(self, file_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extrae el thumbnail del archivo RAW."""
        try:
            with rawpy.imread(file_path) as raw:
                thumb = raw.extract_thumb()

            if thumb.format != rawpy.ThumbFormat.JPEG:
                logger.warning(f"Formato de thumbnail no soportado: {thumb.format}")
                return None, None

            nparr = np.frombuffer(thumb.data, np.uint8)
            img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img_color is None:
                logger.warning("Error decodificando thumbnail")
                return None, None

            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            return img_color, img_gray

        except Exception as e:
            logger.error(f"Error extrayendo thumbnail: {e}")
            return None, None

    def _detect_subjects(
            self,
            img_color: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray], float, bool]:
        """
        Detecta sujetos de interés en la imagen usando YOLO.

        Returns:
            Tuple de (sujeto_encontrado, mejor_bbox, confianza_maxima, es_ave)
        """
        try:
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
                results = self.model(
                    img_color,
                    verbose=False,
                    imgsz=self.YOLO_IMAGE_SIZE,
                    conf=self.SUBJECT_CONFIDENCE_THRESHOLD,
                    iou=self.IOU_THRESHOLD,
                    device=self.device
                )

            subject_found = False
            best_box = None
            max_conf = 0.0
            is_bird = False

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    if cls_id in self.target_classes and conf > max_conf:
                        subject_found = True
                        max_conf = conf
                        best_box = box.xyxy[0].cpu().numpy().astype(int)
                        is_bird = (cls_id == self.BIRD_CLASS_ID)

            return subject_found, best_box, max_conf, is_bird

        except Exception as e:
            logger.error(f"Error en detección YOLO: {e}")
            return False, None, 0.0, False

    def _identify_bird_species(self, img_color: np.ndarray) -> Optional[Dict]:
        """
        Identifica la especie de ave en la imagen.

        Args:
            img_color: Imagen en formato BGR

        Returns:
            Diccionario con species, scientific_name, confidence o None
        """
        if not self.bird_identifier:
            return None

        try:
            # Guardar imagen temporal para el identificador
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cv2.imwrite(tmp.name, img_color)
                tmp_path = tmp.name

            # Identificar especie
            detections = self.bird_identifier.identify(
                tmp_path,
                confidence_threshold=0.3,
                max_detections=1
            )

            # Limpiar archivo temporal
            os.unlink(tmp_path)

            if detections:
                det = detections[0]
                return {
                    "species": det.species,
                    "scientific_name": det.scientific_name,
                    "confidence": det.confidence
                }

        except Exception as e:
            logger.warning(f"Error identificando especie: {e}")

        return None

    def _analyze_sharpness(
            self,
            img_gray: np.ndarray,
            subject_found: bool,
            best_box: Optional[np.ndarray],
            confidence: float,
            debug_img: Optional[np.ndarray] = None
    ) -> Tuple[float, str]:
        """Analiza la nitidez de la imagen en la región de interés."""
        h, w = img_gray.shape

        if subject_found and best_box is not None:
            x1, y1, x2, y2 = best_box

            margin_x = int((x2 - x1) * 0.1)
            margin_y = int((y2 - y1) * 0.1)
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(w, x2 + margin_x)
            y2 = min(h, y2 + margin_y)

            if debug_img is not None:
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label = f"Sujeto: {int(confidence * 100)}%"
                cv2.putText(
                    debug_img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
                )

            crop = img_gray[y1:y2, x1:x2]
            if crop.size > 0:
                sharpness = cv2.Laplacian(crop, cv2.CV_64F).var() * self.SHARPNESS_WEIGHT_SUBJECT
                desc = "Sujeto 🦅"
            else:
                sharpness = 0
                desc = "Error ROI"
        else:
            cx1, cy1 = int(w * 0.25), int(h * 0.25)
            cx2, cy2 = int(w * 0.75), int(h * 0.75)

            if debug_img is not None:
                cv2.rectangle(debug_img, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)

            center_crop = img_gray[cy1:cy2, cx1:cx2]
            sharpness = cv2.Laplacian(center_crop, cv2.CV_64F).var() * self.SHARPNESS_WEIGHT_CENTER
            desc = "Centro"

        return sharpness, desc

    def _analyze_exposure(self, img_gray: np.ndarray) -> float:
        """Analiza la exposición y retorna penalización si hay problemas."""
        penalty = 0.0

        hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        highlights = np.sum(hist[250:]) / img_gray.size

        if highlights > self.HIGHLIGHT_THRESHOLD:
            penalty += self.HIGHLIGHT_PENALTY

        mean_brightness = np.mean(img_gray)
        if mean_brightness < self.LOW_BRIGHTNESS_THRESHOLD:
            penalty += self.LOW_BRIGHTNESS_PENALTY

        return penalty

    def _save_debug_image(
            self,
            file_path: str,
            debug_img: np.ndarray,
            sharpness: float,
            desc: str,
            img_gray: np.ndarray,
            species_info: Optional[Dict] = None
    ) -> None:
        """Guarda imagen de debug con información visual del análisis."""
        try:
            parent_dir = os.path.dirname(file_path)
            debug_dir = os.path.join(parent_dir, "Debug_View")
            os.makedirs(debug_dir, exist_ok=True)

            h, w = img_gray.shape

            # Info básica
            info_text = f"Score: {int(sharpness)} | {desc}"
            cv2.putText(
                debug_img, info_text, (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2
            )

            # Info de especie si está disponible
            if species_info:
                species_text = f"{species_info['species']} ({species_info['scientific_name']})"
                cv2.putText(
                    debug_img, species_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )

                conf_text = f"Confianza: {species_info['confidence']:.1%}"
                cv2.putText(
                    debug_img, conf_text, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )

            file_name = os.path.basename(file_path) + ".jpg"
            output_path = os.path.join(debug_dir, file_name)
            cv2.imwrite(output_path, debug_img)

            logger.debug(f"Debug guardado: {output_path}")

        except Exception as e:
            logger.warning(f"Error guardando debug: {e}")

    def get_bird_species_list(self) -> list:
        """
        Retorna la lista de especies de aves que el modelo puede identificar.

        Returns:
            Lista de diccionarios con información de cada especie
        """
        if not self.bird_identifier:
            return []
        return self.bird_identifier.get_species_list()