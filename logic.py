# logic.py
"""
Lógica principal de procesamiento de ráfagas fotográficas.
Gestiona el flujo completo desde la detección hasta la organización de archivos.
v2.0 - Con identificación de especies de aves chilenas.
"""

import os
import shutil
import platform
import subprocess
import time
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Callable, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import flet as ft

from models import Photo, BurstGroup, ProcessingStats
from services.metadata import MetadataService
from services.ai import AIService

logger = logging.getLogger(__name__)


class BurstProcessor:
    """
    Procesador principal de ráfagas fotográficas.

    Workflow:
    1. Escanea carpeta y filtra archivos RAW
    2. Extrae metadatos (fecha, configuración técnica)
    3. Agrupa fotos en ráfagas basándose en timestamps
    4. Analiza calidad con IA (opcional) o por peso de archivo
    5. Identifica especies de aves (si está habilitado)
    6. Selecciona las mejores fotos de cada ráfaga
    7. Organiza archivos en carpetas estructuradas
    """

    # Extensiones RAW soportadas
    SUPPORTED_EXTENSIONS = ('.nef', '.cr3', '.cr2', '.arw', '.dng', '.orf', '.raf')

    # Configuración de agrupamiento
    BURST_TIME_THRESHOLD = 20.0  # segundos

    # Nombres de carpetas de salida
    OUTPUT_FOLDER = "Seleccionadas_AI"
    BURST_FOLDER_PREFIX = "Burst_"
    DEBUG_FOLDER = "Debug_View"

    # Configuración del modelo de aves
    BIRD_MODEL_PATH = "models/chilean_birds_best.pt"
    BIRD_CLASS_MAPPING_PATH = "models/class_mapping.json"

    def __init__(
            self,
            logger_callback: Callable[[str, str], None],
            progress_updater: Callable[[float], None],
            enable_bird_id: bool = True
    ):
        """
        Inicializa el procesador de ráfagas.

        Args:
            logger_callback: Función para mostrar logs en UI (msg, color)
            progress_updater: Función para actualizar barra de progreso (0.0-1.0)
            enable_bird_id: Si habilitar identificación de especies de aves
        """
        self.meta_service = MetadataService()
        self.log = logger_callback
        self.update_progress = progress_updater

        # Inicializar servicio de IA
        self._init_ai_service(enable_bird_id)

        # Optimización para M1/M2: usar todos los cores menos 1 para la UI
        cpu_cores = os.cpu_count() or 4
        self.max_workers = max(1, cpu_cores - 1)

        # Estadísticas de especies detectadas
        self.species_stats: Dict[str, int] = defaultdict(int)

        logger.info(f"BurstProcessor inicializado con {self.max_workers} workers")

    def _init_ai_service(self, enable_bird_id: bool) -> None:
        """
        Inicializa el servicio de IA con o sin identificación de aves.

        Args:
            enable_bird_id: Si habilitar identificación de especies
        """
        bird_model = None
        bird_mapping = None

        if enable_bird_id:
            # Verificar si existe el modelo de aves
            if os.path.exists(self.BIRD_MODEL_PATH):
                bird_model = self.BIRD_MODEL_PATH
                if os.path.exists(self.BIRD_CLASS_MAPPING_PATH):
                    bird_mapping = self.BIRD_CLASS_MAPPING_PATH
                logger.info("Modelo de identificación de aves encontrado")
            else:
                logger.info("Modelo de aves no encontrado, continuando sin identificación de especies")

        self.ai_service = AIService(
            model_name="yolov8l.pt",
            bird_model_path=bird_model,
            bird_class_mapping_path=bird_mapping
        )

    def process_folder(
            self,
            folder_path: str,
            use_ai: bool,
            only_copy: bool,
            debug_mode: bool
    ) -> None:
        """
        Procesa una carpeta completa de archivos RAW.

        Args:
            folder_path: Ruta a la carpeta con archivos RAW
            use_ai: Si True, usa análisis IA; si False, ordena por tamaño
            only_copy: Si True, solo copia; si False, mueve archivos originales
            debug_mode: Si True, genera imágenes de debug visual
        """
        stats = ProcessingStats(ai_enabled=use_ai)
        start_time = time.time()

        # Resetear estadísticas de especies
        self.species_stats.clear()

        # Ajustar workers según modo
        current_workers = self.max_workers if use_ai else (self.max_workers * 2)

        # Mostrar info de identificación de aves
        bird_id_status = "ON" if self.ai_service.bird_identification_enabled else "OFF"

        self.log(
            f"🚀 MODO TURBO ACTIVADO (Hilos: {current_workers}, IA: {'ON' if use_ai else 'OFF'}, ID Aves: {bird_id_status})",
            ft.Colors.BLUE
        )

        try:
            # Fase 1: Escaneo de archivos
            photos = self._scan_folder(folder_path, stats)

            if not photos:
                self.log("❌ No se encontraron archivos RAW válidos.", ft.Colors.RED)
                return

            # Fase 2: Agrupamiento en ráfagas
            bursts = self._group_into_bursts(photos, stats)

            self.log(
                f"✅ Detectadas {len(bursts)} ráfagas. Iniciando análisis...",
                ft.Colors.GREEN
            )

            # Fase 3: Análisis de calidad (incluye identificación de especies)
            self._analyze_photos(bursts, use_ai, debug_mode, current_workers, stats)

            # Fase 4: Organización de archivos
            self._organize_files(folder_path, bursts, only_copy, stats)

            # Fase 5: Mostrar resumen de especies detectadas
            self._show_species_summary()

            # Fase 6: Finalización y estadísticas
            stats.processing_time = time.time() - start_time
            self._show_final_stats(stats, folder_path)

        except Exception as e:
            logger.exception("Error crítico en process_folder")
            self.log(f"❌ ERROR CRÍTICO: {str(e)}", ft.Colors.RED)
            stats.errors += 1
        finally:
            self.update_progress(1.0)

    def _scan_folder(self, folder_path: str, stats: ProcessingStats) -> List[Photo]:
        """
        Escanea la carpeta y extrae metadatos de todos los archivos RAW.
        """
        try:
            all_files = os.listdir(folder_path)
        except Exception as e:
            self.log(f"❌ Error accediendo carpeta: {e}", ft.Colors.RED)
            return []

        raw_files = [
            f for f in all_files
            if f.lower().endswith(self.SUPPORTED_EXTENSIONS)
        ]

        if not raw_files:
            return []

        total_files = len(raw_files)
        stats.total_files = total_files

        self.log(f"📸 Encontrados {total_files} archivos RAW", ft.Colors.BLUE)
        self.log("⏳ Extrayendo metadatos...", ft.Colors.BLUE)

        photos: List[Photo] = []

        for idx, filename in enumerate(raw_files):
            progress = (idx / total_files) * 0.10
            self.update_progress(progress)

            full_path = os.path.join(folder_path, filename)

            try:
                date_taken, tech_info = self.meta_service.extract(full_path)
                photo = Photo(filename, full_path, date_taken, tech_info)
                photos.append(photo)

            except Exception as e:
                logger.warning(f"Error procesando {filename}: {e}")
                stats.errors += 1
                continue

        photos.sort(key=lambda x: x.date)

        self.log(f"✅ Metadatos extraídos de {len(photos)} fotos", ft.Colors.GREEN)

        return photos

    def _group_into_bursts(
            self,
            photos: List[Photo],
            stats: ProcessingStats
    ) -> Dict[int, BurstGroup]:
        """
        Agrupa fotos en ráfagas basándose en el tiempo entre capturas.
        """
        if not photos:
            return {}

        bursts: Dict[int, BurstGroup] = {}
        burst_id = 1
        current_burst = BurstGroup(burst_id=burst_id, photos=[photos[0]])

        for i in range(1, len(photos)):
            prev_photo = photos[i - 1]
            curr_photo = photos[i]

            time_diff = (curr_photo.date - prev_photo.date).total_seconds()

            if time_diff <= self.BURST_TIME_THRESHOLD:
                current_burst.add_photo(curr_photo)
            else:
                bursts[burst_id] = current_burst
                burst_id += 1
                current_burst = BurstGroup(burst_id=burst_id, photos=[curr_photo])

        if current_burst.photos:
            bursts[burst_id] = current_burst

        stats.total_bursts = len(bursts)

        return bursts

    def _analyze_photos(
            self,
            bursts: Dict[int, BurstGroup],
            use_ai: bool,
            debug_mode: bool,
            num_workers: int,
            stats: ProcessingStats
    ) -> None:
        """
        Analiza la calidad de todas las fotos en paralelo.
        """
        all_photos: List[Photo] = []
        for burst in bursts.values():
            all_photos.extend(burst.photos)

        total_photos = len(all_photos)

        if use_ai:
            self.log(
                f"🤖 Analizando {total_photos} fotos con IA ({num_workers} hilos)...",
                ft.Colors.BLUE
            )
            self._analyze_with_ai(all_photos, debug_mode, num_workers, stats)
        else:
            self.log(
                f"⚖️ Ordenando por peso de archivo ({total_photos} fotos)...",
                ft.Colors.BLUE
            )
            self._analyze_by_size(all_photos)

        self.update_progress(0.90)

    def _analyze_with_ai(
            self,
            photos: List[Photo],
            debug_mode: bool,
            num_workers: int,
            stats: ProcessingStats
    ) -> None:
        """
        Analiza fotos con IA en paralelo usando ThreadPoolExecutor.
        Incluye identificación de especies.
        """
        total = len(photos)
        completed = 0

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_photo = {
                executor.submit(self.ai_service.analyze, p.path, debug_mode): p
                for p in photos
            }

            for future in as_completed(future_to_photo):
                photo = future_to_photo[future]

                try:
                    score, details = future.result()
                    photo.score = score
                    photo.details = details

                    # Extraer especie de los detalles si está presente
                    self._extract_species_from_details(details)

                except Exception as exc:
                    logger.error(f"Error analizando {photo.name}: {exc}")
                    photo.score = 0
                    photo.details = f"Error: {exc}"
                    stats.errors += 1

                completed += 1

                if completed % 10 == 0 or completed == total:
                    progress = 0.10 + ((completed / total) * 0.80)
                    self.update_progress(progress)

    def _extract_species_from_details(self, details: str) -> None:
        """
        Extrae el nombre de la especie de los detalles del análisis.

        Args:
            details: String con detalles del análisis
        """
        # Buscar patrón de especie: "nombre_especie (XX%)"
        if "🦅" in details:
            try:
                # Extraer la parte antes del paréntesis
                parts = details.split("|")[0].strip()
                if "(" in parts:
                    species = parts.split("(")[0].strip()
                    if species and species not in ["Sujeto", "Centro", "Error"]:
                        self.species_stats[species] += 1
            except Exception:
                pass

    def _analyze_by_size(self, photos: List[Photo]) -> None:
        """
        Analiza fotos usando el tamaño de archivo como métrica de calidad.
        """
        for photo in photos:
            try:
                photo.score = os.path.getsize(photo.path)
                photo.details = "Ordenado por peso"
            except Exception as e:
                logger.warning(f"Error obteniendo tamaño de {photo.name}: {e}")
                photo.score = 0
                photo.details = "Error"

    def _organize_files(
            self,
            folder_path: str,
            bursts: Dict[int, BurstGroup],
            only_copy: bool,
            stats: ProcessingStats
    ) -> None:
        """
        Organiza archivos: copia los mejores y opcionalmente mueve originales.
        """
        self.log("📁 Organizando archivos...", ft.Colors.BLUE)

        dest_path = os.path.join(folder_path, self.OUTPUT_FOLDER)
        os.makedirs(dest_path, exist_ok=True)

        for burst_id, burst in bursts.items():
            self._process_burst(folder_path, dest_path, burst, only_copy, stats)

    def _process_burst(
            self,
            folder_path: str,
            dest_path: str,
            burst: BurstGroup,
            only_copy: bool,
            stats: ProcessingStats
    ) -> None:
        """
        Procesa una ráfaga individual: copia las mejores y organiza.
        """
        top_photos = burst.get_top_n(3)

        if not top_photos:
            logger.warning(f"Ráfaga {burst.burst_id} sin fotos válidas")
            return

        tech_info = top_photos[0].tech_info
        winner = top_photos[0]

        # Mostrar info de especie si está disponible
        species_info = ""
        if "🦅" in winner.details:
            # Extraer nombre de especie
            try:
                parts = winner.details.split("|")[0].strip()
                if "(" in parts:
                    species_info = f" [{parts.split('(')[0].strip()}]"
            except Exception:
                pass

        winner_info = f"{winner.name} ({int(winner.score)}){species_info}"

        self.log(
            f"Ráfaga #{burst.burst_id} [{tech_info}] → 🏆 {winner_info}",
            ft.Colors.BLACK
        )

        for rank, photo in enumerate(top_photos, start=1):
            new_name = f"G{burst.burst_id:02d}_TOP{rank}_{photo.name}"
            dest_file = os.path.join(dest_path, new_name)

            try:
                shutil.copy2(photo.path, dest_file)
            except Exception as e:
                logger.error(f"Error copiando {photo.name}: {e}")
                stats.errors += 1

        if not only_copy:
            self._move_burst_to_subfolder(folder_path, burst, stats)

    def _move_burst_to_subfolder(
            self,
            folder_path: str,
            burst: BurstGroup,
            stats: ProcessingStats
    ) -> None:
        """
        Mueve todas las fotos de una ráfaga a una subcarpeta.
        """
        burst_folder = os.path.join(
            folder_path,
            f"{self.BURST_FOLDER_PREFIX}{burst.burst_id:03d}"
        )
        os.makedirs(burst_folder, exist_ok=True)

        for photo in burst.photos:
            try:
                dest = os.path.join(burst_folder, photo.name)
                shutil.move(photo.path, dest)
            except Exception as e:
                logger.warning(f"Error moviendo {photo.name}: {e}")
                stats.errors += 1

    def _show_species_summary(self) -> None:
        """
        Muestra un resumen de las especies de aves detectadas.
        """
        if not self.species_stats:
            return

        self.log("-" * 40, ft.Colors.GREY)
        self.log("🐦 ESPECIES DETECTADAS:", ft.Colors.GREEN)

        # Ordenar por cantidad de detecciones
        sorted_species = sorted(
            self.species_stats.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for species, count in sorted_species:
            self.log(f"   • {species}: {count} foto(s)", ft.Colors.BLACK)

    def _show_final_stats(self, stats: ProcessingStats, folder_path: str) -> None:
        """
        Muestra estadísticas finales y abre la carpeta de resultados.
        """
        self.log("-" * 40, ft.Colors.GREY)
        self.log(
            f"⏱️ TIEMPO: {stats.processing_time:.2f}s | "
            f"VELOCIDAD: {stats.files_per_second:.2f} fps",
            ft.Colors.PURPLE
        )

        if stats.errors > 0:
            self.log(f"⚠️ Errores encontrados: {stats.errors}", ft.Colors.ORANGE)

        self.log("🎉 PROCESO TERMINADO", ft.Colors.PURPLE)

        logger.info(f"Estadísticas finales:\n{stats}")

        dest_path = os.path.join(folder_path, self.OUTPUT_FOLDER)
        self._open_folder(dest_path)

    def _open_folder(self, path: str) -> None:
        """
        Abre una carpeta en el explorador del sistema operativo.
        """
        try:
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            logger.warning(f"No se pudo abrir la carpeta: {e}")