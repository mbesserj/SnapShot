# services/metadata.py
"""
Servicio de extracción de metadatos EXIF de archivos RAW.
Proporciona información técnica de fotografía y fechas de captura.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple

import exifread

logger = logging.getLogger(__name__)


class MetadataService:
    """
    Servicio para extraer información EXIF de archivos RAW.

    Extrae:
    - Fecha y hora de captura
    - Configuración de cámara (apertura, velocidad, ISO)
    """

    # Fallbacks para cuando no hay EXIF
    DEFAULT_FSTOP = "f/?"
    DEFAULT_EXPOSURE = "?s"
    DEFAULT_ISO = "Auto"

    def extract(self, file_path: str) -> Tuple[datetime, str]:
        """
        Extrae fecha de captura e información técnica de un archivo RAW.

        Args:
            file_path: Ruta al archivo RAW

        Returns:
            Tuple de (fecha_captura, info_tecnica)
            donde info_tecnica tiene formato: "f/4.0 | 1/500s | ISO 800"

        Raises:
            FileNotFoundError: Si el archivo no existe
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

        try:
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)

            # Extraer fecha
            capture_date = self._extract_date(file_path, tags)

            # Extraer información técnica
            f_stop = self._extract_fstop(tags)
            exposure = self._extract_exposure(tags)
            iso = self._extract_iso(tags)

            # Construir string de información técnica
            tech_info = f"{f_stop} | {exposure} | ISO {iso}"

            logger.debug(f"{Path(file_path).name}: {tech_info}")

            return capture_date, tech_info

        except Exception as e:
            logger.error(f"Error extrayendo metadata de {Path(file_path).name}: {e}")
            # Retornar valores por defecto en caso de error
            return datetime.now(), "Error Metadata"

    def _extract_date(self, file_path: str, tags: dict) -> datetime:
        """
        Extrae la fecha de captura del EXIF o usa fecha de modificación.

        Args:
            file_path: Ruta al archivo (para fallback)
            tags: Diccionario de tags EXIF

        Returns:
            Fecha de captura como datetime
        """
        # Intentar obtener fecha original de captura
        date_str = str(tags.get('EXIF DateTimeOriginal', ''))

        if date_str and date_str != '':
            try:
                return datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
            except ValueError as e:
                logger.warning(f"Formato de fecha inválido '{date_str}': {e}")

        # Fallback: usar fecha de modificación del archivo
        try:
            mtime = os.path.getmtime(file_path)
            return datetime.fromtimestamp(mtime)
        except Exception as e:
            logger.warning(f"Error obteniendo fecha de archivo: {e}")
            return datetime.now()

    def _extract_fstop(self, tags: dict) -> str:
        """
        Extrae la apertura (f-stop) del EXIF.

        Args:
            tags: Diccionario de tags EXIF

        Returns:
            String con formato "f/X.X" o "f/?" si no está disponible
        """
        f_tag = tags.get('EXIF FNumber')

        if not f_tag:
            return self.DEFAULT_FSTOP

        try:
            # El valor viene como fracción (ej: 40/10 = 4.0)
            val = float(f_tag.values[0])

            # Formatear con un decimal si es necesario
            if val == int(val):
                return f"f/{int(val)}"
            else:
                return f"f/{val:.1f}"

        except (ValueError, IndexError, AttributeError) as e:
            logger.debug(f"Error parseando f-stop: {e}")
            return str(f_tag) if f_tag else self.DEFAULT_FSTOP

    def _extract_exposure(self, tags: dict) -> str:
        """
        Extrae el tiempo de exposición del EXIF.

        Args:
            tags: Diccionario de tags EXIF

        Returns:
            String con el tiempo de exposición (ej: "1/500s") o "?s" si no disponible
        """
        exposure_tag = tags.get('EXIF ExposureTime')

        if not exposure_tag:
            return self.DEFAULT_EXPOSURE

        try:
            exposure_str = str(exposure_tag)

            # Limpiar y formatear si es necesario
            if '/' in exposure_str:
                # Ya viene en formato fracción (1/500)
                return f"{exposure_str}s"
            else:
                # Puede venir como decimal
                try:
                    val = float(exposure_str)
                    if val >= 1:
                        return f"{val:.1f}s"
                    else:
                        # Convertir a fracción aproximada
                        return f"1/{int(1 / val)}s"
                except ValueError:
                    return f"{exposure_str}s"

        except Exception as e:
            logger.debug(f"Error parseando exposición: {e}")
            return self.DEFAULT_EXPOSURE

    def _extract_iso(self, tags: dict) -> str:
        """
        Extrae el valor ISO del EXIF.

        Maneja casos especiales de diferentes marcas de cámaras.

        Args:
            tags: Diccionario de tags EXIF

        Returns:
            String con el valor ISO o "Auto" si no disponible
        """
        # Intentar EXIF ISOSpeedRatings primero
        iso_tag = tags.get('EXIF ISOSpeedRatings')

        if iso_tag:
            iso_str = str(iso_tag)
            # Limpiar valores extraños
            if '?' not in iso_str and 'ISO' not in iso_str.upper():
                return self._clean_iso_value(iso_str)

        # Fallback: Image ISOSpeedRatings (común en Nikon)
        iso_tag = tags.get('Image ISOSpeedRatings')
        if iso_tag:
            return self._clean_iso_value(str(iso_tag))

        # Si no hay ISO, probablemente es modo Auto
        return self.DEFAULT_ISO

    def _clean_iso_value(self, iso_str: str) -> str:
        """
        Limpia el valor ISO de caracteres innecesarios.

        Args:
            iso_str: String con el valor ISO raw

        Returns:
            Valor ISO limpio
        """
        # Remover espacios y caracteres no numéricos al inicio/final
        iso_str = iso_str.strip()

        # Intentar extraer solo el número
        try:
            # Si contiene solo dígitos, retornar directamente
            if iso_str.isdigit():
                return iso_str

            # Intentar parsear como número
            iso_val = int(float(iso_str))
            return str(iso_val)

        except ValueError:
            # Si no se puede parsear, retornar el string original limpio
            return iso_str.replace('[', '').replace(']', '')

    def extract_camera_model(self, file_path: str) -> str:
        """
        Extrae el modelo de cámara del EXIF.

        Args:
            file_path: Ruta al archivo RAW

        Returns:
            Nombre del modelo de cámara o "Unknown" si no disponible
        """
        try:
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)

            # Intentar obtener modelo
            model = tags.get('Image Model', tags.get('EXIF Model'))

            if model:
                return str(model).strip()

            return "Unknown"

        except Exception as e:
            logger.warning(f"Error extrayendo modelo de cámara: {e}")
            return "Unknown"

    def extract_lens_model(self, file_path: str) -> str:
        """
        Extrae el modelo de lente del EXIF.

        Args:
            file_path: Ruta al archivo RAW

        Returns:
            Nombre del modelo de lente o "Unknown" si no disponible
        """
        try:
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)

            # Buscar información de lente
            lens = tags.get('EXIF LensModel', tags.get('Image LensModel'))

            if lens:
                return str(lens).strip()

            return "Unknown"

        except Exception as e:
            logger.warning(f"Error extrayendo modelo de lente: {e}")
            return "Unknown"