# models.py
"""
Modelos de datos para el sistema SnapShot AI.
Define las estructuras de datos principales utilizadas en el análisis de imágenes.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class Photo:
    """
    Representa una única imagen RAW y sus propiedades de análisis.

    Attributes:
        name: Nombre del archivo
        path: Ruta completa al archivo
        date: Fecha de captura (EXIF o fecha de modificación)
        tech_info: Información técnica (apertura, velocidad, ISO)
        score: Puntuación de calidad asignada por IA (default: 0.0)
        details: Detalles del análisis (tipo de enfoque, métricas, etc.)
    """
    name: str
    path: str
    date: datetime
    tech_info: str
    score: float = 0.0
    details: str = ""

    def __post_init__(self):
        """Validación de datos después de la inicialización."""
        if not self.name:
            raise ValueError("El nombre del archivo no puede estar vacío")

        if not Path(self.path).exists():
            raise FileNotFoundError(f"El archivo no existe: {self.path}")

        if self.score < 0:
            self.score = 0.0

    @property
    def file_size_mb(self) -> float:
        """Retorna el tamaño del archivo en MB."""
        return Path(self.path).stat().st_size / (1024 * 1024)

    @property
    def extension(self) -> str:
        """Retorna la extensión del archivo en mayúsculas."""
        return Path(self.path).suffix.upper()

    def __str__(self) -> str:
        """Representación legible de la foto."""
        return f"{self.name} - Score: {self.score:.2f} - {self.tech_info}"

    def __repr__(self) -> str:
        """Representación técnica de la foto."""
        return f"Photo(name='{self.name}', score={self.score:.2f}, date={self.date})"


@dataclass
class BurstGroup:
    """
    Representa un grupo de fotos tomadas en ráfaga.

    Attributes:
        burst_id: Identificador único del grupo
        photos: Lista de fotos en el grupo
        time_threshold: Umbral de tiempo (segundos) para agrupar fotos
    """
    burst_id: int
    photos: list[Photo] = field(default_factory=list)
    time_threshold: float = 20.0

    def add_photo(self, photo: Photo) -> None:
        """Añade una foto al grupo."""
        self.photos.append(photo)

    @property
    def size(self) -> int:
        """Número de fotos en el grupo."""
        return len(self.photos)

    @property
    def top_photo(self) -> Optional[Photo]:
        """Retorna la foto con mayor score."""
        return max(self.photos, key=lambda p: p.score) if self.photos else None

    def get_top_n(self, n: int = 3) -> list[Photo]:
        """
        Retorna las N mejores fotos del grupo ordenadas por score.

        Args:
            n: Número de fotos a retornar

        Returns:
            Lista de las N mejores fotos
        """
        sorted_photos = sorted(self.photos, key=lambda p: p.score, reverse=True)
        return sorted_photos[:min(n, len(sorted_photos))]

    @property
    def date_range(self) -> tuple[datetime, datetime]:
        """Retorna el rango de fechas (primera, última) del grupo."""
        if not self.photos:
            now = datetime.now()
            return (now, now)

        dates = [p.date for p in self.photos]
        return (min(dates), max(dates))

    @property
    def duration_seconds(self) -> float:
        """Duración total de la ráfaga en segundos."""
        start, end = self.date_range
        return (end - start).total_seconds()

    def __str__(self) -> str:
        """Representación legible del grupo."""
        if not self.photos:
            return f"Burst #{self.burst_id}: vacío"

        top = self.top_photo
        return (f"Burst #{self.burst_id}: {self.size} fotos, "
                f"Mejor: {top.name} ({top.score:.0f})")

    def __repr__(self) -> str:
        """Representación técnica del grupo."""
        return f"BurstGroup(id={self.burst_id}, size={self.size})"


@dataclass
class ProcessingStats:
    """
    Estadísticas del procesamiento de imágenes.

    Attributes:
        total_files: Total de archivos procesados
        total_bursts: Total de ráfagas detectadas
        processing_time: Tiempo total de procesamiento (segundos)
        ai_enabled: Si se usó análisis IA
        files_per_second: Velocidad de procesamiento
    """
    total_files: int = 0
    total_bursts: int = 0
    processing_time: float = 0.0
    ai_enabled: bool = True
    errors: int = 0

    @property
    def files_per_second(self) -> float:
        """Calcula la velocidad de procesamiento."""
        return self.total_files / self.processing_time if self.processing_time > 0 else 0.0

    @property
    def avg_burst_size(self) -> float:
        """Promedio de fotos por ráfaga."""
        return self.total_files / self.total_bursts if self.total_bursts > 0 else 0.0

    def __str__(self) -> str:
        """Resumen de estadísticas."""
        return (
            f"Procesadas: {self.total_files} fotos en {self.total_bursts} ráfagas\n"
            f"Tiempo: {self.processing_time:.2f}s ({self.files_per_second:.2f} fps)\n"
            f"IA: {'Activada' if self.ai_enabled else 'Desactivada'}\n"
            f"Errores: {self.errors}"
        )