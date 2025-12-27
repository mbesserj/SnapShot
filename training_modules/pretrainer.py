import os
import requests
import tarfile
import shutil
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
from config_manager import load_config


class AnatomyPretrainer:
    def __init__(self, log_callback, dash_callback=None):
        self.log = log_callback
        self.update_dash = dash_callback if dash_callback else lambda x: None

        self.config = load_config()
        self.url = self.config.get("cub_200_url", "https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz")

        # --- LÓGICA DE NOMBRE DE MODELO (YOLO11 Ready) ---
        raw_variant = self.config.get("yolo_model_variant", "yolov8n")
        # Si es config antigua ("n", "m"), lo convertimos a yolov8
        if len(raw_variant) == 1:
            self.model_name = f"yolov8{raw_variant}"
        else:
            self.model_name = raw_variant
        # -------------------------------------------------

        self.base_dir = Path.cwd() / "datasets" / "CUB_200"
        self.raw_path = self.base_dir / "CUB_200_2011"
        self.yolo_dir = self.base_dir / "yolo_format"
        self.models_dir = Path.cwd() / "models"
        self.stop_requested = False

    def request_stop(self):
        self.stop_requested = True

    def run(self) -> str:
        try:
            if not self._prepare_dataset(): return ""
            return self._train_anatomy_model()
        except Exception as e:
            self.log(f"❌ Error Fase 1: {e}", "red");
            return ""

    def _prepare_dataset(self):
        # 1. Descarga y Extracción
        if not (self.raw_path / "images").exists():
            if not self._download_extract(): return False

        # 2. Conversión a YOLO
        if not (self.yolo_dir / "data.yaml").exists():
            self.log("🛠️ Convirtiendo anotaciones a formato YOLO...", "blue")
            self._convert_labels()
            self._create_yaml()
        return True

    def _download_extract(self):
        self.base_dir.mkdir(parents=True, exist_ok=True)
        archive = self.base_dir / "cub.tgz"

        if not (archive.exists() and archive.stat().st_size > 0):
            try:
                self.log(f"📥 Descargando CUB-200...", "blue")
                with requests.get(self.url, stream=True) as r:
                    r.raise_for_status()
                    with open(archive, 'wb') as f: shutil.copyfileobj(r.raw, f)
            except Exception as e:
                self.log(f"❌ Error descarga: {e}", "red");
                return False

        try:
            self.log("📦 Extrayendo archivos...", "blue")
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(path=self.base_dir)
            return True
        except Exception as e:
            self.log(f"❌ Error extracción: {e}", "red");
            return False

    def _convert_labels(self):
        # (Código de conversión idéntico al anterior para ahorrar espacio,
        #  la lógica de conversión de etiquetas no cambia por la versión de YOLO)
        for s in ["train", "val"]:
            (self.yolo_dir / "images" / s).mkdir(parents=True, exist_ok=True)
            (self.yolo_dir / "labels" / s).mkdir(parents=True, exist_ok=True)

        images = {}
        if not (self.raw_path / "images.txt").exists(): return
        with open(self.raw_path / "images.txt", 'r') as f:
            for line in f: p = line.split(); images[p[0]] = p[1] if len(p) >= 2 else ""

        bboxes = {}
        with open(self.raw_path / "bounding_boxes.txt", 'r') as f:
            for line in f: p = line.split(); bboxes[p[0]] = (float(p[1]), float(p[2]), float(p[3]), float(p[4])) if len(
                p) >= 5 else None

        count, total = 0, len(images)
        for img_id, rel_path in images.items():
            if self.stop_requested: break
            src = self.raw_path / "images" / rel_path
            if not src.exists(): continue

            img = cv2.imread(str(src))
            if img is None: continue
            h_img, w_img, _ = img.shape

            if img_id not in bboxes or not bboxes[img_id]: continue
            x, y, w, h = bboxes[img_id]

            xc, yc = (x + w / 2) / w_img, (y + h / 2) / h_img
            wn, hn = w / w_img, h / h_img
            split = "train" if int(img_id) % 5 != 0 else "val"

            shutil.copy(src, self.yolo_dir / "images" / split / f"cub_{img_id}.jpg")
            with open(self.yolo_dir / "labels" / split / f"cub_{img_id}.txt", "w") as f:
                f.write(f"0 {xc} {yc} {wn} {hn}")
            count += 1
            if count % 2000 == 0: self.log(f"🔄 Procesadas {count}/{total} anotaciones...", "white")

    def _create_yaml(self):
        content = f"path: {self.yolo_dir.absolute()}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['bird']"
        with open(self.yolo_dir / "data.yaml", 'w') as f: f.write(content)

    def _train_anatomy_model(self):
        output_dir = self.models_dir / "phase1_anatomy"
        output_weights = output_dir / "weights" / "best.pt"

        # Si ya existe, asumimos que el usuario lo quiere usar (a menos que lo borrara en config)
        if output_weights.exists():
            self.log("✅ Fase 1 completada anteriormente. Saltando.", "green")
            return str(output_weights)

        # Usamos el nombre dinámico (ej: yolo11n.pt)
        model_file = f"{self.model_name}.pt"

        # Batch size
        batch_size = 16
        if "m" in self.model_name or "l" in self.model_name: batch_size = 8

        self.log(f"🧠 Fase 1: Entrenando {model_file} (Batch: {batch_size})", "green")
        device = 'mps' if torch.backends.mps.is_available() else 'auto'

        model = YOLO(model_file)  # Ultralytics bajará yolo11n.pt solo
        model.add_callback("on_train_epoch_end", self._monitor_progress)

        model.train(
            data=str(self.yolo_dir / "data.yaml"),
            epochs=20,
            imgsz=640,
            project=str(self.models_dir),
            name="phase1_anatomy",
            exist_ok=True,
            verbose=False,
            device=device,
            batch=batch_size,
            workers=0
        )

        return str(output_weights) if output_weights.exists() else ""

    def _monitor_progress(self, trainer):
        if self.stop_requested: raise KeyboardInterrupt("Stop")