import os
import torch
import yaml
import hashlib
import cv2
import requests
import time
import shutil
from pathlib import Path
from ultralytics import YOLO
from species_manager import get_species_manager
from config_manager import load_config
from concurrent.futures import ThreadPoolExecutor


class DatasetGenerator:
    """Encargado de descargar imágenes y generar etiquetas automáticas."""

    def __init__(self, log_callback, dataset_dir):
        self.log = log_callback
        self.dataset_dir = dataset_dir
        self.headers = {"User-Agent": "SnapShotAI/1.0"}
        self.taxon_to_class_id = {}

    def download_species_images(self, species_list, max_images, stop_event):
        self.log(f"📡 Iniciando descarga masiva ({max_images} imgs/especie)...", "blue")
        (self.dataset_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.dataset_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)

        total_species = len(species_list)

        for idx, sp in enumerate(species_list):
            if stop_event(): break

            taxon_id = sp.get("taxon_id")
            if not taxon_id: continue

            clean_name = sp["common_name"].replace(" ", "_").lower()
            current_count = len(list((self.dataset_dir / "images" / "train").glob(f"{taxon_id}_*.jpg")))
            needed = max_images - current_count

            if needed <= 0:
                self.log(f"✅ {sp['common_name']} completa ({current_count} imgs).", "grey")
                continue

            self.log(f"⬇️ Descargando {needed} fotos para: {sp['common_name']} ({idx + 1}/{total_species})", "cyan")
            self._fetch_from_inaturalist(taxon_id, needed, clean_name)

    def _fetch_from_inaturalist(self, taxon_id, count, name_prefix):
        url = "https://api.inaturalist.org/v1/observations"
        params = {"taxon_id": taxon_id, "quality_grade": "research", "photos": "true", "per_page": min(count, 200),
                  "order_by": "votes"}

        try:
            r = requests.get(url, params=params, headers=self.headers, timeout=10)
            if not r.ok: return

            results = r.json().get("results", [])
            downloaded = 0

            for obs in results:
                if downloaded >= count: break
                photos = obs.get("photos", [])
                if not photos: continue

                img_url = photos[0].get("url", "").replace("square", "medium")
                filename = f"{taxon_id}_{obs['id']}.jpg"
                filepath = self.dataset_dir / "images" / "train" / filename

                if not filepath.exists():
                    self._download_file(img_url, filepath)
                    downloaded += 1
        except:
            pass

    def _download_file(self, url, path):
        try:
            with requests.get(url, stream=True, timeout=10) as r:
                if r.ok:
                    with open(path, 'wb') as f: shutil.copyfileobj(r.raw, f)
        except:
            pass

    def auto_label_dataset(self, model_path, stop_event):
        """Usa el modelo base (YOLO11 o v8) para detectar pájaros."""
        self.log(f"🏷️ Auto-etiquetando con {Path(model_path).name}...", "blue")

        model = YOLO(model_path)
        images = list((self.dataset_dir / "images" / "train").glob("*.jpg"))
        count = 0

        for img_path in images:
            if stop_event(): break

            label_path = self.dataset_dir / "labels" / "train" / img_path.name.replace(".jpg", ".txt")
            if label_path.exists(): continue

            # Detectar pájaro (bird=clase 0 en CUB/Anatomia, o bird=14 en COCO)
            results = model(str(img_path), verbose=False, conf=0.25)

            best_box = None
            max_conf = 0

            for r in results:
                for box in r.boxes:
                    if box.conf[0] > max_conf:
                        best_box = box
                        max_conf = box.conf[0]

            if best_box:
                taxon_id = img_path.name.split("_")[0]
                class_id = self.taxon_to_class_id.get(str(taxon_id))

                if class_id is not None:
                    xywh = best_box.xywhn[0].tolist()
                    with open(label_path, "w") as f:
                        f.write(f"{class_id} {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]}\n")
                    count += 1
            else:
                img_path.unlink()  # Borrar si no hay pájaro

        self.log(f"✅ Etiquetadas {count} imágenes nuevas.", "green")


class SpeciesFineTuner:
    def __init__(self, log_callback, dashboard_callback, country, images_per_species, epochs):
        self.log = log_callback
        self.update_dash = dashboard_callback if dashboard_callback else lambda x: None

        self.config = load_config()

        # --- LÓGICA DE NOMBRE DE MODELO (YOLO11 Ready) ---
        raw_variant = self.config.get("yolo_model_variant", "yolov8n")
        if len(raw_variant) == 1:
            self.model_name = f"yolov8{raw_variant}"
        else:
            self.model_name = raw_variant
        # -------------------------------------------------

        self.country = country
        self.target_images = images_per_species
        self.epochs = epochs
        self.manager = get_species_manager()
        self.stop_requested = False

        clean_country = country.lower().replace(' ', '_')
        self.dataset_dir = Path.cwd() / f"dataset_{clean_country}"
        self.models_dir = Path.cwd() / "models"

        self.generator = DatasetGenerator(self.log, self.dataset_dir)

    def request_stop(self):
        self.stop_requested = True

    def run(self, base_model_path=None):
        try:
            # Si no viene un modelo base específico (ej. fase 1), usamos el stock (ej. yolo11n.pt)
            if not base_model_path:
                base_model_path = f"{self.model_name}.pt"

            self.log(f"🚀 Iniciando Pipeline para {self.country} (Base: {Path(base_model_path).name})", "blue")

            species_list = self.manager.get_species(self.country)
            if not species_list:
                self.log("❌ Lista de especies vacía.", "red");
                return

            self._create_class_mapping(species_list)

            self.generator.download_species_images(species_list, self.target_images, lambda: self.stop_requested)

            self.generator.taxon_to_class_id = self.taxon_map
            self.generator.auto_label_dataset(base_model_path, lambda: self.stop_requested)

            self._create_yaml(species_list)
            self._train_final_model(base_model_path)

        except Exception as e:
            self.log(f"❌ Error crítico: {e}", "red")

    def _create_class_mapping(self, species_list):
        self.taxon_map = {}
        self.names_map = {}
        valid_species = [s for s in species_list if s.get("taxon_id")]
        valid_species.sort(key=lambda x: x["common_name"])

        for idx, sp in enumerate(valid_species):
            tid = str(sp["taxon_id"])
            self.taxon_map[tid] = idx
            self.names_map[idx] = sp["common_name"]

        import json
        map_path = self.models_dir / "class_mapping.json"
        with open(map_path, "w") as f:
            json.dump({
                "class_to_id": {v: k for k, v in self.names_map.items()},
                "species_info": {sp["common_name"]: sp for sp in valid_species}
            }, f, indent=2)

    def _create_yaml(self, species_list):
        names_list = [self.names_map[i] for i in range(len(self.names_map))]
        content = f"path: {self.dataset_dir.absolute()}\ntrain: images/train\nval: images/train\nnc: {len(names_list)}\nnames: {names_list}"
        with open(self.dataset_dir / "data.yaml", "w") as f: f.write(content)

    def _train_final_model(self, base_weights):
        if self.stop_requested: return
        project_name = f"{self.country.lower()}_birds_final"
        device = 'mps' if torch.backends.mps.is_available() else 'auto'

        self.log(f"🧠 Entrenando modelo final con {Path(base_weights).name}...", "green")

        model = YOLO(base_weights)
        model.add_callback("on_train_epoch_end", self._monitor_progress)

        bs = 16
        if "m" in self.model_name or "l" in self.model_name: bs = 8

        model.train(
            data=str(self.dataset_dir / "data.yaml"),
            epochs=self.epochs,
            imgsz=640,
            project=str(self.models_dir),
            name=project_name,
            exist_ok=True,
            verbose=False,
            device=device,
            batch=bs,
            workers=2
        )

        final = self.models_dir / project_name / "weights" / "best.pt"
        if final.exists():
            self.log(f"🏆 ¡MODELO LISTO! Guardado en: {final}", "green")

    def _monitor_progress(self, trainer):
        if self.stop_requested: raise KeyboardInterrupt("Stop")
        try:
            current = trainer.epoch + 1
            total = trainer.epochs
            losses = trainer.loss_items
            data = {'epoch': current, 'total_epochs': total, 'box': losses[0].item(), 'cls': losses[1].item(),
                    'dfl': losses[2].item()}
            self.update_dash(data)
        except:
            pass