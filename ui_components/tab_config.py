import flet as ft
import shutil
from pathlib import Path
from config_manager import load_config, save_config

class TabConfiguracion(ft.Container):
    def __init__(self, page):
        super().__init__()
        self.page = page
        self.padding = 20
        self.config = load_config()

        self.phase1_path = Path.cwd() / "models" / "phase1_anatomy"

        self.input_url = ft.TextField(
            label="URL Dataset CUB-200 (Anatomía)",
            value=self.config.get("cub_200_url", "https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz"),
            width=600
        )

        # --- AQUÍ ESTÁ EL CAMBIO IMPORTANTE ---
        # Normalizamos lo que viene del config antiguo (si dice "n", lo pasamos a "yolov8n")
        saved_variant = self.config.get("yolo_model_variant", "yolov8n")
        if len(saved_variant) == 1: saved_variant = f"yolov8{saved_variant}"

        self.dd_model_type = ft.Dropdown(
            label="Arquitectura del Modelo Base",
            width=600,
            options=[
                ft.dropdown.Option("yolo11n", "YOLO11 Nano (2025) - ⭐ Recomendado"),
                ft.dropdown.Option("yolo11s", "YOLO11 Small - Más preciso"),
                ft.dropdown.Option("yolo11m", "YOLO11 Medium - Potencia bruta"),
                ft.dropdown.Option("yolov8n", "YOLOv8 Nano (Clásico)"),
                ft.dropdown.Option("yolov8m", "YOLOv8 Medium (Clásico)"),
            ],
            value=saved_variant
        )
        # --------------------------------------

        self.btn_save = ft.ElevatedButton("Guardar Configuración", icon=ft.Icons.SAVE, on_click=self.on_save_click)
        self.snack_bar = ft.SnackBar(ft.Text("Configuración guardada"))

        self.dlg_confirm = ft.AlertDialog(
            modal=True,
            title=ft.Text("⚠️ Cambio de Arquitectura"),
            content=ft.Text(
                "Has cambiado la versión del modelo (ej. de v8 a 11).\n"
                "Para aplicar esto, DEBEMOS borrar el entrenamiento anterior de la Fase 1.\n\n"
                "¿Deseas borrar la carpeta 'phase1_anatomy' y re-entrenar?"
            ),
            actions=[
                ft.TextButton("Cancelar", on_click=self.close_dialog),
                ft.TextButton("Borrar y Guardar", on_click=self.confirm_delete_and_save, style=ft.ButtonStyle(color=ft.Colors.RED)),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self.content = ft.Column([
            ft.Text("Configuración del Sistema", size=20, weight=ft.FontWeight.BOLD),
            ft.Divider(),
            self.input_url,
            ft.Container(height=10),
            ft.Text("Selecciona el Cerebro (Modelo):", size=14, weight="bold"),
            ft.Text("• YOLO11: Última generación. Más rápido y preciso con menos parámetros.", size=12, color=ft.Colors.BLUE),
            self.dd_model_type,
            ft.Divider(),
            self.btn_save,
            self.snack_bar
        ])

    def on_save_click(self, e):
        current = self.dd_model_type.value
        # Leemos config de nuevo para comparar
        saved_raw = self.config.get("yolo_model_variant", "yolov8n")
        if len(saved_raw) == 1: saved_raw = f"yolov8{saved_raw}"

        if current != saved_raw and self.phase1_path.exists():
            self.page.dialog = self.dlg_confirm
            self.dlg_confirm.open = True
            self.page.update()
        else:
            self.execute_save()

    def close_dialog(self, e):
        self.dlg_confirm.open = False
        self.page.update()

    def confirm_delete_and_save(self, e):
        try:
            if self.phase1_path.exists():
                shutil.rmtree(self.phase1_path)
            self.execute_save()
            self.close_dialog(e)
            self.page.show_snack_bar(ft.SnackBar(ft.Text("¡Listo para YOLO11!")))
        except Exception as ex:
            self.close_dialog(e)
            self.page.show_snack_bar(ft.SnackBar(ft.Text(f"Error: {ex}"), bgcolor=ft.Colors.RED))

    def execute_save(self):
        self.config["cub_200_url"] = self.input_url.value
        self.config["yolo_model_variant"] = self.dd_model_type.value
        save_config(self.config)
        self.snack_bar.open = True
        self.page.update()