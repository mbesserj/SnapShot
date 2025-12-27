# ui_components/tab_entrenar.py
import flet as ft
import threading
from datetime import datetime
from species_manager import get_species_manager


class TabEntrenar(ft.Container):
    def __init__(self, page):
        super().__init__()
        self.page = page
        self.padding = 20
        self.manager = get_species_manager()
        self.current_trainer = None
        self.is_running = False

        # --- UI Elements ---
        countries = self.manager.get_countries()

        self.dropdown_country = ft.Dropdown(
            label="País Objetivo",
            options=[ft.dropdown.Option(c) for c in countries],
            value="Chile",
            expand=True
        )

        # Sliders
        self.txt_images_val = ft.Text("200", weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE)
        self.txt_epochs_val = ft.Text("50", weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE)

        self.slider_images = ft.Slider(min=50, max=1000, value=200, label="{value}", divisions=19,
                                       on_change=lambda e: self.update_labels())
        self.slider_epochs = ft.Slider(min=10, max=300, value=50, label="{value}", divisions=29,
                                       on_change=lambda e: self.update_labels())

        self.btn_train = ft.FilledButton("Iniciar Entrenamiento", icon=ft.Icons.ROCKET_LAUNCH,
                                         on_click=self.on_train_click)
        self.btn_stop = ft.OutlinedButton("DETENER", icon=ft.Icons.STOP, style=ft.ButtonStyle(color=ft.Colors.RED),
                                          visible=False, on_click=self.stop_process)

        # --- NUEVO: DASHBOARD DE MÉTRICAS ---
        self.lbl_epoch_big = ft.Text("- / -", size=30, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE)
        self.lbl_box_loss = ft.Text("-", size=20, weight=ft.FontWeight.BOLD)
        self.lbl_cls_loss = ft.Text("-", size=20, weight=ft.FontWeight.BOLD)
        self.lbl_dfl_loss = ft.Text("-", size=20, weight=ft.FontWeight.BOLD)

        self.metrics_container = ft.Container(
            visible=False,
            padding=10,
            bgcolor=ft.Colors.BLUE_50,
            border_radius=10,
            content=ft.Column([
                ft.Text("Métricas en Tiempo Real", size=12, color=ft.Colors.GREY),
                ft.Row([
                    ft.Column([ft.Text("ÉPOCA", size=10), self.lbl_epoch_big], alignment=ft.MainAxisAlignment.CENTER),
                    ft.VerticalDivider(),
                    self._build_metric_card("Box Loss", self.lbl_box_loss, ft.Colors.ORANGE),
                    self._build_metric_card("Class Loss", self.lbl_cls_loss, ft.Colors.GREEN),
                    self._build_metric_card("DFL Loss", self.lbl_dfl_loss, ft.Colors.PURPLE),
                ], alignment=ft.MainAxisAlignment.SPACE_EVENLY)
            ])
        )
        # ------------------------------------

        self.progress_bar = ft.ProgressBar(value=0, color="green", visible=False)
        self.log_list = ft.ListView(expand=True, spacing=2, auto_scroll=True)

        self.content = ft.Column([
            ft.Row([self.dropdown_country]),
            ft.Divider(),
            ft.Row([ft.Text("Imágenes por especie:"), self.txt_images_val]), self.slider_images,
            ft.Row([ft.Text("Épocas de entrenamiento:"), self.txt_epochs_val]), self.slider_epochs,
            ft.Divider(),
            ft.Row([self.btn_train, self.btn_stop], alignment=ft.MainAxisAlignment.CENTER),

            # Insertamos el Dashboard aquí
            self.metrics_container,
            self.progress_bar,

            ft.Container(content=self.log_list, bgcolor=ft.Colors.BLACK87, border_radius=10, padding=10, expand=True)
        ])

    def _build_metric_card(self, title, control, color):
        return ft.Container(
            padding=5, border=ft.border.all(1, color), border_radius=5,
            content=ft.Column([
                ft.Text(title, size=10, color=color),
                control
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
        )

    def update_labels(self):
        self.txt_images_val.value = str(int(self.slider_images.value))
        self.txt_epochs_val.value = str(int(self.slider_epochs.value))
        self.page.update()

    def log_msg(self, msg, color="white"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}")
        ui_color = color if color != "black" else "white"
        self.log_list.controls.append(ft.Text(msg, color=ui_color, font_family="Consolas", size=12))
        self.page.update()

    # --- NUEVA FUNCIÓN QUE RECIBE DATOS ---
    def update_dashboard(self, data):
        """
        Recibe un diccionario con datos: { 'epoch': 1, 'total_epochs': 20, 'box': 1.2, 'cls': 2.5, 'dfl': 1.5 }
        """
        if not self.metrics_container.visible:
            self.metrics_container.visible = True

        # Actualizar textos
        if 'epoch' in data:
            self.lbl_epoch_big.value = f"{data['epoch']} / {data['total_epochs']}"

        # Actualizar losses (formatear a 4 decimales)
        if 'box' in data: self.lbl_box_loss.value = f"{data['box']:.4f}"
        if 'cls' in data: self.lbl_cls_loss.value = f"{data['cls']:.4f}"
        if 'dfl' in data: self.lbl_dfl_loss.value = f"{data['dfl']:.4f}"

        # Actualizar barra de progreso (cálculo porcentual)
        if 'epoch' in data and 'total_epochs' in data:
            self.progress_bar.value = data['epoch'] / data['total_epochs']

        self.page.update()

    def on_train_click(self, e):
        if self.is_running: return
        self.is_running = True
        self.btn_train.disabled = True
        self.slider_images.disabled = True
        self.slider_epochs.disabled = True
        self.dropdown_country.disabled = True
        self.btn_stop.visible = True
        self.progress_bar.visible = True
        self.metrics_container.visible = False  # Ocultar al inicio hasta tener datos
        self.page.update()
        threading.Thread(target=self.orchestrate_training, daemon=True).start()

    def stop_process(self, e):
        if self.current_trainer:
            self.current_trainer.request_stop()
            self.btn_stop.disabled = True
            self.log_msg("🛑 Deteniendo...", "orange")

    def orchestrate_training(self):
        try:
            # FASE 1
            from training_modules.pretrainer import AnatomyPretrainer
            # Pasamos update_dashboard como callback también (aunque phase 1 es corta)
            pretrainer = AnatomyPretrainer(self.log_msg, self.update_dashboard)
            self.current_trainer = pretrainer

            self.log_msg("--- FASE 1: BASE ANATÓMICA (CUB-200) ---", "cyan")
            base_model = pretrainer.run()

            if self.current_trainer.stop_requested: return
            if not base_model:
                from config_manager import load_config
                base_model = load_config().get("yolo_base_model", "yolov8m.pt")

            # FASE 2
            from training_modules.finetuner import SpeciesFineTuner
            finetuner = SpeciesFineTuner(
                self.log_msg,
                self.update_dashboard,  # <--- Pasamos la nueva función
                self.dropdown_country.value,
                int(self.slider_images.value),
                int(self.slider_epochs.value)
            )
            self.current_trainer = finetuner

            self.log_msg(f"--- FASE 2: ESPECIALIZACIÓN ({self.dropdown_country.value}) ---", "cyan")
            finetuner.run(base_model_path=base_model)

        except Exception as e:
            self.log_msg(f"Error crítico: {e}", "red")
        finally:
            self.is_running = False
            self.btn_train.disabled = False
            self.slider_images.disabled = False
            self.slider_epochs.disabled = False
            self.dropdown_country.disabled = False
            self.btn_stop.visible = False
            self.btn_stop.disabled = False
            self.progress_bar.visible = False
            self.page.update()