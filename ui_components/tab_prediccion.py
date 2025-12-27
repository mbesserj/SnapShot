import flet as ft
from ultralytics import YOLO
from pathlib import Path


class TabPrediccion(ft.Container):
    def __init__(self, page):
        super().__init__()
        self.page = page
        self.padding = 20
        self.model = None
        self.selected_file_path = None

        # --- DICCIONARIO DE TRADUCCIÓN ---
        self.CHILE_MAP = {
            "zonotrichia capensis": "Chincol",
            "turdus falcklandii": "Zorzal",
            "sturnella loyca": "Loica",
            "vanellus chilensis": "Queltehue",
            "zenaida auriculata": "Tortola",
            "diuca diuca": "Diuca",
            "mimus thenca": "Tenca",
            "colaptes pitius": "Pitio",
            "vultur gryphus": "Cóndor",
            "parabuteo unicinctus": "Peuco",
            "glaucidium nana": "Chuncho",
            # ... puedes seguir agregando aquí ...
        }

        # --- UI Components ---
        self.img_preview = ft.Image(
            src="https://via.placeholder.com/400x300?text=Sube+una+foto",
            width=600,
            height=400,
            fit=ft.ImageFit.CONTAIN,
            visible=True
        )

        # Texto Principal (Nombre Común)
        self.lbl_result = ft.Text("Esperando imagen...", size=24, weight=ft.FontWeight.BOLD)

        # Texto Secundario (Científico + Certeza)
        self.lbl_scientific = ft.Text("", size=16, italic=True, color=ft.Colors.BLUE_GREY)
        self.lbl_confidence = ft.Text("", size=14, color=ft.Colors.GREY)

        self.btn_upload = ft.ElevatedButton(
            "Seleccionar Foto",
            icon=ft.Icons.UPLOAD_FILE,
            on_click=lambda _: self.file_picker.pick_files(allow_multiple=False)
        )

        self.btn_predict = ft.FilledButton(
            "Identificar Ave",
            icon=ft.Icons.SEARCH,
            on_click=self.predict_image,
            disabled=True
        )

        self.file_picker = ft.FilePicker(on_result=self.on_file_selected)
        self.page.overlay.append(self.file_picker)

        self.content = ft.Column([
            ft.Text("Identificador de Aves Chilenas", size=24, weight=ft.FontWeight.BOLD),
            ft.Divider(),
            ft.Row([self.btn_upload, self.btn_predict], alignment=ft.MainAxisAlignment.CENTER),
            ft.Container(height=20),
            ft.Row([self.img_preview], alignment=ft.MainAxisAlignment.CENTER),
            ft.Container(height=20),

            # Bloque de Resultados
            ft.Column([
                self.lbl_result,  # CHINCOL
                self.lbl_scientific,  # Zonotrichia capensis
                self.lbl_confidence  # 95%
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),

        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)

    def on_file_selected(self, e: ft.FilePickerResultEvent):
        if e.files and len(e.files) > 0:
            self.selected_file_path = e.files[0].path
            self.img_preview.src = self.selected_file_path
            self.img_preview.update()
            self.btn_predict.disabled = False

            # Resetear textos
            self.lbl_result.value = "Foto cargada."
            self.lbl_result.color = ft.Colors.BLACK
            self.lbl_scientific.value = ""
            self.lbl_confidence.value = ""
            self.page.update()

    def predict_image(self, e):
        model_path = Path.cwd() / "models" / "chile_birds_final" / "weights" / "best.pt"

        if not model_path.exists():
            self.lbl_result.value = "⚠️ Modelo no encontrado (Entrena primero)."
            self.lbl_result.color = ft.Colors.ORANGE
            self.lbl_scientific.value = ""
            self.lbl_confidence.value = ""
            self.page.update()
            return

        try:
            self.lbl_result.value = "🔍 Analizando..."
            self.page.update()

            if self.model is None:
                self.model = YOLO(model_path)

            results = self.model(self.selected_file_path)
            result = results[0]

            # --- FILTRO DE HONESTIDAD ---
            UMBRAL_CONFIANZA = 0.40  # 40% de certeza mínima

            if len(result.boxes) == 0:
                self.lbl_result.value = "🤷‍♂️ No veo ningún ave."
                self.lbl_scientific.value = ""
                self.lbl_confidence.value = ""
            else:
                box = result.boxes[0]
                confidence = float(box.conf[0])

                # Si la certeza es muy baja, no nos arriesgamos
                if confidence < UMBRAL_CONFIANZA:
                    self.lbl_result.value = "⚠️ Especie no identificada"
                    self.lbl_result.color = ft.Colors.ORANGE
                    self.lbl_scientific.value = "(Certeza insuficiente para clasificar)"
                    self.lbl_confidence.value = f"Nivel de confianza: {confidence * 100:.1f}% (Muy bajo)"
                else:
                    # Si pasa el umbral, mostramos el resultado
                    class_id = int(box.cls[0])
                    raw_name = result.names[class_id]
                    clean_name = raw_name.replace("_", " ").lower()

                    # Traducción
                    common_name = self.CHILE_MAP.get(clean_name, clean_name.capitalize())

                    self.lbl_result.value = f"¡Es un {common_name}!"
                    self.lbl_result.color = ft.Colors.GREEN
                    self.lbl_scientific.value = f"({clean_name.capitalize()})"
                    self.lbl_confidence.value = f"Certeza: {confidence * 100:.1f}%"

            self.page.update()

        except Exception as ex:
            self.lbl_result.value = f"Error: {ex}"
            self.page.update()