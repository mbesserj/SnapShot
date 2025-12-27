import flet as ft
import threading
from logic import BurstProcessor


class TabProcesar(ft.Container):
    def __init__(self, page):
        super().__init__()
        self.page = page
        self.padding = 20
        self.selected_path_txt = ft.Text("Ninguna carpeta", italic=True, color=ft.Colors.GREY_600)
        self.log_list = ft.ListView(expand=True, spacing=5, auto_scroll=True)
        self.progress_bar = ft.ProgressBar(expand=True, value=0, color="blue")
        self.progress_text = ft.Text("0%", width=40)
        self.progress_row = ft.Row([self.progress_bar, self.progress_text], visible=False)
        self.chk_copy = ft.Checkbox(label="Solo copiar", value=True)
        self.chk_ai = ft.Checkbox(label="IA", value=True)
        self.chk_bird_id = ft.Checkbox(label="Especies", value=True)
        self.chk_debug = ft.Checkbox(label="Debug", value=False)

        self.file_picker = ft.FilePicker(on_result=self.on_folder_selected)
        self.page.overlay.append(self.file_picker)

        self.content = ft.Column([
            ft.Row([ft.ElevatedButton("Seleccionar Carpeta", icon=ft.Icons.FOLDER_OPEN,
                                      on_click=lambda _: self.file_picker.get_directory_path()),
                    self.selected_path_txt]),
            ft.Divider(),
            ft.Row([self.chk_copy, self.chk_ai, self.chk_bird_id, self.chk_debug]),
            ft.FilledButton("INICIAR PROCESAMIENTO", icon=ft.Icons.PLAY_ARROW, on_click=self.on_process_click,
                            height=45),
            self.progress_row,
            ft.Container(expand=True, content=self.log_list, bgcolor=ft.Colors.GREY_50, border_radius=10, padding=10,
                         border=ft.border.all(1, ft.Colors.GREY_200))
        ])

    def on_folder_selected(self, e):
        if e.path:
            self.selected_path_txt.value = e.path
            self.page.update()

    def on_process_click(self, e):
        if "Ninguna" in self.selected_path_txt.value: return
        self.progress_row.visible = True
        processor = BurstProcessor(self.log_msg, self.update_prog, self.chk_bird_id.value)
        threading.Thread(target=lambda: processor.process_folder(self.selected_path_txt.value, self.chk_ai.value,
                                                                 self.chk_copy.value, self.chk_debug.value),
                         daemon=True).start()

    def log_msg(self, m, c=ft.Colors.BLACK):
        self.log_list.controls.append(ft.Text(m, color=c, size=12)); self.page.update()

    def update_prog(self, v):
        self.progress_bar.value = v; self.progress_text.value = f"{int(v * 100)}%"; self.page.update()