# ui.py
import flet as ft
from ui_components.tab_procesar import TabProcesar
from ui_components.tab_especies import TabEspecies
from ui_components.tab_entrenar import TabEntrenar


class SnapshotUI:
    def __init__(self, page: ft.Page, icon_path: str = None):
        self.page = page

        # Instanciamos los componentes como clases independientes
        self.tab_procesar = TabProcesar(page)
        self.tab_especies = TabEspecies(page)
        self.tab_entrenar = TabEntrenar(page)

        # Configuramos la navegación por pestañas
        self.tabs = ft.Tabs(
            selected_index=0,
            expand=True,
            animation_duration=300,
            tabs=[
                ft.Tab(
                    text="📸 PROCESAR",
                    icon=ft.Icons.IMAGE_SEARCH,
                    content=self.tab_procesar
                ),
                ft.Tab(
                    text="🐦 ESPECIES",
                    icon=ft.Icons.LIST_ALT,
                    content=self.tab_especies
                ),
                ft.Tab(
                    text="🚀 ENTRENAR",
                    icon=ft.Icons.MODEL_TRAINING,
                    content=self.tab_entrenar
                ),
            ]
        )

        self.page.add(self.tabs)