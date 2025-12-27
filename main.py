import flet as ft

# 1. Importamos las clases EXACTAS
from ui_components.tab_especies import TabEspecies
from ui_components.tab_entrenar import TabEntrenar
from ui_components.tab_procesar import TabProcesar
from ui_components.tab_config import TabConfiguracion
from ui_components.tab_prediccion import TabPrediccion  # <--- IMPORTANTE

def main(page: ft.Page):
    # 2. Configuración de la ventana
    page.title = "SnapShot AI Trainer"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window.width = 1100
    page.window.height = 800
    page.padding = 10

    # 3. Instanciar tus pestañas
    tab_especies = TabEspecies(page)
    tab_entrenar = TabEntrenar(page)
    tab_procesar = TabProcesar(page)
    tab_config = TabConfiguracion(page)
    tab_prediccion = TabPrediccion(page)  # <--- INSTANCIA

    # 4. Crear el sistema de navegación
    t = ft.Tabs(
        selected_index=0,
        animation_duration=300,
        tabs=[
            ft.Tab(
                text="Gestión de Especies",
                icon=ft.Icons.LIST_ALT,
                content=tab_especies
            ),
            ft.Tab(
                text="Entrenamiento",
                icon=ft.Icons.ROCKET_LAUNCH,
                content=tab_entrenar
            ),
            ft.Tab(
                text="Prueba / Predicción",  # <--- NUEVA PESTAÑA
                icon=ft.Icons.VISIBILITY,
                content=tab_prediccion
            ),
            ft.Tab(
                text="Procesamiento",
                icon=ft.Icons.IMAGE_SEARCH,
                content=tab_procesar
            ),
            ft.Tab(
                text="Configuración",
                icon=ft.Icons.SETTINGS,
                content=tab_config
            ),
        ],
        expand=True,
    )

    page.add(t)

if __name__ == "__main__":
    ft.app(target=main)