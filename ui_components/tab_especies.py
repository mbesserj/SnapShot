import flet as ft
import threading
import requests
from species_manager import get_species_manager, reload_species


class TabEspecies(ft.Container):
    def __init__(self, page):
        super().__init__()
        self.page = page
        self.padding = ft.padding.only(top=15, left=15, right=15, bottom=15)
        self.manager = get_species_manager()

        # --- UI Elements ---
        countries = self.manager.get_countries()

        self.species_country_dropdown = ft.Dropdown(
            label="País",
            options=[ft.dropdown.Option(c) for c in countries],
            value="Chile",
            width=200,
            on_change=lambda _: self.refresh_table()
        )

        self.download_btn = ft.FilledButton("📥 Sincronizar Catálogo", on_click=self.on_download)

        # Botón para agregar manual
        self.add_btn = ft.OutlinedButton("➕ Agregar Manualmente", on_click=self.open_add_dialog)

        self.progress_bar = ft.ProgressBar(value=0, visible=False)
        self.status_txt = ft.Text("", size=12, italic=True)

        # Tabla con columna extra para borrar
        self.table = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Nombre")),
                ft.DataColumn(ft.Text("Científico")),
                ft.DataColumn(ft.Text("IA")),
                ft.DataColumn(ft.Text("Acción")),  # Columna para el basurero
            ],
            show_checkbox_column=False,
            expand=True
        )

        # Preview Image
        self.preview_image = ft.Image(src="", width=280, height=280, fit=ft.ImageFit.CONTAIN, border_radius=10)
        self.preview_title = ft.Text("Selecciona ave", size=18, weight="bold")
        self.preview_subtitle = ft.Text("", italic=True, size=14)

        # --- DIÁLOGO PARA AGREGAR ESPECIE ---
        self.input_common = ft.TextField(label="Nombre Común (ej. Picaflor Gigante)")
        self.input_scientific = ft.TextField(label="Nombre Científico (ej. Patagona gigas)")

        self.dlg_add = ft.AlertDialog(
            title=ft.Text("Agregar Especie Nueva"),
            content=ft.Column([
                ft.Text("Ingresa los datos del ave faltante:"),
                self.input_common,
                self.input_scientific
            ], height=200, tight=True),
            actions=[
                ft.TextButton("Cancelar", on_click=self.close_add_dialog),
                ft.TextButton("Guardar", on_click=self.save_new_species),
            ],
        )

        # Layout Principal
        self.content = ft.Row([
            ft.Column([
                ft.Row([
                    self.species_country_dropdown,
                    self.download_btn,
                    self.add_btn  # Nuevo botón
                ]),
                self.progress_bar,
                self.status_txt,
                ft.Container(
                    content=ft.Column([self.table], scroll=ft.ScrollMode.AUTO),
                    expand=True,
                    border=ft.border.all(1, ft.Colors.GREY_300),
                    border_radius=10
                )
            ], expand=True),

            # Panel Lateral Derecho (Preview)
            ft.Container(
                content=ft.Column([
                    self.preview_title,
                    self.preview_subtitle,
                    ft.Divider(),
                    self.preview_image
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                width=300,
                padding=15,
                bgcolor=ft.Colors.GREY_50,
                border_radius=10
            )
        ], expand=True, spacing=15)

        self.refresh_table()

    # --- LÓGICA DE AGREGAR ---
    def open_add_dialog(self, e):
        self.input_common.value = ""
        self.input_scientific.value = ""
        self.page.dialog = self.dlg_add
        self.dlg_add.open = True
        self.page.update()

    def close_add_dialog(self, e):
        self.dlg_add.open = False
        self.page.update()

    def save_new_species(self, e):
        if not self.input_common.value or not self.input_scientific.value:
            return  # Validación simple

        country = self.species_country_dropdown.value
        self.manager.add_species(country, self.input_common.value, self.input_scientific.value)

        self.close_add_dialog(None)
        self.refresh_table()
        self.page.show_snack_bar(ft.SnackBar(ft.Text(f"Agregado: {self.input_common.value}")))

    # --- LÓGICA DE BORRAR ---
    def delete_species(self, scientific_name, common_name):
        country = self.species_country_dropdown.value
        self.manager.delete_species(country, scientific_name)
        self.refresh_table()
        self.page.show_snack_bar(ft.SnackBar(ft.Text(f"Eliminado: {common_name}"), bgcolor=ft.Colors.RED_400))

    # --- VISUALIZACIÓN ---
    def show_preview(self, taxon_id, common, scientific):
        self.preview_title.value = common
        self.preview_subtitle.value = scientific

        # Imagen por defecto si no hay ID (manual) o falló carga
        self.preview_image.src = "https://www.inaturalist.org/assets/copyright-none.png"
        self.page.update()

        if taxon_id == 0: return  # Especie manual sin foto online

        def fetch():
            try:
                r = requests.get(f"https://api.inaturalist.org/v1/taxa/{taxon_id}", timeout=5).json()
                if r['results'] and r['results'][0].get('default_photo'):
                    self.preview_image.src = r['results'][0]['default_photo']['url'].replace("square", "medium")
                    self.page.update()
            except:
                pass

        threading.Thread(target=fetch, daemon=True).start()

    def refresh_table(self):
        self.table.rows.clear()

        # Obtenemos la lista actual
        species_list = self.manager.get_species(self.species_country_dropdown.value)

        for sp in species_list:
            sc_name = sp["scientific_name"]
            cm_name = sp["common_name"]

            # Fila de datos
            self.table.rows.append(
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(cm_name, weight="bold")),
                        ft.DataCell(ft.Text(sc_name, italic=True)),
                        ft.DataCell(ft.Icon(ft.Icons.CIRCLE_OUTLINED, color=ft.Colors.GREY_400)),
                        # Botón de Borrar (ROJO)
                        ft.DataCell(
                            ft.IconButton(
                                icon=ft.Icons.DELETE_OUTLINE,
                                icon_color=ft.Colors.RED,
                                tooltip="Eliminar especie",
                                on_click=lambda e, s=sc_name, c=cm_name: self.delete_species(s, c)
                            )
                        )
                    ],
                    # Al hacer click en la fila mostramos preview
                    on_select_changed=lambda e, s=sp: self.show_preview(s.get("taxon_id", 0), s["common_name"],
                                                                        s["scientific_name"])
                )
            )
        self.page.update()

    def on_download(self, e):
        country = self.species_country_dropdown.value
        place_id = self.manager.get_country_place_id(country)
        self.download_btn.disabled = True
        self.progress_bar.visible = True
        self.status_txt.value = "Conectando con iNaturalist..."
        self.page.update()
        threading.Thread(target=self._run_download, args=(country, place_id), daemon=True).start()

    def _run_download(self, country, place_id):
        try:
            from species_downloader import SpeciesDownloader
            downloader = SpeciesDownloader(
                log_callback=lambda m, c=None: setattr(self.status_txt, 'value', m) or self.page.update(),
                progress_callback=lambda v: setattr(self.progress_bar, 'value', v) or self.page.update()
            )
            # Descargamos
            downloader.download_and_save(country, place_id, 1, "bird_species.json")

            # Recargamos la interfaz
            reload_species()
            self.manager = get_species_manager()
            self.refresh_table()

            self.status_txt.value = "¡Sincronización completada!"
        except Exception as e:
            self.status_txt.value = f"Error: {e}"
        finally:
            self.download_btn.disabled = False
            self.progress_bar.visible = False
            self.page.update()