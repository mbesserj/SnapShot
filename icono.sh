#!/bin/bash
# generate_icons.sh
# Script automatizado para generar todos los iconos necesarios

set -e  # Salir si hay error

echo "🎨 Generador de Iconos para SnapShot AI"
echo "========================================"
echo ""

# Verificar que se proporcionó una imagen
if [ -z "$1" ]; then
    echo "❌ Error: Debes proporcionar una imagen PNG"
    echo ""
    echo "Uso: ./generate_icons.sh imagen_original.png"
    echo ""
    echo "Requisitos de la imagen:"
    echo "  - Formato: PNG"
    echo "  - Tamaño mínimo: 1024x1024 píxeles"
    echo "  - Fondo: Transparente preferiblemente"
    exit 1
fi

ORIGINAL_IMAGE="$1"

# Verificar que el archivo existe
if [ ! -f "$ORIGINAL_IMAGE" ]; then
    echo "❌ Error: No se encuentra el archivo '$ORIGINAL_IMAGE'"
    exit 1
fi

echo "📁 Imagen original: $ORIGINAL_IMAGE"
echo ""

# Crear carpeta assets si no existe
mkdir -p assets

# ============================================
# GENERAR ICONO PARA macOS (.icns)
# ============================================
echo "🍎 Generando icono para macOS..."

# Crear carpeta temporal para iconset
ICONSET_DIR="camara.iconset"
mkdir -p "$ICONSET_DIR"

# Generar todos los tamaños necesarios
echo "  Generando tamaños..."
sips -z 16 16     "$ORIGINAL_IMAGE" --out "$ICONSET_DIR/icon_16x16.png" > /dev/null 2>&1
sips -z 32 32     "$ORIGINAL_IMAGE" --out "$ICONSET_DIR/icon_16x16@2x.png" > /dev/null 2>&1
sips -z 32 32     "$ORIGINAL_IMAGE" --out "$ICONSET_DIR/icon_32x32.png" > /dev/null 2>&1
sips -z 64 64     "$ORIGINAL_IMAGE" --out "$ICONSET_DIR/icon_32x32@2x.png" > /dev/null 2>&1
sips -z 128 128   "$ORIGINAL_IMAGE" --out "$ICONSET_DIR/icon_128x128.png" > /dev/null 2>&1
sips -z 256 256   "$ORIGINAL_IMAGE" --out "$ICONSET_DIR/icon_128x128@2x.png" > /dev/null 2>&1
sips -z 256 256   "$ORIGINAL_IMAGE" --out "$ICONSET_DIR/icon_256x256.png" > /dev/null 2>&1
sips -z 512 512   "$ORIGINAL_IMAGE" --out "$ICONSET_DIR/icon_256x256@2x.png" > /dev/null 2>&1
sips -z 512 512   "$ORIGINAL_IMAGE" --out "$ICONSET_DIR/icon_512x512.png" > /dev/null 2>&1
sips -z 1024 1024 "$ORIGINAL_IMAGE" --out "$ICONSET_DIR/icon_512x512@2x.png" > /dev/null 2>&1

# Convertir a .icns
echo "  Compilando .icns..."
iconutil -c icns "$ICONSET_DIR" -o assets/camara.icns

# Limpiar
rm -rf "$ICONSET_DIR"

echo "  ✅ assets/camara.icns creado"
echo ""

# ============================================
# GENERAR ICONO PARA WINDOWS (.ico)
# ============================================
echo "🪟 Generando icono para Windows..."

# Verificar si ImageMagick está instalado
if command -v convert &> /dev/null; then
    echo "  Usando ImageMagick..."
    
    # Crear múltiples tamaños y combinar en .ico
    convert "$ORIGINAL_IMAGE" \
        -resize 16x16 \
        -resize 32x32 \
        -resize 48x48 \
        -resize 64x64 \
        -resize 128x128 \
        -resize 256x256 \
        assets/camara.ico
    
    echo "  ✅ assets/camara.ico creado"
else
    echo "  ⚠️ ImageMagick no está instalado"
    echo "  Por favor, instálalo con: brew install imagemagick"
    echo ""
    echo "  O usa una herramienta online:"
    echo "  https://convertio.co/es/png-ico/"
    echo ""
    echo "  Luego guarda el archivo como: assets/camara.ico"
fi

echo ""

# ============================================
# GENERAR ICONO PARA LINUX (.png)
# ============================================
echo "🐧 Generando icono para Linux..."

# Para Linux, simplemente copiamos en diferentes tamaños
sips -z 256 256 "$ORIGINAL_IMAGE" --out assets/camara.png > /dev/null 2>&1
sips -z 512 512 "$ORIGINAL_IMAGE" --out assets/camara_512.png > /dev/null 2>&1

echo "  ✅ assets/camara.png creado (256x256)"
echo "  ✅ assets/camara_512.png creado (512x512)"
echo ""

# ============================================
# RESUMEN
# ============================================
echo "✅ GENERACIÓN COMPLETA"
echo "====================="
echo ""
echo "Archivos generados en carpeta 'assets/':"
ls -lh assets/camara*

echo ""
echo "📋 Próximos pasos:"
echo "  1. Verifica que los iconos se ven bien"
echo "  2. Ejecuta: pyinstaller snapshot_ai.spec"
echo "  3. Disfruta tu app con icono personalizado! 🎉"
