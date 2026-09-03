#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${1:-.}"
OUTPUT="${2:-mosaic.tif}"
CRS="${3:-}"   # e.g. EPSG:3857

cd "$INPUT_DIR"

export GDAL_GEOREF_SOURCES=WORLDFILE,PAM,INTERNAL

# # Determine image extension
# shopt -s nullglob

if (( $(ls -1 *.webp 2>/dev/null | wc -l) > 0 )); then
    EXT="webp"
elif (( $(ls -1 *.png 2>/dev/null | wc -l) > 0 )); then
    EXT="png"
else
    echo "No .png or .webp files found."
    exit 1
fi

echo "Using *.$EXT files"

gdalbuildvrt mosaic.vrt *."$EXT"

if [[ -n "$CRS" ]]; then
    gdal_translate \
        -a_srs "$CRS" \
        -of GTiff \
        -co COMPRESS=DEFLATE \
        -co TILED=YES \
        -co BIGTIFF=IF_SAFER \
        mosaic.vrt "$OUTPUT"
else
    gdal_translate \
        -of GTiff \
        -co COMPRESS=DEFLATE \
        -co TILED=YES \
        -co BIGTIFF=IF_SAFER \
        mosaic.vrt "$OUTPUT"
fi

rm mosaic.vrt

echo "Created $OUTPUT"
