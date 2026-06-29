#!/usr/bin/env bash
#
# package_redist.sh — assemble a self-contained Windows folder from an MSYS2
# build: ffmpeg.exe + ffprobe.exe plus all the ucrt64 runtime DLLs they need
# (so it runs without MSYS2 installed). Windows system DLLs are not bundled.
#
#   ./tools/package_redist.sh --bin-dir build-redist --out dist \
#       [--extra /c/path/onnxruntime.dll --extra /c/path/DirectML.dll]
#
set -euo pipefail

BIN_DIR="" OUT="" ; EXTRAS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bin-dir) BIN_DIR="$2"; shift 2 ;;
        --out)     OUT="$2";     shift 2 ;;
        --extra)   EXTRAS+=("$2"); shift 2 ;;   # bundle an extra DLL (repeatable)
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

[[ -n "$BIN_DIR" && -n "$OUT" ]] || { echo "Usage: $0 --bin-dir DIR --out DIR [--extra DLL ...]" >&2; exit 1; }
for e in ffmpeg.exe ffprobe.exe; do
    [[ -f "$BIN_DIR/$e" ]] || { echo "Error: $BIN_DIR/$e not found" >&2; exit 1; }
done

# ldd lines are tab-indented:  "\tname.dll => /c/msys64/ucrt64/bin/name.dll (0x..)"
# Field 3 ($3) is the resolved path. awk is indentation-agnostic (unlike anchored regex).
ucrt_dlls() {
    ldd "$1" 2>/dev/null | awk '/=>/ {print $3}'
}

declare -A seen
warn=()
for exe in "$BIN_DIR/ffmpeg.exe" "$BIN_DIR/ffprobe.exe"; do
    while IFS= read -r p; do
        [[ -z "$p" ]] && continue
        case "${p,,}" in
            */ucrt64/*) seen["$p"]=1 ;;             # bundle these
            /c/windows/*) : ;;                       # OS DLLs — skip silently
            *) warn["$p"]=1 ;;                       # anything else — flag
        esac
    done < <(ucrt_dlls "$exe")
done

if [[ ${#warn[@]} -gt 0 ]]; then
    echo "NOTE: these deps resolve outside /ucrt64 and /c/Windows — bundle via --extra if needed:" >&2
    printf '  %s\n' "${!warn[@]}" >&2
fi

rm -rf "$OUT"; mkdir -p "$OUT"
cp "$BIN_DIR/ffmpeg.exe" "$BIN_DIR/ffprobe.exe" "$OUT/"
for p in "${!seen[@]}"; do cp -- "$p" "$OUT/"; done
for e in "${EXTRAS[@]+"${EXTRAS[@]}"}"; do
    [[ -f "$e" ]] || { echo "Error: --extra '$e' not found" >&2; exit 1; }
    cp -- "$e" "$OUT/"
done

cat > "$OUT/README.txt" <<'EOF'
ffmpeg-gpu-npu-av1 — redistributable Windows x64 build (GPL v2+)

Self-contained: ffmpeg.exe / ffprobe.exe with all required runtime DLLs.
No MSYS2 install needed. Hardware codecs need the matching GPU driver at runtime:
  NVENC/NVDEC -> NVIDIA driver,  QSV -> Intel driver,  AMF -> AMD driver.
AV1 decode is native (libdav1d); AV1/HEVC/H.264 encode via NVENC/QSV/AMF + SVT-AV1.
See https://github.com/ikeno-web/ffmpeg-gpu-npu-av1
EOF

n=${#seen[@]}; x=${#EXTRAS[@]}
echo "Bundled $n ucrt64 DLL(s) + $x extra(s).  Total: $(du -sh "$OUT" | cut -f1)"
( cd "$OUT" && ls *.dll | sort )
echo "Output: $OUT"
