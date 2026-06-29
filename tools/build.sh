#!/usr/bin/env bash
#
# build.sh — one-command build of ffmpeg-gpu-npu-av1 (MSYS2 / ucrt64).
# Run from the repo root inside an MSYS2 ucrt64 bash shell.
#
#   ./tools/build.sh                  # full/personal build (fdk-aac + scale_npp; nonfree)
#   ./tools/build.sh --redistributable  # GPL-redistributable build (no fdk-aac / libnpp)
#   ./tools/build.sh --jobs 16
#
set -euo pipefail

# ── Critical environment (MSYS2 ucrt64) ──────────────────────────────
export PATH=/ucrt64/bin:/usr/bin:$PATH
export PKG_CONFIG_PATH=/ucrt64/lib/pkgconfig:/ucrt64/share/pkgconfig
export TMP=/tmp TEMP=/tmp TMPDIR=/tmp   # Windows TEMP is non-writable -> gcc fails without this

MODE="personal"
JOBS="$(nproc 2>/dev/null || echo 4)"
DRYRUN=0

usage() {
    cat <<EOF
Usage: ./tools/build.sh [--redistributable] [--jobs N] [--dry-run]
  --redistributable   GPL-redistributable build (drops nonfree fdk-aac and libnpp)
  --jobs N            parallel make jobs (default: nproc = ${JOBS})
  --dry-run           print the configure command and exit (no install/build)
  -h, --help          this help
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --redistributable) MODE="redistributable"; shift ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --dry-run) DRYRUN=1; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── CUDA Toolkit detection (libnpp, personal build only) ─────────────
# Returns a SPACE-FREE 8.3 short path (spaces in "Program Files" break the
# linker). cygpath -d gives the DOS short name; -u converts back to /c/... form.
detect_cuda() {
    local base="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA"
    local d newest=""
    for d in "$base"/v*; do
        [[ -d "$d" ]] || continue
        [[ -z "$newest" || "$d" > "$newest" ]] && newest="$d"
    done
    [[ -z "$newest" ]] && return 1
    if command -v cygpath >/dev/null 2>&1; then
        cygpath -u "$(cygpath -d "$newest")"   # e.g. /c/PROGRA~1/NVIDIA~2/CUDA/v12.6
    else
        echo "$newest"
    fi
}

# ── Dependencies ─────────────────────────────────────────────────────
if [[ "$DRYRUN" == 0 ]]; then
echo "==> Installing dependencies (pacman --needed) ..."
pacman -S --noconfirm --needed \
    mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-nasm make diffutils pkgconf \
    mingw-w64-ucrt-x86_64-x264 mingw-w64-ucrt-x86_64-libass \
    mingw-w64-ucrt-x86_64-shaderc mingw-w64-ucrt-x86_64-vulkan-headers \
    mingw-w64-ucrt-x86_64-vulkan-loader mingw-w64-ucrt-x86_64-opencl-headers \
    mingw-w64-ucrt-x86_64-opencl-icd mingw-w64-ucrt-x86_64-ffnvcodec-headers \
    mingw-w64-ucrt-x86_64-libvpl mingw-w64-ucrt-x86_64-amf-headers \
    mingw-w64-ucrt-x86_64-vmaf mingw-w64-ucrt-x86_64-svt-av1 mingw-w64-ucrt-x86_64-dav1d
[[ "$MODE" == "personal" ]] && \
    pacman -S --noconfirm --needed mingw-w64-ucrt-x86_64-fdk-aac
fi

echo ""
echo "==> Mode: $MODE   (jobs: $JOBS)"

# ── Encoder / filter lists (mode-dependent extras) ───────────────────
ENCODERS="libx264,aac,rawvideo,wrapped_avframe,mpeg4,mp3,h264_nvenc,hevc_nvenc,av1_nvenc,h264_qsv,hevc_qsv,av1_qsv,h264_amf,hevc_amf,av1_amf,libsvtav1"
FILTERS_DNN="scale,format,ass,subtitles,amix,aresample,amerge,volume,aformat,overlay,crop,pad,vflip,hflip,transpose,rotate,trim,atrim,concat,split,asplit,fps,setpts,null,anull,dnn_processing,sr,derain,dnn_detect,scale_qsv,vpp_qsv,hwupload,hwupload_cuda,hwdownload,psnr,ssim,xpsnr,libvmaf"
FILTERS_VULKAN="avgblur_vulkan,scale_vulkan,transpose_vulkan,overlay_vulkan,nlmeans_vulkan"

FLAGS=(
    --cc=gcc --enable-gpl
    --enable-libx264 --enable-libass
    --enable-libonnxruntime --enable-libvmaf --enable-libsvtav1 --enable-libdav1d
    --enable-vulkan --enable-libshaderc --enable-opencl
    --enable-ffnvcodec --enable-cuvid --enable-nvenc --enable-nvdec
    --enable-libvpl --enable-amf
    --enable-protocol=file,pipe,data
    --enable-demuxer=mov,matroska,avi,mpegts,rawvideo,flv,ogg,wav,mp3,aac,flac,yuv4mpegpipe
    --enable-muxer=mp4,matroska,avi,mpegts,rawvideo,ogg,null,flv,mp3,adts,flac,wav
    --enable-decoder=h264,hevc,vp8,vp9,av1,mpeg4,aac,mp3,ac3,opus,vorbis,flac,pcm_s16le,rawvideo,wrapped_avframe,ass,ssa,libdav1d,h264_cuvid,hevc_cuvid,av1_cuvid,vp9_cuvid,h264_qsv,hevc_qsv,av1_qsv
    --enable-hwaccel=h264_nvdec,hevc_nvdec,av1_nvdec,vp9_nvdec,h264_d3d11va,hevc_d3d11va,av1_d3d11va
    --enable-indev=lavfi
    --disable-doc
)

if [[ "$MODE" == "personal" ]]; then
    CUDA="$(detect_cuda)" || { echo "ERROR: CUDA Toolkit not found; use --redistributable." >&2; exit 1; }
    echo "==> CUDA: $CUDA"
    ENCODERS="$ENCODERS,libfdk_aac"
    FILTERS_DNN="$FILTERS_DNN,scale_npp"
    FLAGS+=(--enable-nonfree --enable-libfdk-aac --enable-libnpp
            --extra-cflags="-I$CUDA/include" --extra-ldflags="-L$CUDA/lib/x64")
fi

FLAGS+=(--enable-encoder="$ENCODERS"
        --enable-filter="$FILTERS_DNN"
        --enable-filter="$FILTERS_VULKAN")

if [[ "$DRYRUN" == 1 ]]; then
    echo ""
    echo "./configure ${FLAGS[*]}"
    exit 0
fi

# ── Configure (capture stdout to verify the License line) ────────────
echo ""
echo "==> ./configure ..."
CONF_OUT="$(mktemp)"
if ! ./configure "${FLAGS[@]}" 2>&1 | tee "$CONF_OUT"; then
    echo "ERROR: configure failed." >&2
    exit 1
fi

LICENSE="$(grep -i '^License:' "$CONF_OUT" || true)"
echo "==> $LICENSE"
if [[ "$MODE" == "personal" ]]; then
    grep -qi 'nonfree' <<<"$LICENSE" || { echo "ERROR: expected nonfree license." >&2; exit 1; }
else
    grep -qi 'GPL version 2 or later' <<<"$LICENSE" || {
        echo "ERROR: expected redistributable GPL license, got: $LICENSE" >&2; exit 1; }
fi

# ── Build ────────────────────────────────────────────────────────────
echo ""
echo "==> make -j$JOBS ..."
make -j"$JOBS"

[[ -f ./ffmpeg.exe ]] || { echo "ERROR: ffmpeg.exe missing after build." >&2; exit 1; }
echo ""
echo "==> Build OK: $(realpath ./ffmpeg.exe)"
./ffmpeg.exe -version | head -n 1
