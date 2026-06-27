#!/usr/bin/env bash
#
# batch_transcode.sh — throughput-aware batch transcoder for ffmpeg-plus
#
# Applies the verified best practice (see FEATURES.md "Batch transcoding"):
#   * Hardware encoders (*_nvenc / *_qsv / *_amf): a single job already
#     saturates the GPU's video engine, so jobs are run SERIALLY. Running them
#     in parallel does not raise throughput and only wastes VRAM / sessions.
#   * CPU encoders (libx264 / libx265 / ...): throughput scales with cores, so
#     jobs are run in PARALLEL (default: one job per CCD-sized worker group).
#
# Usage:
#   batch_transcode.sh -i INPUT_DIR -o OUTPUT_DIR [options]
#
# Options:
#   -i DIR        input directory (processes *.mp4 *.mkv *.mov *.avi)
#   -o DIR        output directory
#   -c ENCODER    video encoder (default: h264_nvenc)
#   -a ARGS       extra ffmpeg output args, quoted (e.g. "-preset p5 -cq 28")
#   -j N          parallel jobs for CPU encoders (default: auto = nproc/4)
#   -e EXT        output container extension (default: mp4)
#   -n            dry-run: print the commands without executing
#
# Examples:
#   # GPU (serial, saturates NVENC):
#   batch_transcode.sh -i in/ -o out/ -c h264_nvenc -a "-preset p5 -cq 26"
#   # CPU (parallel across CCDs):
#   batch_transcode.sh -i in/ -o out/ -c libx264 -a "-preset medium -crf 20" -j 8

set -euo pipefail

FFMPEG="${FFMPEG:-ffmpeg}"
IN=""; OUT=""; ENC="h264_nvenc"; EXTRA=""; JOBS=""; EXT="mp4"; DRY=0

while getopts "i:o:c:a:j:e:n" opt; do
  case "$opt" in
    i) IN="$OPTARG" ;;
    o) OUT="$OPTARG" ;;
    c) ENC="$OPTARG" ;;
    a) EXTRA="$OPTARG" ;;
    j) JOBS="$OPTARG" ;;
    e) EXT="$OPTARG" ;;
    n) DRY=1 ;;
    *) echo "see header for usage" >&2; exit 1 ;;
  esac
done

[ -n "$IN" ] && [ -n "$OUT" ] || { echo "Error: -i and -o are required" >&2; exit 1; }
mkdir -p "$OUT"

# Is this a hardware encoder? -> run serially.
case "$ENC" in
  *_nvenc|*_qsv|*_amf|*_vaapi|*_mf) HW=1 ;;
  *) HW=0 ;;
esac

# Default CPU parallelism: nproc/4 (leave headroom; each job gets a few threads).
if [ -z "$JOBS" ]; then
  NPROC="$(nproc 2>/dev/null || echo 4)"
  JOBS=$(( NPROC / 4 )); [ "$JOBS" -lt 1 ] && JOBS=1
fi

shopt -s nullglob nocaseglob
mapfile -t FILES < <(printf '%s\n' "$IN"/*.mp4 "$IN"/*.mkv "$IN"/*.mov "$IN"/*.avi)
shopt -u nullglob nocaseglob

[ "${#FILES[@]}" -gt 0 ] || { echo "No input videos found in $IN" >&2; exit 1; }

emit() {  # $1 = input file
  local f="$1" base out
  base="$(basename "$f")"; base="${base%.*}"
  out="$OUT/$base.$EXT"
  if [ "$HW" = 1 ]; then
    # GPU decode+encode; -threads 1 avoids the NVDEC >32-surface failure.
    echo "$FFMPEG -hide_banner -threads 1 -hwaccel auto -i \"$f\" -c:v $ENC $EXTRA -c:a copy -y \"$out\""
  else
    echo "$FFMPEG -hide_banner -i \"$f\" -c:v $ENC $EXTRA -c:a copy -y \"$out\""
  fi
}

if [ "$HW" = 1 ]; then
  echo "Hardware encoder ($ENC): running ${#FILES[@]} job(s) SERIALLY (engine-bound)." >&2
  for f in "${FILES[@]}"; do
    cmd="$(emit "$f")"
    if [ "$DRY" = 1 ]; then echo "$cmd"; else eval "$cmd"; fi
  done
else
  echo "CPU encoder ($ENC): running ${#FILES[@]} job(s), $JOBS in PARALLEL." >&2
  for f in "${FILES[@]}"; do emit "$f"; done \
    | if [ "$DRY" = 1 ]; then cat; else xargs -P "$JOBS" -I {} bash -c '{}'; fi
fi

echo "Done. Output in: $OUT" >&2
