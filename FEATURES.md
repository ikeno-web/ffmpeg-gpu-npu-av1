# ffmpeg-gpu-npu-av1 — Feature Guide

An enhanced fork of **FFmpeg n7.1.5** focused on modern hardware:
multi-CCD CPUs, any-vendor GPUs, and NPUs. Same LGPL/GPL license as upstream.

| Area | What it adds | Key option |
|------|--------------|-----------|
| CPU / threads | CCD/NUMA-aware thread pinning for chiplet CPUs | `-numa_aware` |
| GPU | Vendor-neutral compute dispatch (Vulkan/OpenCL/D3D12) | `-gpu_backend`, `*_gpu` filters |
| HW transcode | NVENC/NVDEC + QSV + AMF hardware encode/decode, full-GPU pipeline | `-c:v h264_nvenc`, `-hwaccel cuda` |
| NPU / ML | ONNX Runtime backend with DirectML offload | `dnn_backend=onnxruntime:execution_provider=directml` |

---

## 0. Hardware-accelerated transcode (NVENC / NVDEC / QSV / AMF)

The build ships with vendor hardware encoders and decoders so the GPU's
dedicated video engines do the encode/decode work instead of the CPU.

| Vendor | Decoders | Encoders | Scale filter |
|--------|----------|----------|--------------|
| NVIDIA | `*_cuvid`, `*_nvdec` hwaccel | `h264_nvenc`, `hevc_nvenc`, `av1_nvenc` | `scale_npp` |
| Intel  | `*_qsv` | `h264_qsv`, `hevc_qsv`, `av1_qsv` | `scale_qsv`, `vpp_qsv` |
| AMD    | (d3d11va) | `h264_amf`, `hevc_amf`, `av1_amf` | — |

```bash
# Simple: GPU encode (CPU decodes, NVENC encodes)
ffmpeg -i in.mp4 -c:v h264_nvenc -preset p7 out.mp4

# Full-GPU pipeline: NVDEC decodes -> scale_npp scales -> NVENC encodes,
# the frame never leaves the GPU (no CPU<->GPU copy).
ffmpeg -threads 1 -hwaccel cuda -hwaccel_output_format cuda -i in.mp4 \
  -vf scale_npp=1920:1080 -c:v h264_nvenc -preset p7 out.mp4
```

**Measured on RTX 4090 (4K source):**
| Job | CPU `libx264` | `h264_nvenc` | Full-GPU pipeline |
|-----|---------------|--------------|-------------------|
| 4K → 4K (quality preset) | 0.95× realtime (29 fps) | **2.81×** (85 fps) | — |
| 4K → 1080p downscale | — | — | **7.28×** (220 fps) |

> **Important — `-threads 1` with `-hwaccel cuda`:** this fork raises the
> automatic frame-thread cap to 64 (see §1). NVDEC allocates a decode surface
> per thread and fails above 32 surfaces (`CUDA_ERROR_INVALID_VALUE`). When
> hardware-decoding, the GPU does the decode, so CPU threads add nothing — pass
> `-threads 1` to keep the surface count in range. (CPU-only decode is
> unaffected.)

### Batch transcoding — parallelize the CPU, serialize the GPU

A common instinct is to run many transcode jobs in parallel for throughput.
Whether that helps depends entirely on the encoder:

**Measured on RTX 4090 (4K → 1080p `scale_npp` + `h264_nvenc`), aggregate fps:**
| Concurrent jobs | 1 | 2 | 4 | 6 | 8 |
|-----------------|---|---|---|---|---|
| Aggregate fps | 201 | 226 | 233 | 234 | **236** |

A single NVENC job already saturates the GPU's encode engine — running 8 in
parallel raises throughput by only ~17 % while consuming 8× the VRAM and encode
sessions. **Hardware encoders should be run serially.** CPU encoders
(`libx264`/`libx265`) are the opposite: throughput scales with cores, so they
*should* be run in parallel (and benefit from the CCD pinning in §1).

`tools/batch_transcode.sh` encodes this rule — it auto-detects the encoder and
runs hardware jobs serially, CPU jobs in parallel:
```bash
# GPU: serial (engine-bound)
FFMPEG=./ffmpeg.exe tools/batch_transcode.sh -i in/ -o out/ -c h264_nvenc -a "-preset p5 -cq 26"
# CPU: parallel across CCDs
FFMPEG=./ffmpeg.exe tools/batch_transcode.sh -i in/ -o out/ -c libx264 -a "-preset medium -crf 20" -j 8
```

### AV1 hardware encoding — ~⅓ smaller files at equal quality

The build includes `av1_nvenc` (and `hevc_nvenc`). On an Ada GPU (RTX 4090)
these are *faster* than `h264_nvenc` and produce substantially smaller files at
the same quality.

**Measured** — Big Buck Bunny 1080p, NVENC `-preset p6`, quality = VMAF
(libvmaf, model v0.6.1) vs the source:

| Bitrate | H.264 VMAF | HEVC VMAF | AV1 VMAF |
|---------|-----------:|----------:|---------:|
| ~1 Mbps | 65.6 | 74.9 | **78.1** |
| ~2 Mbps | 81.4 | 85.6 | **87.8** |
| ~4 Mbps | 89.9 | 91.9 | **92.9** |
| ~8 Mbps | 94.2 | 95.3 | **95.7** |

Bitrate needed for the **same** quality (log-interpolated from the table):

| Target | H.264 | HEVC | AV1 |
|--------|-------|------|-----|
| VMAF 85 | baseline | −29 % | **−39 %** |
| VMAF 90 | baseline | −20 % | **−34 %** |

So `av1_nvenc` reaches the same perceptual quality as `h264_nvenc` at roughly
**one-third less bitrate** — biggest at low bitrates, where it matters most.

```bash
# AV1 at a quality target (CQ), hardware speed:
ffmpeg -i in.mp4 -c:v av1_nvenc -preset p6 -rc vbr -cq 28 -c:a copy out.mp4
```

> Gains here use the *hardware* encoder. Re-verified on **live-action** content
> (Xiph `foreman`, 720p) the advantage is *larger*, not smaller: AV1 reaches
> VMAF 93 at ~1 Mbps where H.264 needs ~2 Mbps — roughly **−50 % bitrate** at
> equal quality (vs −34 % on the animated clip). So the animation numbers are
> conservative; AV1 helps most on the detailed, high-motion footage H.264
> struggles with.

> **Decoding AV1:** the build includes **libdav1d** (VideoLAN's AV1 decoder), so
> AV1 decodes natively in software — no hardware decoder required. `libdav1d` is
> the default AV1 decoder; `av1_cuvid` (NVDEC) and `av1_qsv` remain available for
> hardware decode. (FFmpeg's *built-in* `av1` decoder is incomplete and is not
> used.)

### Hardware vs. software AV1 — `av1_nvenc` vs `libsvtav1`

The build also includes the **SVT-AV1** software encoder (`libsvtav1`). Hardware
and software AV1 sit at opposite ends of the speed/quality curve.

**Measured** — 1080p, same target bitrate, VMAF vs. source (RTX 4090 / Ryzen 9950X):

| Encoder | Preset | 2 Mbps VMAF | 4 Mbps VMAF | Speed |
|---------|--------|------------:|------------:|------:|
| `av1_nvenc` (HW) | p5 | 88.5 | 93.3 | **10.5×** |
| `libsvtav1` (SW) | 10 (fast) | 88.2 | — | 7.8× |
| `libsvtav1` (SW) | 6 (quality) | **93.1** | **95.6** | 2.9× |

Reading the table:
- **For speed / realtime → `av1_nvenc`.** The hardware engine runs ~10× realtime
  at any bitrate; nothing software matches it.
- **For maximum compression → `libsvtav1 -preset 6`.** It reaches VMAF 93 at
  2 Mbps where `av1_nvenc` needs ~4 Mbps — roughly **half the bitrate at equal
  quality** — but at ~⅓ the speed.
- Fast software (`-preset 10`) is the worst of both: slower than hardware with no
  quality gain. The real choice is *HW for speed* vs *SW preset 6 for quality*.

```bash
# fastest (hardware):
ffmpeg -i in.mp4 -c:v av1_nvenc -preset p5 -b:v 2M out.mp4
# smallest at equal quality (software, ~3.6x slower):
ffmpeg -i in.mp4 -c:v libsvtav1 -preset 6 -b:v 2M out.mp4
```

### NVENC codec ladder — which hardware encoder to pick

Combining the quality and speed measurements above (1080p, RTX 4090, NVENC):

| Encoder | Bitrate vs H.264¹ | Encode speed² | Pick when |
|---------|------------------:|--------------:|-----------|
| `h264_nvenc` | baseline | 8.4× | maximum device compatibility |
| `hevc_nvenc` | −20 % | 5.7× | smaller files, broad HW HEVC support |
| `av1_nvenc` | **−34 %** | **9.6×** | AV1 playback available (newest, best) |

¹ Bitrate for equal quality (VMAF 90), from the AV1 efficiency table above.
² Realtime multiple at preset p6, 4 Mbps.

On Ada, **`av1_nvenc` is both the most efficient *and* the fastest** of the three —
the obvious default wherever AV1 decode is supported (browsers, recent TVs/phones).
`hevc_nvenc` is the middle ground for hardware that lacks AV1; `h264_nvenc` remains
the universal-compatibility baseline.

### Target-VMAF auto-tuning — `tools/vmaf_tune.py`

Instead of guessing a CRF/CQ, let the tool *find* the smallest-file setting that
still meets a quality target. It binary-searches the quality parameter, measuring
each step with this build's libvmaf, and reports (optionally producing) the encode.

```bash
FFMPEG=./ffmpeg.exe python tools/vmaf_tune.py -i in.mp4 -o out.mp4 \
    --encoder av1_nvenc --target-vmaf 93 --preset p6
```
It returns the **largest** q whose VMAF ≥ target (smallest file meeting quality).
Example on Xiph `foreman` 720p, `av1_nvenc`, target VMAF 93: it picked `-cq 40`
(≈1.07 Mbps) — vs the naive high-quality `-cq 20` at ≈9 Mbps for the same
perceptual quality, an **8.5× smaller file**. Works with `av1_nvenc`/`hevc_nvenc`/
`h264_nvenc` (`-cq`) and `libsvtav1`/`libx264`/`libx265` (`-crf`); a short
`--sample-seconds` keeps the search fast, then `-o` does the full-length encode.

---

## 1. CCD / NUMA-aware threading

Pins frame/slice worker threads to a single CCD (L3 cache domain) so each
worker's working set stays in one cache — a measurable win on chiplet CPUs
such as AMD Ryzen 9000.

```bash
# auto (default): on when >1 CCD is detected
ffmpeg -i in.mp4 -c:v libx264 -threads 0 out.mp4

# force on / off (for A/B benchmarking)
ffmpeg -numa_aware 1 -i in.mp4 ... out.mp4
ffmpeg -numa_aware 0 -i in.mp4 ... out.mp4
```
`-v verbose` prints the detected topology:
```
CPU topology: 2 CCD(s), 16 physical cores, 32 logical CPUs, SMT=2
  CCD 0: 8 cores, 16 threads, L3=32768KB, mask=0x000000000000ffff
  CCD 1: 8 cores, 16 threads, L3=32768KB, mask=0x00000000ffff0000
```

## 2. Vendor-neutral GPU compute

A `<base>_gpu` filter name resolves at runtime to the best available backend
variant (`_vulkan` / `_opencl`), so the same command line works on NVIDIA,
AMD and Intel GPUs.

```bash
# auto-pick backend (priority: vulkan > opencl > d3d12)
ffmpeg -init_hw_device vulkan=vk -filter_hw_device vk -i in.mp4 \
  -vf "format=nv12,hwupload,scale_gpu=w=1920:h=1080,hwdownload,format=nv12" out.mp4

# force a backend
ffmpeg -gpu_backend opencl -i in.mp4 -vf "...,avgblur_gpu=sizeX=5,..." out.mp4
```
Available meta-filters: `scale_gpu`, `avgblur_gpu`, `transpose_gpu`,
`overlay_gpu`, `nlmeans_gpu` (resolve where a backend variant exists).

## 3. NPU / ML inference (ONNX Runtime + DirectML)

> **For super-resolution / denoise, prefer the standalone [npuscale](https://github.com/ikeno-web/npuscale) tool.**
> The in-graph `dnn_backend=onnxruntime` route below works, but FFmpeg's DNN
> framework spawns a thread per inference and limits optimization. npuscale pipes
> raw frames to/from stock ffmpeg and drives ONNX Runtime + DirectML directly, so
> it gets full control — worker pool, IoBinding, FP16, tiling, and temporal
> (multi-frame) models — and is faster. Use the in-fork backend only when you need
> the DNN step *inside* a single ffmpeg filter graph (e.g. `dnn_detect` side-data).

Run DNN filters on the ONNX Runtime, optionally offloaded to an NPU/GPU via
the DirectML execution provider.

```bash
# generic frame processing on the NPU
ffmpeg -i in.mp4 -vf \
  "format=rgb24,dnn_processing=dnn_backend=onnxruntime:model=model.onnx:execution_provider=directml" \
  out.mp4

# NPU super-resolution
ffmpeg -i in.mp4 -vf \
  "format=yuv420p,sr=dnn_backend=onnxruntime:model=sr.onnx:execution_provider=directml" \
  out.mp4
```
Filters supporting `dnn_backend=onnxruntime`: `dnn_processing`, `sr`, `derain`, `dnn_detect`.

**Pipelined inference:** add `async=1` to run inference on background threads and
`nireq=N` (default 2) to keep several frames in flight, overlapping inference
with decode/encode. Example:
`dnn_processing=dnn_backend=onnxruntime:model=m.onnx:execution_provider=directml:async=1:nireq=4`
Execution providers: `cpu` (always), `directml` (NPU/GPU, Windows),
`cuda`. Verify NPU usage in **Task Manager → Performance → NPU**.

A prebuilt, self-contained Windows x64 package with a DirectML-enabled ONNX
Runtime (plus test models and benchmark scripts) is attached to the
[latest release](../../releases/latest).

---

## Building

See [INSTALL.md](INSTALL.md) for the standard FFmpeg build. Extra options
introduced by this fork:

```
--enable-libonnxruntime    # ONNX Runtime DNN backend (DirectML auto-detected)
```
GPU compute uses the usual `--enable-vulkan` / `--enable-opencl`
(`--enable-libshaderc` for Vulkan compute filters). CCD affinity has no extra
dependency.

### Full-featured build (Windows / MSYS2 ucrt64)

Install dependencies first:
```bash
pacman -S --needed \
  mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-nasm make diffutils pkgconf \
  mingw-w64-ucrt-x86_64-x264 mingw-w64-ucrt-x86_64-libass mingw-w64-ucrt-x86_64-fdk-aac \
  mingw-w64-ucrt-x86_64-shaderc mingw-w64-ucrt-x86_64-vulkan-headers mingw-w64-ucrt-x86_64-vulkan-loader \
  mingw-w64-ucrt-x86_64-opencl-headers mingw-w64-ucrt-x86_64-opencl-icd \
  mingw-w64-ucrt-x86_64-ffnvcodec-headers mingw-w64-ucrt-x86_64-libvpl mingw-w64-ucrt-x86_64-amf-headers \
  mingw-w64-ucrt-x86_64-vmaf mingw-w64-ucrt-x86_64-svt-av1 mingw-w64-ucrt-x86_64-dav1d
```
For NVIDIA NPP (`scale_npp`, GPU-resident scaling) you also need the CUDA
Toolkit installed (provides the NPP libraries); the configure below points at
CUDA 12.6 via the 8.3 short path (spaces in `Program Files` break the linker).

Configure with all features enabled:
```bash
export TMP=/tmp TEMP=/tmp TMPDIR=/tmp   # required on Windows

./configure \
  --cc=gcc \
  --enable-gpl --enable-nonfree \
  --enable-libx264 --enable-libass --enable-libfdk-aac \
  --enable-libonnxruntime --enable-libvmaf --enable-libsvtav1 --enable-libdav1d \
  --enable-vulkan --enable-libshaderc --enable-opencl \
  --enable-ffnvcodec --enable-cuvid --enable-nvenc --enable-nvdec \
  --enable-libvpl --enable-amf --enable-libnpp \
  --extra-cflags=-I/c/PROGRA~1/NVIDIA~2/CUDA/v12.6/include \
  --extra-ldflags=-L/c/PROGRA~1/NVIDIA~2/CUDA/v12.6/lib/x64 \
  --enable-protocol=file,pipe,data \
  --enable-demuxer=mov,matroska,avi,mpegts,rawvideo,flv,ogg,wav,mp3,aac,flac,yuv4mpegpipe \
  --enable-muxer=mp4,matroska,avi,mpegts,rawvideo,ogg,null,flv,mp3,adts,flac,wav \
  --enable-decoder=h264,hevc,vp8,vp9,av1,mpeg4,aac,mp3,ac3,opus,vorbis,flac,pcm_s16le,rawvideo,wrapped_avframe,ass,ssa,libdav1d,h264_cuvid,hevc_cuvid,av1_cuvid,vp9_cuvid,h264_qsv,hevc_qsv,av1_qsv \
  --enable-encoder=libx264,libfdk_aac,aac,rawvideo,wrapped_avframe,mpeg4,mp3,h264_nvenc,hevc_nvenc,av1_nvenc,h264_qsv,hevc_qsv,av1_qsv,h264_amf,hevc_amf,av1_amf,libsvtav1 \
  --enable-hwaccel=h264_nvdec,hevc_nvdec,av1_nvdec,vp9_nvdec,h264_d3d11va,hevc_d3d11va,av1_d3d11va \
  --enable-filter=scale,format,ass,subtitles,amix,aresample,amerge,volume,aformat,overlay,crop,pad,vflip,hflip,transpose,rotate,trim,atrim,concat,split,asplit,fps,setpts,null,anull,dnn_processing,sr,derain,dnn_detect,scale_npp,scale_qsv,vpp_qsv,hwupload,hwupload_cuda,hwdownload,psnr,ssim,xpsnr,libvmaf \
  --enable-filter=avgblur_vulkan,scale_vulkan,transpose_vulkan,overlay_vulkan,nlmeans_vulkan \
  --enable-indev=lavfi \
  --disable-doc

make -j$(nproc)
```

This produces an `ffmpeg.exe` with H.264 (libx264), AAC (libfdk_aac), ASS subtitles,
audio mixing (amix), Vulkan/OpenCL GPU filters, NVENC/NVDEC + QSV + AMF hardware
transcode with a full-GPU `scale_npp` pipeline, and ONNX Runtime NPU inference all
in one binary.

> CUDA filters that require `nvcc` (`scale_cuda`, `overlay_cuda`, `yadif_cuda`)
> are **not** enabled here — they need `--enable-cuda-nvcc`, which on MSYS2 pulls
> in the MSVC host compiler and complicates the build. `scale_npp` covers
> GPU-resident scaling without `nvcc`.

### ⚠️ Redistributable vs. personal build

The recipe above uses `--enable-nonfree` (for **fdk-aac**) and **libnpp**
(`scale_npp`). FFmpeg's configure flags *both* as nonfree, so it prints
**"License: nonfree and unredistributable"** — that binary is fine for personal
use but **cannot be legally redistributed**.

To build a **GPL‑redistributable** binary (for publishing prebuilt downloads),
drop the three nonfree pieces:

| Remove | Replace with |
|--------|--------------|
| `--enable-nonfree` | (just `--enable-gpl`) |
| `--enable-libfdk-aac` + `libfdk_aac` encoder | the native **`aac`** encoder (already enabled) |
| `--enable-libnpp` + `scale_npp` filter | **`scale_vulkan`** (or `scale_cuda` with `--enable-cuda-nvcc`) for GPU scaling |

Everything else — NVENC/NVDEC, QSV, AMF, SVT‑AV1, libdav1d, libvmaf, libx264,
ONNX Runtime, Vulkan/OpenCL filters — stays. Configure then prints
**"License: GPL version 2 or later"** and the binary is redistributable.
(The only functional losses are fdk-aac, marginally better than native aac, and
the `scale_npp` GPU scaler — use `scale_vulkan` for a vendor‑neutral GPU scale.)

For full details of every change, see [CHANGES.md](CHANGES.md).
