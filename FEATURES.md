# FFmpeg Plus — Feature Guide

An enhanced fork of **FFmpeg n7.1.5** focused on modern hardware:
multi-CCD CPUs, any-vendor GPUs, and NPUs. Same LGPL/GPL license as upstream.

| Area | What it adds | Key option |
|------|--------------|-----------|
| CPU / threads | CCD/NUMA-aware thread pinning for chiplet CPUs | `-numa_aware` |
| GPU | Vendor-neutral compute dispatch (Vulkan/OpenCL/D3D12) | `-gpu_backend`, `*_gpu` filters |
| NPU / ML | ONNX Runtime backend with DirectML offload | `dnn_backend=onnxruntime:execution_provider=directml` |

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

For full details of every change, see [CHANGES.md](CHANGES.md).
