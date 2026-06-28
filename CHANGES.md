# ffmpeg-gpu-npu-av1 — Changes

This is a modified fork of [FFmpeg](https://github.com/FFmpeg/FFmpeg) (based on release **n7.1.5**).

## License

This project retains the original FFmpeg license (LGPL v2.1+ / GPL v2+).
All modifications are released under the same license terms.
See [LICENSE.md](LICENSE.md) for details.

## Enhancements

### Performance Optimizations

#### Hardware-accelerated transcode (NVENC / NVDEC / QSV / AMF) (Phase 4)
- Enabled vendor hardware video engines so encode/decode no longer burden the CPU:
  - NVIDIA: `*_cuvid` decoders, `h264/hevc/av1_nvdec` hwaccels, and
    `h264/hevc/av1_nvenc` encoders (via the `ffnvcodec` headers; no CUDA SDK
    runtime linkage, no `nvcc`).
  - Intel: `*_qsv` decoders/encoders and `scale_qsv`/`vpp_qsv` (via oneVPL).
  - AMD: `h264/hevc/av1_amf` encoders (via AMF headers).
- Added `scale_npp` (NVIDIA Performance Primitives) for GPU-resident scaling, so a
  decode → scale → encode chain can run entirely on the GPU
  (`-hwaccel cuda -hwaccel_output_format cuda -vf scale_npp=... -c:v h264_nvenc`)
  with no CPU↔GPU frame copies.
- Verified on RTX 4090: 4K→4K NVENC `p7` ≈ **2.8×** realtime vs `libx264 veryslow`
  at 0.95×; full-GPU 4K→1080p downscale ≈ **7.3×** realtime (220 fps).
- Known interaction: with `-hwaccel cuda`, pass `-threads 1`. NVDEC allocates one
  decode surface per frame-thread and fails above 32; the raised 64-thread cap from
  Phase 1 can exceed that. GPU decode makes CPU threads irrelevant anyway. CPU-only
  decode is unaffected.
- Note: `nvcc`-only CUDA filters (`scale_cuda`, `overlay_cuda`, `yadif_cuda`) are
  intentionally not built — `scale_npp` covers GPU scaling without the MSVC
  host-compiler complication that `--enable-cuda-nvcc` brings on MSYS2.
- Added `--enable-libvmaf` (+ `psnr`/`ssim`/`xpsnr` filters) for objective quality
  measurement. Used to verify `av1_nvenc` reaches the same VMAF as `h264_nvenc` at
  ~34 % lower bitrate (VMAF 90) — up to ~39 % at lower bitrates — on real 1080p
  content. `hevc_nvenc` saves ~20 %. AV1 NVENC is also faster than H.264 NVENC on
  Ada. AV1 *decoding* uses **libdav1d** (`--enable-libdav1d`) — native software AV1
  decode with no hardware required; `av1_cuvid`/`av1_qsv` remain for hardware decode.
- `tools/batch_transcode.sh`: throughput helper — runs hardware-encoder jobs
  serially (a single NVENC job already saturates the engine: 1→8 concurrent jobs
  gained only ~17 % aggregate fps on a 4090) and CPU-encoder jobs in parallel.
- Added `--enable-libsvtav1` (SVT-AV1 software AV1 encoder). Measured vs.
  `av1_nvenc` at 1080p: `libsvtav1 -preset 6` reaches VMAF 93 at 2 Mbps where
  `av1_nvenc` needs ~4 Mbps (≈half the bitrate at equal quality) but runs at ~2.9×
  realtime vs the hardware encoder's ~10.5×. Use hardware AV1 for speed, SVT-AV1
  preset 6 for compression.

#### CCD/NUMA-aware thread affinity (Phase 1)
- New `libavutil/cpu_topology.{c,h}`: detects CPU topology (CCD / L3 cache
  domains) on Windows (`GetLogicalProcessorInformationEx`) and Linux
  (`/sys` cache topology), with a safe single-domain fallback.
- New `libavutil/thread_affinity.{c,h}`: cross-platform thread pinning to a
  CCD (Windows `SetThreadAffinityMask`, Linux `pthread_setaffinity_np`).
- Frame-thread and slice-thread worker pools are now pinned per-CCD to keep
  each worker's working set within one L3 cache domain. Enabled automatically
  when more than one CCD is detected; otherwise behaviour is unchanged.
- Raised the automatic frame-thread cap from 16 to 64 for high core-count
  CPUs (slice-thread auto cap stays at 16 to avoid the historical H.264
  slice-threading issue).
- New public API `av_cpu_force_numa_aware(int mode)` and CLI option
  `-numa_aware <-1|0|1>` (auto / off / on) to control the behaviour.
- Verified end-to-end on AMD Ryzen 9950X (2 CCDs, 8 cores / 16 threads each,
  32 MB L3 per CCD): topology detection, per-CCD pinning, and the on/off/auto
  toggle all confirmed; ~7% faster mpeg4 1080p encode with pinning enabled
  (larger gains expected on cache-heavy codecs).

### Custom Filters

#### Vendor-neutral GPU compute dispatch (Phase 2)
- New `libavutil/gpu_compute.{c,h}`: runtime probe/selection of GPU compute
  backends (Vulkan / OpenCL / D3D12) via `av_hwdevice_ctx_create`, with a
  configurable priority (default Vulkan > OpenCL > D3D12).
- New `*_gpu` meta-filters resolved in the graph parser: a filter named
  `<base>_gpu` (e.g. `scale_gpu`, `avgblur_gpu`, `transpose_gpu`) is mapped
  automatically to the best available backend variant
  (`<base>_vulkan` / `<base>_opencl`), with graceful fallback when the
  preferred backend or a given variant is unavailable. Existing backend
  filters are untouched.
- New public API `av_gpu_compute_set_preferred()` and CLI option
  `-gpu_backend vulkan|opencl|d3d12|auto`.
- Verified on RTX 4090 (Vulkan 1.4): `avgblur_gpu`/`scale_gpu`/`transpose_gpu`
  resolve to the Vulkan variants, compile compute shaders, and run; explicit
  `-gpu_backend opencl` with no OpenCL runtime falls back to Vulkan.

### Codec Improvements

#### ONNX Runtime DNN backend with DirectML/NPU offload (Phase 3)
- New `libavfilter/dnn/dnn_backend_onnxrt.c`: a DNN backend built on the
  ONNX Runtime C API, selectable as `dnn_backend=onnxruntime` on the DNN
  filters (e.g. `dnn_processing`).
- Execution providers via `execution_provider=cpu|directml|cuda`. The
  DirectML EP (for NPU/GPU offload on Windows) is compiled in only when the
  ONNX Runtime build provides `dml_provider_factory.h`
  (`HAVE_ONNXRUNTIME_DML`); otherwise it warns and falls back to the CPU EP.
- Resilient to ONNX Runtime version skew: requests the build's API version
  and falls back to the newest the loaded runtime supports (verified with
  1.26 headers against a 1.17 runtime DLL).
- New configure option `--enable-libonnxruntime`.
- Verified end-to-end on CPU EP (real ONNX model, frame-processing pipeline);
  DirectML path verified to fall back cleanly where DML is absent. (Note: the
  desktop Ryzen 9950X has no NPU; on-NPU execution requires Ryzen AI / Intel
  Core Ultra / Snapdragon X hardware with a DirectML-enabled ONNX Runtime.)
- Backend exposed on `dnn_processing`, and extended to `sr`
  (super-resolution) and `derain`/`dehaze`. NPU 2x super-resolution verified
  end-to-end via DirectML (320x240 -> 640x480).
- Object detection (`dnn_detect`) supported via ONNX Runtime: the backend now
  handles multiple output tensors and the `DFT_ANALYTICS_DETECT` function type
  (NHWC uint8 input), routing SSD/YOLO outputs through the same post-processing
  as OpenVINO. Verified end-to-end on DirectML with an SSD-style model
  (bounding boxes emitted as frame side-data).
- Asynchronous, pipelined inference: with `async=1` the backend runs inference
  on detached threads (ORT `Run()` is thread-safe), and `nireq` (default 2)
  keeps several frames in flight so inference overlaps with decode/encode.
  Throughput benefit scales with model weight; the speedup is largest for
  heavy models where inference time dominates per-frame thread overhead.
- Fixed an input-buffer leak: tensors created with
  `CreateTensorWithDataAsOrtValue` do not own their data, so the backing
  buffer is now tracked per request and freed on recycle.

## Building

```bash
./configure --enable-gpl --enable-nonfree
make -j$(nproc)
```

## Upstream

- Upstream repository: https://github.com/FFmpeg/FFmpeg
- Based on tag: n7.1.5
