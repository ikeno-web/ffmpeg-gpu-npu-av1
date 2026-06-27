# FFmpeg Plus - Changes

This is a modified fork of [FFmpeg](https://github.com/FFmpeg/FFmpeg) (based on release **n7.1.5**).

## License

This project retains the original FFmpeg license (LGPL v2.1+ / GPL v2+).
All modifications are released under the same license terms.
See [LICENSE.md](LICENSE.md) for details.

## Enhancements

### Performance Optimizations

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

## Building

```bash
./configure --enable-gpl --enable-nonfree
make -j$(nproc)
```

## Upstream

- Upstream repository: https://github.com/FFmpeg/FFmpeg
- Based on tag: n7.1.5
