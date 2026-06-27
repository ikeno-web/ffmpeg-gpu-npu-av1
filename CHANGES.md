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
- (Planned) ONNX Runtime + DirectML DNN backend for NPU offloading

## Building

```bash
./configure --enable-gpl --enable-nonfree
make -j$(nproc)
```

## Upstream

- Upstream repository: https://github.com/FFmpeg/FFmpeg
- Based on tag: n7.1.5
