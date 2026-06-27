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
- Verified on AMD Ryzen 9950X (2 CCDs, 8 cores / 16 threads each, 32 MB L3
  per CCD): topology and per-CCD thread distribution detected correctly.

### Custom Filters
- (Planned) Vendor-neutral GPU compute dispatch (Vulkan / OpenCL / D3D12)

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
