/*
 * Vendor-neutral GPU compute backend selection.
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#ifndef AVUTIL_GPU_COMPUTE_H
#define AVUTIL_GPU_COMPUTE_H

#include "hwcontext.h"

/**
 * GPU compute backends, usable as a bitmask of available backends.
 */
typedef enum AVGPUComputeBackend {
    AV_GPU_COMPUTE_NONE   = 0,
    AV_GPU_COMPUTE_VULKAN = 1 << 0,
    AV_GPU_COMPUTE_OPENCL = 1 << 1,
    AV_GPU_COMPUTE_D3D12  = 1 << 2,
    AV_GPU_COMPUTE_AUTO   = 0x7FFFFFFF,
} AVGPUComputeBackend;

#define AV_GPU_DEVICE_NAME_LEN 256

typedef struct AVGPUComputeCapabilities {
    int backends_available;     /* bitmask of AVGPUComputeBackend */
    char vulkan_device_name[AV_GPU_DEVICE_NAME_LEN];
    char opencl_device_name[AV_GPU_DEVICE_NAME_LEN];
    char d3d12_device_name[AV_GPU_DEVICE_NAME_LEN];
} AVGPUComputeCapabilities;

/**
 * Probe which GPU compute backends are available at runtime.
 * Result is cached after the first call.
 * @param caps optional; filled with details when non-NULL.
 * @return bitmask of available AVGPUComputeBackend values.
 */
int av_gpu_compute_probe(AVGPUComputeCapabilities *caps);

/**
 * Select the best available backend.
 * @param preferred AV_GPU_COMPUTE_AUTO or a specific backend to force.
 * @param priority  NULL-terminated array (terminated by AV_GPU_COMPUTE_NONE)
 *                  giving the preference order, or NULL for the default
 *                  (Vulkan > OpenCL > D3D12).
 * @return the selected backend, or AV_GPU_COMPUTE_NONE if none available.
 */
AVGPUComputeBackend av_gpu_compute_select(AVGPUComputeBackend preferred,
                                          const AVGPUComputeBackend *priority);

/**
 * Set/get the user-preferred GPU compute backend (FFmpeg Plus extension).
 * Default is AV_GPU_COMPUTE_AUTO. Used to resolve "*_gpu" meta-filters.
 */
void av_gpu_compute_set_preferred(AVGPUComputeBackend backend);
AVGPUComputeBackend av_gpu_compute_get_preferred(void);

/**
 * Map a GPU compute backend to the corresponding AVHWDeviceType.
 */
enum AVHWDeviceType av_gpu_compute_to_hwdevice(AVGPUComputeBackend backend);

/**
 * Return the canonical lowercase name of a backend ("vulkan", "opencl",
 * "d3d12", or "none"). Never returns NULL.
 */
const char *av_gpu_compute_backend_name(AVGPUComputeBackend backend);

/**
 * Parse a backend name ("vulkan"/"opencl"/"d3d12"/"auto"). Returns
 * AV_GPU_COMPUTE_NONE on an unrecognized string.
 */
AVGPUComputeBackend av_gpu_compute_backend_from_name(const char *name);

#endif /* AVUTIL_GPU_COMPUTE_H */
