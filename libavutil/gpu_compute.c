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

#include "config.h"

#include <string.h>

#include <stdatomic.h>

#include "buffer.h"
#include "gpu_compute.h"
#include "hwcontext.h"
#include "log.h"
#include "thread.h"

static AVGPUComputeCapabilities caps_cache;
static AVOnce caps_once = AV_ONCE_INIT;

static atomic_int preferred_backend = AV_GPU_COMPUTE_AUTO;

void av_gpu_compute_set_preferred(AVGPUComputeBackend backend)
{
    atomic_store_explicit(&preferred_backend, backend, memory_order_relaxed);
}

AVGPUComputeBackend av_gpu_compute_get_preferred(void)
{
    return atomic_load_explicit(&preferred_backend, memory_order_relaxed);
}

static const AVGPUComputeBackend default_priority[] = {
    AV_GPU_COMPUTE_VULKAN,
    AV_GPU_COMPUTE_OPENCL,
    AV_GPU_COMPUTE_D3D12,
    AV_GPU_COMPUTE_NONE,
};

enum AVHWDeviceType av_gpu_compute_to_hwdevice(AVGPUComputeBackend backend)
{
    switch (backend) {
    case AV_GPU_COMPUTE_VULKAN: return AV_HWDEVICE_TYPE_VULKAN;
    case AV_GPU_COMPUTE_OPENCL: return AV_HWDEVICE_TYPE_OPENCL;
    case AV_GPU_COMPUTE_D3D12:  return AV_HWDEVICE_TYPE_D3D12VA;
    default:                    return AV_HWDEVICE_TYPE_NONE;
    }
}

const char *av_gpu_compute_backend_name(AVGPUComputeBackend backend)
{
    switch (backend) {
    case AV_GPU_COMPUTE_VULKAN: return "vulkan";
    case AV_GPU_COMPUTE_OPENCL: return "opencl";
    case AV_GPU_COMPUTE_D3D12:  return "d3d12";
    default:                    return "none";
    }
}

AVGPUComputeBackend av_gpu_compute_backend_from_name(const char *name)
{
    if (!name)
        return AV_GPU_COMPUTE_NONE;
    if (!strcmp(name, "vulkan")) return AV_GPU_COMPUTE_VULKAN;
    if (!strcmp(name, "opencl")) return AV_GPU_COMPUTE_OPENCL;
    if (!strcmp(name, "d3d12"))  return AV_GPU_COMPUTE_D3D12;
    if (!strcmp(name, "auto"))   return AV_GPU_COMPUTE_AUTO;
    return AV_GPU_COMPUTE_NONE;
}

/* Try to create a device of the given type. On success, store the device
 * name (when discoverable) and return 1; otherwise return 0. */
static int probe_backend(AVGPUComputeBackend backend, char *name_out, size_t name_len)
{
    enum AVHWDeviceType type = av_gpu_compute_to_hwdevice(backend);
    AVBufferRef *dev = NULL;
    int ok;

    if (type == AV_HWDEVICE_TYPE_NONE)
        return 0;

    if (av_hwdevice_ctx_create(&dev, type, NULL, NULL, 0) < 0)
        return 0;

    ok = 1;
    if (name_out && name_len) {
        /* A generic, always-available label; backend-specific device name
         * extraction can be added per type later. */
        snprintf(name_out, name_len, "%s device", av_gpu_compute_backend_name(backend));
    }

    av_buffer_unref(&dev);
    return ok;
}

static void probe_init(void)
{
    memset(&caps_cache, 0, sizeof(caps_cache));

#if CONFIG_VULKAN
    if (probe_backend(AV_GPU_COMPUTE_VULKAN, caps_cache.vulkan_device_name,
                      sizeof(caps_cache.vulkan_device_name)))
        caps_cache.backends_available |= AV_GPU_COMPUTE_VULKAN;
#endif
#if CONFIG_OPENCL
    if (probe_backend(AV_GPU_COMPUTE_OPENCL, caps_cache.opencl_device_name,
                      sizeof(caps_cache.opencl_device_name)))
        caps_cache.backends_available |= AV_GPU_COMPUTE_OPENCL;
#endif
#if CONFIG_D3D12VA
    if (probe_backend(AV_GPU_COMPUTE_D3D12, caps_cache.d3d12_device_name,
                      sizeof(caps_cache.d3d12_device_name)))
        caps_cache.backends_available |= AV_GPU_COMPUTE_D3D12;
#endif

    av_log(NULL, AV_LOG_VERBOSE,
           "GPU compute backends available:%s%s%s%s\n",
           caps_cache.backends_available ? "" : " none",
           (caps_cache.backends_available & AV_GPU_COMPUTE_VULKAN) ? " vulkan" : "",
           (caps_cache.backends_available & AV_GPU_COMPUTE_OPENCL) ? " opencl" : "",
           (caps_cache.backends_available & AV_GPU_COMPUTE_D3D12)  ? " d3d12"  : "");
}

int av_gpu_compute_probe(AVGPUComputeCapabilities *caps)
{
    ff_thread_once(&caps_once, probe_init);
    if (caps)
        *caps = caps_cache;
    return caps_cache.backends_available;
}

AVGPUComputeBackend av_gpu_compute_select(AVGPUComputeBackend preferred,
                                          const AVGPUComputeBackend *priority)
{
    int available = av_gpu_compute_probe(NULL);
    const AVGPUComputeBackend *p;

    if (!available)
        return AV_GPU_COMPUTE_NONE;

    /* A specific backend was requested. */
    if (preferred != AV_GPU_COMPUTE_AUTO && preferred != AV_GPU_COMPUTE_NONE) {
        if (available & preferred)
            return preferred;
        av_log(NULL, AV_LOG_WARNING,
               "Requested GPU backend '%s' is not available; falling back.\n",
               av_gpu_compute_backend_name(preferred));
    }

    p = priority ? priority : default_priority;
    for (; *p != AV_GPU_COMPUTE_NONE; p++) {
        if (available & *p)
            return *p;
    }
    return AV_GPU_COMPUTE_NONE;
}
