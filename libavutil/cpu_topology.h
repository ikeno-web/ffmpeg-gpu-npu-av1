/*
 * CPU topology detection for CCD/NUMA-aware thread placement.
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

#ifndef AVUTIL_CPU_TOPOLOGY_H
#define AVUTIL_CPU_TOPOLOGY_H

#include <stdint.h>

#define AV_MAX_CCDS 16
#define AV_MAX_CORES_PER_CCD 64

/**
 * Describes one CCD (Core Complex Die) / L3 cache domain.
 * On systems without CCDs (e.g. monolithic dies), this maps to the
 * single shared-L3 group containing all cores.
 */
typedef struct AVCCDInfo {
    int ccd_id;             /* 0-based CCD index */
    int nb_cores;           /* physical cores in this CCD */
    int nb_threads;         /* logical threads in this CCD */
    uint64_t core_mask;     /* bitmask of logical CPU ids (for <= 64 CPUs) */
    int processor_group;    /* Windows processor group, -1 if N/A */
    uint64_t group_mask;    /* mask within the processor group */
} AVCCDInfo;

/**
 * Full CPU topology description.
 */
typedef struct AVCPUTopology {
    int nb_ccds;            /* number of CCDs / L3 domains */
    int nb_physical_cores;  /* total physical cores */
    int nb_logical_cpus;    /* total logical CPUs (with SMT) */
    int smt_factor;         /* threads per core (1 or 2) */
    int l3_cache_size;      /* per-CCD L3 cache in bytes, 0 if unknown */
    AVCCDInfo ccds[AV_MAX_CCDS];
    int detected;           /* 1 if topology was successfully detected */
} AVCPUTopology;

/**
 * Detect CPU topology. Thread-safe, result is cached after first call.
 * Returns pointer to a static struct; never NULL.
 * On detection failure, returns a single-CCD fallback (detected = 0).
 */
const AVCPUTopology *ff_get_cpu_topology(void);

/**
 * Return the CCD index that logical CPU @p cpu_id belongs to, or -1.
 */
int ff_cpu_get_ccd_for_cpu(const AVCPUTopology *topo, int cpu_id);

/**
 * Returns the current NUMA/CCD affinity mode (-1 auto, 0 off, 1 on).
 * The mode is set through the public av_cpu_force_numa_aware().
 */
int  ff_cpu_get_numa_aware(void);

/**
 * Convenience helper: returns 1 if per-CCD thread pinning should be applied
 * for @p topo given the current mode, 0 otherwise.
 */
int  ff_cpu_should_pin(const AVCPUTopology *topo);

#endif /* AVUTIL_CPU_TOPOLOGY_H */
