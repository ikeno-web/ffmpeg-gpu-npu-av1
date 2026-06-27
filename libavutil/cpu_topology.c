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

#include "config.h"

#include <inttypes.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

#include "common.h"
#include "cpu.h"
#include "cpu_topology.h"
#include "log.h"
#include "macros.h"
#include "mem.h"
#include "thread.h"

#if HAVE_GETPROCESSAFFINITYMASK || HAVE_WINRT
#include <windows.h>
#endif

static AVCPUTopology topology;
static AVOnce topology_once = AV_ONCE_INIT;

/* -1 = auto, 0 = off, 1 = on */
static atomic_int numa_aware_mode = -1;

void av_cpu_force_numa_aware(int mode)
{
    atomic_store_explicit(&numa_aware_mode, mode, memory_order_relaxed);
}

int ff_cpu_get_numa_aware(void)
{
    return atomic_load_explicit(&numa_aware_mode, memory_order_relaxed);
}

int ff_cpu_should_pin(const AVCPUTopology *topo)
{
    int mode = ff_cpu_get_numa_aware();
    if (!topo || !topo->detected || mode == 0)
        return 0;
    if (mode == 1)
        return 1;
    /* auto */
    return topo->nb_ccds > 1;
}

static void topology_fallback(void)
{
    int nb_cpus = av_cpu_count();
    if (nb_cpus < 1)
        nb_cpus = 1;

    topology.nb_ccds          = 1;
    topology.nb_logical_cpus  = nb_cpus;
    topology.nb_physical_cores = nb_cpus;
    topology.smt_factor       = 1;
    topology.l3_cache_size    = 0;
    topology.detected         = 0;

    topology.ccds[0].ccd_id          = 0;
    topology.ccds[0].nb_cores        = nb_cpus;
    topology.ccds[0].nb_threads      = nb_cpus;
    topology.ccds[0].core_mask       = (nb_cpus >= 64) ? ~(uint64_t)0
                                                       : (((uint64_t)1 << nb_cpus) - 1);
    topology.ccds[0].processor_group = -1;
    topology.ccds[0].group_mask      = topology.ccds[0].core_mask;
}

#if HAVE_GETPROCESSAFFINITYMASK || HAVE_WINRT
static int topology_detect_windows(void)
{
    DWORD len = 0;
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *buf, *ptr;
    char *raw;
    int nb_ccds = 0, total_threads = 0, total_cores = 0;

    /* First call to obtain the required buffer size. */
    if (GetLogicalProcessorInformationEx(RelationCache, NULL, &len))
        return 0;
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || !len)
        return 0;

    buf = av_malloc(len);
    if (!buf)
        return 0;

    if (!GetLogicalProcessorInformationEx(RelationCache, buf, &len)) {
        av_free(buf);
        return 0;
    }

    raw = (char *)buf;
    for (DWORD off = 0; off < len; ) {
        ptr = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)(raw + off);
        off += ptr->Size;

        if (ptr->Relationship != RelationCache)
            continue;
        if (ptr->Cache.Level != 3)
            continue;
        if (nb_ccds >= AV_MAX_CCDS)
            break;

        {
            GROUP_AFFINITY ga = ptr->Cache.GroupMask;
            int bits = av_popcount64((uint64_t)ga.Mask);
            AVCCDInfo *ccd = &topology.ccds[nb_ccds];

            ccd->ccd_id          = nb_ccds;
            ccd->nb_threads      = bits;
            ccd->nb_cores        = bits > 1 ? bits / 2 : bits;
            ccd->core_mask       = (uint64_t)ga.Mask;
            ccd->processor_group = ga.Group;
            ccd->group_mask      = (uint64_t)ga.Mask;

            total_threads += ccd->nb_threads;
            total_cores   += ccd->nb_cores;
            if (!topology.l3_cache_size)
                topology.l3_cache_size = ptr->Cache.CacheSize;
            nb_ccds++;
        }
    }

    av_free(buf);

    if (nb_ccds < 1)
        return 0;

    topology.nb_ccds           = nb_ccds;
    topology.nb_logical_cpus   = total_threads;
    topology.nb_physical_cores = total_cores;
    topology.smt_factor        = (total_cores > 0 && total_threads / total_cores >= 2) ? 2 : 1;
    topology.detected          = 1;
    return 1;
}
#endif

#if !( HAVE_GETPROCESSAFFINITYMASK || HAVE_WINRT ) && HAVE_SCHED_GETAFFINITY
static int read_sysfs_int(const char *path, long *out)
{
    FILE *f = fopen(path, "r");
    long v;
    if (!f)
        return -1;
    if (fscanf(f, "%ld", &v) != 1) {
        fclose(f);
        return -1;
    }
    fclose(f);
    *out = v;
    return 0;
}

static int topology_detect_linux(void)
{
    int nb_cpus = av_cpu_count();
    long l3_ids[AV_MAX_CCDS];
    int nb_ccds = 0;

    if (nb_cpus < 1 || nb_cpus > 64)
        return 0;

    memset(l3_ids, 0, sizeof(l3_ids));

    for (int cpu = 0; cpu < nb_cpus; cpu++) {
        char path[256];
        long id;
        int ccd_idx = -1;

        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/cache/index3/id", cpu);
        if (read_sysfs_int(path, &id) < 0)
            return 0;

        for (int j = 0; j < nb_ccds; j++) {
            if (l3_ids[j] == id) {
                ccd_idx = j;
                break;
            }
        }
        if (ccd_idx < 0) {
            if (nb_ccds >= AV_MAX_CCDS)
                return 0;
            ccd_idx = nb_ccds;
            l3_ids[nb_ccds] = id;
            topology.ccds[nb_ccds].ccd_id          = nb_ccds;
            topology.ccds[nb_ccds].processor_group = -1;
            nb_ccds++;
        }

        topology.ccds[ccd_idx].core_mask  |= (uint64_t)1 << cpu;
        topology.ccds[ccd_idx].group_mask  = topology.ccds[ccd_idx].core_mask;
        topology.ccds[ccd_idx].nb_threads++;
    }

    if (nb_ccds < 1)
        return 0;

    {
        long size_kb = 0;
        if (read_sysfs_int("/sys/devices/system/cpu/cpu0/cache/index3/size", &size_kb) == 0)
            topology.l3_cache_size = (int)(size_kb * 1024);
    }

    {
        int total_threads = 0, total_cores = 0;
        for (int j = 0; j < nb_ccds; j++) {
            AVCCDInfo *ccd = &topology.ccds[j];
            ccd->nb_cores = ccd->nb_threads > 1 ? ccd->nb_threads / 2 : ccd->nb_threads;
            total_threads += ccd->nb_threads;
            total_cores   += ccd->nb_cores;
        }
        topology.nb_ccds           = nb_ccds;
        topology.nb_logical_cpus   = total_threads;
        topology.nb_physical_cores = total_cores;
        topology.smt_factor        = (total_cores > 0 && total_threads / total_cores >= 2) ? 2 : 1;
    }

    topology.detected = 1;
    return 1;
}
#endif

static void topology_init(void)
{
    int ok = 0;

    memset(&topology, 0, sizeof(topology));

#if HAVE_GETPROCESSAFFINITYMASK || HAVE_WINRT
    ok = topology_detect_windows();
#elif HAVE_SCHED_GETAFFINITY
    ok = topology_detect_linux();
#endif

    if (!ok)
        topology_fallback();

    av_log(NULL, AV_LOG_VERBOSE,
           "CPU topology: %d CCD(s), %d physical cores, %d logical CPUs, SMT=%d%s\n",
           topology.nb_ccds, topology.nb_physical_cores,
           topology.nb_logical_cpus, topology.smt_factor,
           topology.detected ? "" : " (fallback)");

    for (int i = 0; i < topology.nb_ccds; i++) {
        const AVCCDInfo *ccd = &topology.ccds[i];
        av_log(NULL, AV_LOG_VERBOSE,
               "  CCD %d: %d cores, %d threads, L3=%dKB, group=%d, mask=0x%016"PRIx64"\n",
               ccd->ccd_id, ccd->nb_cores, ccd->nb_threads,
               topology.l3_cache_size / 1024, ccd->processor_group, ccd->core_mask);
    }
}

const AVCPUTopology *ff_get_cpu_topology(void)
{
    ff_thread_once(&topology_once, topology_init);
    return &topology;
}

int ff_cpu_get_ccd_for_cpu(const AVCPUTopology *topo, int cpu_id)
{
    if (!topo || cpu_id < 0 || cpu_id >= 64)
        return -1;
    for (int i = 0; i < topo->nb_ccds; i++) {
        if (topo->ccds[i].core_mask & ((uint64_t)1 << cpu_id))
            return i;
    }
    return -1;
}
