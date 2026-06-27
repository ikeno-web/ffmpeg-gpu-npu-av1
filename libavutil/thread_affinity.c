/*
 * Cross-platform thread affinity for CCD-aware placement.
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

#if HAVE_SCHED_GETAFFINITY
#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif
#include <sched.h>
#endif

#include <errno.h>
#include <stdint.h>

#include "cpu_topology.h"
#include "error.h"
#include "log.h"
#include "thread.h"
#include "thread_affinity.h"

#if HAVE_GETPROCESSAFFINITYMASK || HAVE_WINRT
#include <windows.h>
#endif

int ff_compute_ccd_for_thread(const AVCPUTopology *topo,
                              int thread_index, int nb_threads)
{
    int total, acc, i;

    if (!topo || topo->nb_ccds <= 1)
        return 0;
    if (nb_threads < 1)
        nb_threads = 1;
    if (thread_index < 0)
        thread_index = 0;

    /* Distribute threads proportionally to each CCD's thread count. */
    total = topo->nb_logical_cpus;
    if (total < 1) {
        /* Even split across CCDs. */
        return (thread_index * topo->nb_ccds) / nb_threads;
    }

    acc = 0;
    for (i = 0; i < topo->nb_ccds; i++) {
        int share = (int)(((int64_t)topo->ccds[i].nb_threads * nb_threads + total - 1) / total);
        if (share < 1)
            share = 1;
        acc += share;
        if (thread_index < acc)
            return i;
    }
    return topo->nb_ccds - 1;
}

#if HAVE_GETPROCESSAFFINITYMASK || HAVE_WINRT

/* On w32threads the pthread_t wrapper has the OS HANDLE as its first member. */
static HANDLE handle_from_ptr(void *thread_handle)
{
    return *(HANDLE *)thread_handle;
}

static int set_affinity_handle(HANDLE h, const AVCCDInfo *ccd)
{
    DWORD_PTR mask = (DWORD_PTR)ccd->core_mask;
    if (!mask)
        return 0;
    if (!SetThreadAffinityMask(h, mask)) {
        av_log(NULL, AV_LOG_DEBUG,
               "SetThreadAffinityMask failed for CCD %d (err=%lu)\n",
               ccd->ccd_id, (unsigned long)GetLastError());
        return AVERROR_EXTERNAL;
    }
    av_log(NULL, AV_LOG_DEBUG, "Thread pinned to CCD %d (mask=0x%016llx)\n",
           ccd->ccd_id, (unsigned long long)ccd->core_mask);
    return 0;
}

int ff_set_current_thread_affinity_ccd(const AVCPUTopology *topo, int ccd_index)
{
    if (!topo || ccd_index < 0 || ccd_index >= topo->nb_ccds)
        return AVERROR(EINVAL);
    return set_affinity_handle(GetCurrentThread(), &topo->ccds[ccd_index]);
}

int ff_set_thread_affinity_ccd(void *thread_handle,
                               const AVCPUTopology *topo, int ccd_index)
{
    if (!topo || !thread_handle || ccd_index < 0 || ccd_index >= topo->nb_ccds)
        return AVERROR(EINVAL);
    return set_affinity_handle(handle_from_ptr(thread_handle), &topo->ccds[ccd_index]);
}

#elif HAVE_SCHED_GETAFFINITY && HAVE_PTHREADS

static void build_cpuset(cpu_set_t *set, const AVCCDInfo *ccd)
{
    CPU_ZERO(set);
    for (int cpu = 0; cpu < 64; cpu++) {
        if (ccd->core_mask & ((uint64_t)1 << cpu))
            CPU_SET(cpu, set);
    }
}

int ff_set_current_thread_affinity_ccd(const AVCPUTopology *topo, int ccd_index)
{
    cpu_set_t set;
    if (!topo || ccd_index < 0 || ccd_index >= topo->nb_ccds)
        return AVERROR(EINVAL);
    build_cpuset(&set, &topo->ccds[ccd_index]);
    if (sched_setaffinity(0, sizeof(set), &set)) {
        av_log(NULL, AV_LOG_DEBUG, "sched_setaffinity failed for CCD %d\n", ccd_index);
        return AVERROR(errno);
    }
    av_log(NULL, AV_LOG_DEBUG, "Thread pinned to CCD %d\n", ccd_index);
    return 0;
}

int ff_set_thread_affinity_ccd(void *thread_handle,
                               const AVCPUTopology *topo, int ccd_index)
{
    cpu_set_t set;
    pthread_t tid;
    int ret;

    if (!topo || !thread_handle || ccd_index < 0 || ccd_index >= topo->nb_ccds)
        return AVERROR(EINVAL);

    tid = *(pthread_t *)thread_handle;
    build_cpuset(&set, &topo->ccds[ccd_index]);
    ret = pthread_setaffinity_np(tid, sizeof(set), &set);
    if (ret) {
        av_log(NULL, AV_LOG_DEBUG, "pthread_setaffinity_np failed for CCD %d\n", ccd_index);
        return AVERROR(ret);
    }
    av_log(NULL, AV_LOG_DEBUG, "Thread pinned to CCD %d\n", ccd_index);
    return 0;
}

#else

int ff_set_current_thread_affinity_ccd(const AVCPUTopology *topo, int ccd_index)
{
    return 0;
}

int ff_set_thread_affinity_ccd(void *thread_handle,
                               const AVCPUTopology *topo, int ccd_index)
{
    return 0;
}

#endif
