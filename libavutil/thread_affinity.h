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

#ifndef AVUTIL_THREAD_AFFINITY_H
#define AVUTIL_THREAD_AFFINITY_H

#include "cpu_topology.h"

/**
 * Pin the calling thread to the cores of the specified CCD.
 * Returns 0 on success (including the no-op case on unsupported
 * platforms), negative AVERROR on failure.
 */
int ff_set_current_thread_affinity_ccd(const AVCPUTopology *topo, int ccd_index);

/**
 * Pin a thread (given by platform handle) to the specified CCD.
 * On w32threads, @p thread_handle points to the pthread_t wrapper struct.
 * On POSIX, @p thread_handle points to a pthread_t.
 * Returns 0 on success, negative AVERROR on failure.
 */
int ff_set_thread_affinity_ccd(void *thread_handle,
                               const AVCPUTopology *topo, int ccd_index);

/**
 * Compute which CCD a worker thread should be assigned to, distributing
 * @p nb_threads worker threads proportionally across all CCDs.
 * Returns the CCD index, or 0 if topology is unavailable.
 */
int ff_compute_ccd_for_thread(const AVCPUTopology *topo,
                              int thread_index, int nb_threads);

#endif /* AVUTIL_THREAD_AFFINITY_H */
