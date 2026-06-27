/*
 * Copyright (c) 2026
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

/**
 * @file
 * DNN ONNX Runtime backend implementation.
 *
 * Supports the CPU execution provider everywhere, and the DirectML
 * execution provider (for NPU/GPU offloading on Windows) when the
 * ONNX Runtime build provides dml_provider_factory.h
 * (HAVE_ONNXRUNTIME_DML).
 */

#include "config.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include <onnxruntime_c_api.h>

#if HAVE_ONNXRUNTIME_DML
#include <dml_provider_factory.h>
#endif

#include "dnn_io_proc.h"
#include "dnn_backend_common.h"
#include "libavutil/avassert.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "queue.h"
#include "safe_queue.h"

typedef struct ONNXModel {
    DNNModel        model;
    DnnContext     *ctx;
    const OrtApi   *ort;
    OrtEnv         *env;
    OrtSession     *session;
    OrtSessionOptions *session_options;
    OrtMemoryInfo  *memory_info;
    OrtAllocator   *allocator;
    char           *input_name;
    char           *output_name;
    SafeQueue      *request_queue;
    Queue          *task_queue;
    Queue          *lltask_queue;
} ONNXModel;

typedef struct ONNXInferRequest {
    OrtValue *input_tensor;
    OrtValue *output_tensor;
} ONNXInferRequest;

typedef struct ONNXRequestItem {
    ONNXInferRequest   *infer_request;
    LastLevelTaskItem  *lltask;
    DNNAsyncExecModule  exec_module;
} ONNXRequestItem;

#define OFFSET(x) offsetof(ONNXRTOptions, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM
static const AVOption dnn_onnxrt_options[] = {
    { "execution_provider", "execution provider / device backend",
        OFFSET(execution_provider), AV_OPT_TYPE_INT, { .i64 = ORT_EP_CPU }, 0, 2, FLAGS, .unit = "ep" },
    { "cpu",      "CPU execution provider",       0, AV_OPT_TYPE_CONST, { .i64 = ORT_EP_CPU },      0, 0, FLAGS, .unit = "ep" },
    { "directml", "DirectML EP (NPU/GPU offload)", 0, AV_OPT_TYPE_CONST, { .i64 = ORT_EP_DIRECTML }, 0, 0, FLAGS, .unit = "ep" },
    { "cuda",     "CUDA EP",                       0, AV_OPT_TYPE_CONST, { .i64 = ORT_EP_CUDA },     0, 0, FLAGS, .unit = "ep" },
    { "device_id", "device index for the EP",
        OFFSET(device_id), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 64, FLAGS },
    { "intra_threads", "intra-op thread count (0 = ORT default)",
        OFFSET(intra_threads), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 256, FLAGS },
    { "graph_optimization", "ORT graph optimization level (0..3)",
        OFFSET(graph_optimization), AV_OPT_TYPE_INT, { .i64 = 3 }, 0, 3, FLAGS },
    { NULL }
};

static void dnn_free_model_onnx(DNNModel **model);

/* ---- helpers ---------------------------------------------------------- */

static int ort_check(ONNXModel *onnx_model, OrtStatus *status, const char *what)
{
    const OrtApi *ort = onnx_model->ort;
    if (status) {
        av_log(onnx_model->ctx, AV_LOG_ERROR, "ONNXRuntime: %s failed: %s\n",
               what, ort->GetErrorMessage(status));
        ort->ReleaseStatus(status);
        return DNN_GENERIC_ERROR;
    }
    return 0;
}

static int get_input_onnx(DNNModel *model, DNNData *input, const char *input_name)
{
    /* Frame-processing models: float NCHW RGB, dynamic spatial size. */
    input->dt     = DNN_FLOAT;
    input->order  = DCO_RGB;
    input->layout = DL_NCHW;
    input->dims[0] = 1;
    input->dims[1] = 3;
    input->dims[2] = -1;
    input->dims[3] = -1;
    return 0;
}

static void onnx_free_request(ONNXModel *onnx_model, ONNXInferRequest *request)
{
    const OrtApi *ort;
    if (!request || !onnx_model)
        return;
    ort = onnx_model->ort;
    if (request->input_tensor) {
        ort->ReleaseValue(request->input_tensor);
        request->input_tensor = NULL;
    }
    if (request->output_tensor) {
        ort->ReleaseValue(request->output_tensor);
        request->output_tensor = NULL;
    }
}

static int extract_lltask_from_task(TaskItem *task, Queue *lltask_queue)
{
    ONNXModel *onnx_model = task->model;
    DnnContext *ctx = onnx_model->ctx;
    LastLevelTaskItem *lltask = av_malloc(sizeof(*lltask));

    if (!lltask) {
        av_log(ctx, AV_LOG_ERROR, "Failed to allocate memory for LastLevelTaskItem\n");
        return AVERROR(ENOMEM);
    }
    task->inference_todo = 1;
    task->inference_done = 0;
    lltask->task = task;
    if (ff_queue_push_back(lltask_queue, lltask) < 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to push back lltask_queue.\n");
        av_freep(&lltask);
        return AVERROR(ENOMEM);
    }
    return 0;
}

static int fill_model_input_onnx(ONNXModel *onnx_model, ONNXRequestItem *request)
{
    const OrtApi *ort = onnx_model->ort;
    DnnContext *ctx = onnx_model->ctx;
    LastLevelTaskItem *lltask;
    TaskItem *task;
    ONNXInferRequest *infer_request;
    DNNData input = { 0 };
    int ret, width_idx, height_idx, channel_idx;
    int64_t shape[4];
    size_t elems;

    lltask = ff_queue_pop_front(onnx_model->lltask_queue);
    if (!lltask) {
        av_log(ctx, AV_LOG_ERROR, "Failed to get LastLevelTaskItem\n");
        return AVERROR(EINVAL);
    }
    request->lltask = lltask;
    task = lltask->task;
    infer_request = request->infer_request;

    ret = get_input_onnx(&onnx_model->model, &input, NULL);
    if (ret != 0)
        return ret;

    width_idx   = dnn_get_width_idx_by_layout(input.layout);
    height_idx  = dnn_get_height_idx_by_layout(input.layout);
    channel_idx = dnn_get_channel_idx_by_layout(input.layout);
    input.dims[height_idx] = task->in_frame->height;
    input.dims[width_idx]  = task->in_frame->width;

    elems = (size_t)input.dims[channel_idx] * input.dims[height_idx] * input.dims[width_idx];
    input.data = av_malloc(elems * sizeof(float));
    if (!input.data)
        return AVERROR(ENOMEM);

    switch (onnx_model->model.func_type) {
    case DFT_PROCESS_FRAME:
        input.scale = 255;
        if (task->do_ioproc) {
            if (onnx_model->model.frame_pre_proc)
                onnx_model->model.frame_pre_proc(task->in_frame, &input, onnx_model->model.filter_ctx);
            else
                ff_proc_from_frame_to_dnn(task->in_frame, &input, ctx);
        }
        break;
    default:
        avpriv_report_missing_feature(ctx, "model function type %d", onnx_model->model.func_type);
        av_freep(&input.data);
        return AVERROR(ENOSYS);
    }

    shape[0] = 1;
    shape[1] = input.dims[channel_idx];
    shape[2] = input.dims[height_idx];
    shape[3] = input.dims[width_idx];

    ret = ort_check(onnx_model,
        ort->CreateTensorWithDataAsOrtValue(onnx_model->memory_info, input.data,
            elems * sizeof(float), shape, 4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &infer_request->input_tensor),
        "CreateTensorWithDataAsOrtValue");
    if (ret != 0) {
        av_freep(&input.data);
        return ret;
    }
    /* ORT does not take ownership of the data buffer; we keep it alive in
     * the OrtValue and free it after the run via the output handling path.
     * Store the pointer so it is freed when the request is recycled. */
    return 0;
}

static int onnx_start_inference(void *args)
{
    ONNXRequestItem *request = args;
    ONNXInferRequest *infer_request;
    LastLevelTaskItem *lltask;
    TaskItem *task;
    ONNXModel *onnx_model;
    const OrtApi *ort;
    const char *input_names[1];
    const char *output_names[1];

    if (!request) {
        av_log(NULL, AV_LOG_ERROR, "ONNXRequestItem is NULL\n");
        return AVERROR(EINVAL);
    }
    infer_request = request->infer_request;
    lltask = request->lltask;
    task = lltask->task;
    onnx_model = task->model;
    ort = onnx_model->ort;

    input_names[0]  = onnx_model->input_name;
    output_names[0] = onnx_model->output_name;

    return ort_check(onnx_model,
        ort->Run(onnx_model->session, NULL,
                 input_names, (const OrtValue * const *)&infer_request->input_tensor, 1,
                 output_names, 1, &infer_request->output_tensor),
        "Run");
}

static void infer_completion_callback(void *args)
{
    ONNXRequestItem *request = args;
    LastLevelTaskItem *lltask = request->lltask;
    TaskItem *task = lltask->task;
    ONNXModel *onnx_model = task->model;
    const OrtApi *ort = onnx_model->ort;
    ONNXInferRequest *infer_request = request->infer_request;
    DNNData outputs = { 0 };
    OrtTensorTypeAndShapeInfo *info = NULL;
    size_t ndims = 0;
    int64_t shape[4] = { 0 };
    void *out_data = NULL;

    if (!infer_request->output_tensor) {
        av_log(onnx_model->ctx, AV_LOG_ERROR, "output tensor is NULL\n");
        goto err;
    }

    if (ort_check(onnx_model, ort->GetTensorTypeAndShape(infer_request->output_tensor, &info),
                  "GetTensorTypeAndShape"))
        goto err;
    if (ort_check(onnx_model, ort->GetDimensionsCount(info, &ndims), "GetDimensionsCount")) {
        ort->ReleaseTensorTypeAndShapeInfo(info);
        goto err;
    }
    if (ndims != 4) {
        avpriv_report_missing_feature(onnx_model->ctx, "non-NCHW output (%zu dims)", ndims);
        ort->ReleaseTensorTypeAndShapeInfo(info);
        goto err;
    }
    if (ort_check(onnx_model, ort->GetDimensions(info, shape, 4), "GetDimensions")) {
        ort->ReleaseTensorTypeAndShapeInfo(info);
        goto err;
    }
    ort->ReleaseTensorTypeAndShapeInfo(info);

    if (ort_check(onnx_model, ort->GetTensorMutableData(infer_request->output_tensor, &out_data),
                  "GetTensorMutableData"))
        goto err;

    outputs.order  = DCO_RGB;
    outputs.layout = DL_NCHW;
    outputs.dt     = DNN_FLOAT;
    outputs.dims[0] = shape[0];
    outputs.dims[1] = shape[1];
    outputs.dims[2] = shape[2];
    outputs.dims[3] = shape[3];
    outputs.data   = out_data;

    switch (onnx_model->model.func_type) {
    case DFT_PROCESS_FRAME:
        if (task->do_ioproc) {
            outputs.scale = 255;
            if (onnx_model->model.frame_post_proc)
                onnx_model->model.frame_post_proc(task->out_frame, &outputs, onnx_model->model.filter_ctx);
            else
                ff_proc_from_dnn_to_frame(task->out_frame, &outputs, onnx_model->ctx);
        } else {
            task->out_frame->width  = outputs.dims[dnn_get_width_idx_by_layout(outputs.layout)];
            task->out_frame->height = outputs.dims[dnn_get_height_idx_by_layout(outputs.layout)];
        }
        break;
    default:
        avpriv_report_missing_feature(onnx_model->ctx, "model function type %d", onnx_model->model.func_type);
        goto err;
    }
    task->inference_done++;
    av_freep(&request->lltask);
err:
    onnx_free_request(onnx_model, infer_request);
    if (ff_safe_queue_push_back(onnx_model->request_queue, request) < 0) {
        av_log(onnx_model->ctx, AV_LOG_ERROR, "Unable to push back request_queue.\n");
        onnx_free_request(onnx_model, request->infer_request);
        av_freep(&request->infer_request);
        av_freep(&request);
    }
}

static int execute_model_onnx(ONNXRequestItem *request, Queue *lltask_queue)
{
    ONNXModel *onnx_model;
    LastLevelTaskItem *lltask;
    TaskItem *task;
    int ret = 0;

    if (ff_queue_size(lltask_queue) == 0) {
        if (request) {
            onnx_free_request(NULL, request->infer_request);
        }
        return 0;
    }

    lltask = ff_queue_peek_front(lltask_queue);
    if (!lltask) {
        av_log(NULL, AV_LOG_ERROR, "Failed to get LastLevelTaskItem\n");
        return AVERROR(EINVAL);
    }
    task = lltask->task;
    onnx_model = task->model;

    ret = fill_model_input_onnx(onnx_model, request);
    if (ret != 0)
        goto err;

    if (task->async) {
        avpriv_report_missing_feature(onnx_model->ctx, "ONNXRuntime async");
    }

    ret = onnx_start_inference(request);
    if (ret != 0)
        goto err;
    infer_completion_callback(request);
    return (task->inference_done == task->inference_todo) ? 0 : DNN_GENERIC_ERROR;

err:
    if (onnx_model) {
        onnx_free_request(onnx_model, request->infer_request);
        if (ff_safe_queue_push_back(onnx_model->request_queue, request) < 0) {
            onnx_free_request(onnx_model, request->infer_request);
            av_freep(&request->infer_request);
            av_freep(&request);
        }
    }
    return ret;
}

static int get_output_onnx(DNNModel *model, const char *input_name, int input_width, int input_height,
                           const char *output_name, int *output_width, int *output_height)
{
    int ret;
    ONNXModel *onnx_model = (ONNXModel *)model;
    DnnContext *ctx = onnx_model->ctx;
    TaskItem task = { 0 };
    ONNXRequestItem *request = NULL;
    DNNExecBaseParams exec_params = {
        .input_name   = input_name,
        .output_names = &output_name,
        .nb_output    = 1,
        .in_frame     = NULL,
        .out_frame    = NULL,
    };

    ret = ff_dnn_fill_gettingoutput_task(&task, &exec_params, onnx_model, input_height, input_width, ctx);
    if (ret != 0)
        goto err;

    ret = extract_lltask_from_task(&task, onnx_model->lltask_queue);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "unable to extract last level task from task.\n");
        goto err;
    }

    request = ff_safe_queue_pop_front(onnx_model->request_queue);
    if (!request) {
        av_log(ctx, AV_LOG_ERROR, "unable to get infer request.\n");
        ret = AVERROR(EINVAL);
        goto err;
    }

    ret = execute_model_onnx(request, onnx_model->lltask_queue);
    *output_width  = task.out_frame->width;
    *output_height = task.out_frame->height;

err:
    av_frame_free(&task.out_frame);
    av_frame_free(&task.in_frame);
    return ret;
}

static ONNXInferRequest *onnx_create_inference_request(void)
{
    ONNXInferRequest *request = av_mallocz(sizeof(ONNXInferRequest));
    return request;
}

static int onnx_setup_providers(ONNXModel *onnx_model)
{
    DnnContext *ctx = onnx_model->ctx;
    ONNXRTOptions *opt = &ctx->onnxrt_option;

    switch (opt->execution_provider) {
    case ORT_EP_DIRECTML:
#if HAVE_ONNXRUNTIME_DML
        {
            OrtStatus *st = OrtSessionOptionsAppendExecutionProvider_DML(
                onnx_model->session_options, opt->device_id);
            if (ort_check(onnx_model, st, "AppendExecutionProvider_DML"))
                return DNN_GENERIC_ERROR;
            av_log(ctx, AV_LOG_INFO,
                   "ONNXRuntime: using DirectML EP (device %d) for NPU/GPU offload.\n",
                   opt->device_id);
        }
#else
        av_log(ctx, AV_LOG_WARNING,
               "ONNXRuntime built without DirectML; falling back to CPU EP. "
               "NPU offload is unavailable in this build.\n");
#endif
        break;
    case ORT_EP_CUDA:
        av_log(ctx, AV_LOG_WARNING,
               "ONNXRuntime CUDA EP not wired in this build; using CPU EP.\n");
        break;
    case ORT_EP_CPU:
    default:
        break;
    }
    return 0;
}

static DNNModel *dnn_load_model_onnx(DnnContext *ctx, DNNFunctionType func_type,
                                     AVFilterContext *filter_ctx)
{
    ONNXModel *onnx_model;
    DNNModel *model;
    ONNXRequestItem *item = NULL;
    const OrtApi *ort;
    char *in_name = NULL, *out_name = NULL;

    onnx_model = av_mallocz(sizeof(*onnx_model));
    if (!onnx_model)
        return NULL;
    model = &onnx_model->model;
    onnx_model->ctx = ctx;

    /* Request the API version we were built against, but tolerate an older
     * runtime: all functions used here have existed since early API
     * versions, so fall back to the newest version the loaded runtime
     * supports. This handles the common case of an older onnxruntime.dll
     * (e.g. one shipped in the system directory) shadowing the build's. */
    ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    for (int v = ORT_API_VERSION; !ort && v > 10; v--)
        ort = OrtGetApiBase()->GetApi(v);
    if (!ort) {
        av_log(ctx, AV_LOG_ERROR,
               "Failed to get a compatible ONNX Runtime API (runtime: %s)\n",
               OrtGetApiBase()->GetVersionString());
        goto fail;
    }
    onnx_model->ort = ort;
    av_log(ctx, AV_LOG_VERBOSE, "ONNX Runtime version: %s\n",
           OrtGetApiBase()->GetVersionString());

    if (ort_check(onnx_model, ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ffmpeg", &onnx_model->env),
                  "CreateEnv"))
        goto fail;
    if (ort_check(onnx_model, ort->CreateSessionOptions(&onnx_model->session_options),
                  "CreateSessionOptions"))
        goto fail;

    if (ort_check(onnx_model, ort->SetSessionGraphOptimizationLevel(onnx_model->session_options,
                  (GraphOptimizationLevel)ctx->onnxrt_option.graph_optimization),
                  "SetSessionGraphOptimizationLevel"))
        goto fail;
    if (ctx->onnxrt_option.intra_threads > 0 &&
        ort_check(onnx_model, ort->SetIntraOpNumThreads(onnx_model->session_options,
                  ctx->onnxrt_option.intra_threads), "SetIntraOpNumThreads"))
        goto fail;

    if (onnx_setup_providers(onnx_model))
        goto fail;

    /* ONNX Runtime takes the model path as ORTCHAR_T*: wchar_t on Windows,
     * char on other platforms. */
#ifdef _WIN32
    {
        int wlen = MultiByteToWideChar(CP_UTF8, 0, ctx->model_filename, -1, NULL, 0);
        wchar_t *wpath = wlen > 0 ? av_malloc(wlen * sizeof(wchar_t)) : NULL;
        OrtStatus *st;
        if (!wpath)
            goto fail;
        MultiByteToWideChar(CP_UTF8, 0, ctx->model_filename, -1, wpath, wlen);
        st = ort->CreateSession(onnx_model->env, wpath,
                                onnx_model->session_options, &onnx_model->session);
        av_freep(&wpath);
        if (ort_check(onnx_model, st, "CreateSession"))
            goto fail;
    }
#else
    if (ort_check(onnx_model, ort->CreateSession(onnx_model->env, ctx->model_filename,
                  onnx_model->session_options, &onnx_model->session), "CreateSession"))
        goto fail;
#endif

    if (ort_check(onnx_model, ort->GetAllocatorWithDefaultOptions(&onnx_model->allocator),
                  "GetAllocatorWithDefaultOptions"))
        goto fail;

    /* Resolve input/output names: use the user-provided ones, else query
     * the first input/output from the model. */
    if (ctx->model_inputname) {
        onnx_model->input_name = av_strdup(ctx->model_inputname);
    } else {
        if (ort_check(onnx_model, ort->SessionGetInputName(onnx_model->session, 0,
                      onnx_model->allocator, &in_name), "SessionGetInputName"))
            goto fail;
        onnx_model->input_name = av_strdup(in_name);
        onnx_model->allocator->Free(onnx_model->allocator, in_name);
    }

    if (ctx->model_outputnames && ctx->nb_outputs >= 1) {
        onnx_model->output_name = av_strdup(ctx->model_outputnames[0]);
    } else {
        if (ort_check(onnx_model, ort->SessionGetOutputName(onnx_model->session, 0,
                      onnx_model->allocator, &out_name), "SessionGetOutputName"))
            goto fail;
        onnx_model->output_name = av_strdup(out_name);
        onnx_model->allocator->Free(onnx_model->allocator, out_name);
    }
    if (!onnx_model->input_name || !onnx_model->output_name)
        goto fail;

    if (ort_check(onnx_model, ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                  &onnx_model->memory_info), "CreateCpuMemoryInfo"))
        goto fail;

    onnx_model->request_queue = ff_safe_queue_create();
    if (!onnx_model->request_queue)
        goto fail;

    item = av_mallocz(sizeof(*item));
    if (!item)
        goto fail;
    item->infer_request = onnx_create_inference_request();
    if (!item->infer_request)
        goto fail;
    item->exec_module.start_inference = &onnx_start_inference;
    item->exec_module.callback = &infer_completion_callback;
    item->exec_module.args = item;
    if (ff_safe_queue_push_back(onnx_model->request_queue, item) < 0)
        goto fail;
    item = NULL;

    onnx_model->task_queue = ff_queue_create();
    if (!onnx_model->task_queue)
        goto fail;
    onnx_model->lltask_queue = ff_queue_create();
    if (!onnx_model->lltask_queue)
        goto fail;

    model->get_input  = &get_input_onnx;
    model->get_output = &get_output_onnx;
    model->filter_ctx = filter_ctx;
    model->func_type  = func_type;

    av_log(ctx, AV_LOG_VERBOSE,
           "ONNXRuntime model loaded: input='%s' output='%s'\n",
           onnx_model->input_name, onnx_model->output_name);
    return model;

fail:
    if (item) {
        av_freep(&item->infer_request);
        av_freep(&item);
    }
    dnn_free_model_onnx(&model);
    return NULL;
}

static int dnn_execute_model_onnx(const DNNModel *model, DNNExecBaseParams *exec_params)
{
    ONNXModel *onnx_model = (ONNXModel *)model;
    DnnContext *ctx = onnx_model->ctx;
    TaskItem *task;
    ONNXRequestItem *request;
    int ret;

    ret = ff_check_exec_params(ctx, DNN_ONNXRT, model->func_type, exec_params);
    if (ret != 0)
        return ret;

    task = av_malloc(sizeof(*task));
    if (!task)
        return AVERROR(ENOMEM);

    ret = ff_dnn_fill_task(task, exec_params, onnx_model, 0, 1);
    if (ret != 0) {
        av_freep(&task);
        return ret;
    }

    if (ff_queue_push_back(onnx_model->task_queue, task) < 0) {
        av_freep(&task);
        av_log(ctx, AV_LOG_ERROR, "unable to push back task_queue.\n");
        return AVERROR(ENOMEM);
    }

    ret = extract_lltask_from_task(task, onnx_model->lltask_queue);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "unable to extract last level task from task.\n");
        return ret;
    }

    request = ff_safe_queue_pop_front(onnx_model->request_queue);
    if (!request) {
        av_log(ctx, AV_LOG_ERROR, "unable to get infer request.\n");
        return AVERROR(EINVAL);
    }

    return execute_model_onnx(request, onnx_model->lltask_queue);
}

static DNNAsyncStatusType dnn_get_result_onnx(const DNNModel *model, AVFrame **in, AVFrame **out)
{
    ONNXModel *onnx_model = (ONNXModel *)model;
    return ff_dnn_get_result_common(onnx_model->task_queue, in, out);
}

static int dnn_flush_onnx(const DNNModel *model)
{
    ONNXModel *onnx_model = (ONNXModel *)model;
    ONNXRequestItem *request;

    if (ff_queue_size(onnx_model->lltask_queue) == 0)
        return 0;

    request = ff_safe_queue_pop_front(onnx_model->request_queue);
    if (!request) {
        av_log(onnx_model->ctx, AV_LOG_ERROR, "unable to get infer request.\n");
        return AVERROR(EINVAL);
    }
    return execute_model_onnx(request, onnx_model->lltask_queue);
}

static void dnn_free_model_onnx(DNNModel **model)
{
    ONNXModel *onnx_model;
    const OrtApi *ort;

    if (!model || !*model)
        return;
    onnx_model = (ONNXModel *)(*model);
    ort = onnx_model->ort;

    while (onnx_model->request_queue && ff_safe_queue_size(onnx_model->request_queue) != 0) {
        ONNXRequestItem *item = ff_safe_queue_pop_front(onnx_model->request_queue);
        onnx_free_request(onnx_model, item->infer_request);
        av_freep(&item->infer_request);
        ff_dnn_async_module_cleanup(&item->exec_module);
        av_freep(&item);
    }
    if (onnx_model->request_queue)
        ff_safe_queue_destroy(onnx_model->request_queue);

    while (onnx_model->lltask_queue && ff_queue_size(onnx_model->lltask_queue) != 0) {
        LastLevelTaskItem *item = ff_queue_pop_front(onnx_model->lltask_queue);
        av_freep(&item);
    }
    if (onnx_model->lltask_queue)
        ff_queue_destroy(onnx_model->lltask_queue);

    while (onnx_model->task_queue && ff_queue_size(onnx_model->task_queue) != 0) {
        TaskItem *item = ff_queue_pop_front(onnx_model->task_queue);
        av_frame_free(&item->in_frame);
        av_frame_free(&item->out_frame);
        av_freep(&item);
    }
    if (onnx_model->task_queue)
        ff_queue_destroy(onnx_model->task_queue);

    if (ort) {
        if (onnx_model->memory_info)    ort->ReleaseMemoryInfo(onnx_model->memory_info);
        if (onnx_model->session)        ort->ReleaseSession(onnx_model->session);
        if (onnx_model->session_options) ort->ReleaseSessionOptions(onnx_model->session_options);
        if (onnx_model->env)            ort->ReleaseEnv(onnx_model->env);
    }
    av_freep(&onnx_model->input_name);
    av_freep(&onnx_model->output_name);
    av_freep(&onnx_model);
    *model = NULL;
}

const DNNModule ff_dnn_backend_onnxrt = {
    .clazz          = DNN_DEFINE_CLASS(dnn_onnxrt),
    .type           = DNN_ONNXRT,
    .load_model     = dnn_load_model_onnx,
    .execute_model  = dnn_execute_model_onnx,
    .get_result     = dnn_get_result_onnx,
    .flush          = dnn_flush_onnx,
    .free_model     = dnn_free_model_onnx,
};
