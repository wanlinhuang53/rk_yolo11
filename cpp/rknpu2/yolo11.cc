// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <regex>

#ifdef __linux__
#include <unistd.h>
#include <fcntl.h>
#endif

#include "yolo11.h"
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"

static bool g_enable_perf = false;
static bool g_enable_perf_run_print = true;
static bool g_enable_perf_detail_print = true;
static bool g_print_io = true;
static bool g_print_preprocess = true;

static float get_env_float(const char* name, float default_val)
{
    const char* v = getenv(name);
    if (v == NULL || v[0] == '\0')
    {
        return default_val;
    }
    char* end = NULL;
    float f = strtof(v, &end);
    if (end == v)
    {
        return default_val;
    }
    return f;
}

static int get_env_int(const char* name, int default_val)
{
    const char* v = getenv(name);
    if (v == NULL || v[0] == '\0')
    {
        return default_val;
    }
    char* end = NULL;
    long x = strtol(v, &end, 0);
    if (end == v)
    {
        return default_val;
    }
    return (int)x;
}

static bool get_env_bool(const char* name, bool default_val)
{
    const char* v = getenv(name);
    if (v == NULL || v[0] == '\0')
    {
        return default_val;
    }
    if (v[0] == '0')
    {
        return false;
    }
    if (v[0] == '1')
    {
        return true;
    }
    return default_val;
}

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

int init_yolo11_model(const char *model_path, rknn_app_context_t *app_ctx)
{
    int ret;
    int model_len = 0;
    char *model;
    rknn_context ctx = 0;

    // Load RKNN Model
    model_len = read_data_from_file(model_path, &model);
    if (model == NULL)
    {
        printf("load_model fail!\n");
        return -1;
    }

    const char* perf_env = getenv("RKNN_PERF");
    g_enable_perf = (perf_env != NULL) && (perf_env[0] == '1');
    g_enable_perf_run_print = get_env_bool("RKNN_PERF_RUN", true);
    g_enable_perf_detail_print = get_env_bool("RKNN_PERF_DETAIL", true);
    g_print_io = get_env_bool("RKNN_DEMO_PRINT_IO", true);
    g_print_preprocess = get_env_bool("RKNN_DEMO_PRINT_PREPROCESS", true);

    uint32_t rknn_flags = 0;
    if (g_enable_perf)
    {
        rknn_flags |= RKNN_FLAG_COLLECT_PERF_MASK;
    }

    ret = rknn_init(&ctx, model, model_len, rknn_flags, NULL);
    free(model);
    if (ret < 0)
    {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    {
        const char* core_mask_env = getenv("RKNN_NPU_CORE_MASK");
        int core_env = get_env_int("RKNN_NPU_CORE", -1);
        int mask = -1;
        if (core_mask_env != NULL && core_mask_env[0] != '\0')
        {
            mask = get_env_int("RKNN_NPU_CORE_MASK", -1);
        }
        else if (core_env >= 0)
        {
            mask = (core_env >= 0 && core_env <= 2) ? (1 << core_env) : -1;
        }

        if (mask > 0)
        {
            int r = rknn_set_core_mask(ctx, (rknn_core_mask)mask);
            if (r != RKNN_SUCC)
            {
                printf("rknn_set_core_mask fail! ret=%d mask=%d\n", r, mask);
            }
            else
            {
                printf("rknn_set_core_mask ok: mask=%d\n", mask);
            }
        }
    }

    // ##huagnadd
    rknn_mem_size mem_size;
    memset(&mem_size, 0, sizeof(mem_size));
    ret = rknn_query(ctx, RKNN_QUERY_MEM_SIZE, &mem_size, sizeof(mem_size));
    if (ret == RKNN_SUCC)
    {
        if (g_print_io)
        {
            printf("mem_size: weight=%u, internal=%u, dma=%llu, sram_total=%u, sram_free=%u\n",
                   mem_size.total_weight_size, mem_size.total_internal_size,
                   (unsigned long long)mem_size.total_dma_allocated_size,
                   mem_size.total_sram_size, mem_size.free_sram_size);
        }
    }

    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    if (g_print_io)
    {
        printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);
    }

    // Get Model Input Info
    if (g_print_io)
    {
        printf("input tensors:\n");
    }
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        if (g_print_io)
        {
            dump_tensor_attr(&(input_attrs[i]));
        }
    }

    // Get Model Output Info
    if (g_print_io)
    {
        printf("output tensors:\n");
    }
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        if (g_print_io)
        {
            dump_tensor_attr(&(output_attrs[i]));
        }
    }

    // Set to context
    app_ctx->rknn_ctx = ctx;

    // TODO
    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type == RKNN_TENSOR_INT8)
    {
        app_ctx->is_quant = true;
    }
    else
    {
        app_ctx->is_quant = false;
    }

    app_ctx->io_num = io_num;
    app_ctx->input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        if (g_print_io)
        {
            printf("model is NCHW input fmt\n");
        }
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height = input_attrs[0].dims[2];
        app_ctx->model_width = input_attrs[0].dims[3];
    }
    else
    {
        if (g_print_io)
        {
            printf("model is NHWC input fmt\n");
        }
        app_ctx->model_height = input_attrs[0].dims[1];
        app_ctx->model_width = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }
    if (g_print_io)
    {
        printf("model input height=%d, width=%d, channel=%d\n",
               app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);
    }

    return 0;
}

int release_yolo11_model(rknn_app_context_t *app_ctx)
{
    if (app_ctx->input_attrs != NULL)
    {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL)
    {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    if (app_ctx->rknn_ctx != 0)
    {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}

int inference_yolo11_model(rknn_app_context_t *app_ctx, image_buffer_t *img, object_detect_result_list *od_results)
{
    int ret;
    image_buffer_t dst_img;
    letterbox_t letter_box;
    rknn_input inputs[app_ctx->io_num.n_input];
    rknn_output outputs[app_ctx->io_num.n_output];
    float nms_threshold = get_env_float("RKNN_NMS_THRESH", NMS_THRESH);      // 默认的NMS阈值
    float box_conf_threshold = get_env_float("RKNN_BOX_THRESH", BOX_THRESH); // 默认的置信度阈值
    int bg_color = 114;

    if ((!app_ctx) || !(img) || (!od_results))
    {
        return -1;
    }

    memset(od_results, 0x00, sizeof(*od_results));
    memset(&letter_box, 0, sizeof(letterbox_t));
    memset(&dst_img, 0, sizeof(image_buffer_t));
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    // Pre Process
    dst_img.width = app_ctx->model_width;
    dst_img.height = app_ctx->model_height;
    dst_img.format = IMAGE_FORMAT_RGB888;
    dst_img.size = get_image_size(&dst_img);
    dst_img.virt_addr = (unsigned char *)malloc(dst_img.size);
    if (dst_img.virt_addr == NULL)
    {
        printf("malloc buffer size:%d fail!\n", dst_img.size);
        return -1;
    }

    // letterbox
#ifdef __linux__
    int saved_stdout = -1;
    int saved_stderr = -1;
    int nullfd = -1;
    if (!g_print_preprocess)
    {
        fflush(stdout);
        fflush(stderr);
        saved_stdout = dup(fileno(stdout));
        saved_stderr = dup(fileno(stderr));
        nullfd = open("/dev/null", O_WRONLY);
        if (nullfd >= 0)
        {
            dup2(nullfd, fileno(stdout));
            dup2(nullfd, fileno(stderr));
        }
    }
#endif

    ret = convert_image_with_letterbox(img, &dst_img, &letter_box, bg_color);

#ifdef __linux__
    if (!g_print_preprocess)
    {
        fflush(stdout);
        fflush(stderr);
        if (saved_stdout >= 0)
        {
            dup2(saved_stdout, fileno(stdout));
            close(saved_stdout);
        }
        if (saved_stderr >= 0)
        {
            dup2(saved_stderr, fileno(stderr));
            close(saved_stderr);
        }
        if (nullfd >= 0)
        {
            close(nullfd);
        }
    }
#endif

    if (ret < 0)
    {
        printf("convert_image_with_letterbox fail! ret=%d\n", ret);
        return -1;
    }

    // Set Input Data
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
    inputs[0].buf = dst_img.virt_addr;

    ret = rknn_inputs_set(app_ctx->rknn_ctx, app_ctx->io_num.n_input, inputs);
    if (ret < 0)
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    // Run
    if (g_enable_perf)
    {
        if (g_enable_perf_run_print)
        {
            printf("rknn_run\n");
        }
    }
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    // Get Output
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < app_ctx->io_num.n_output; i++)
    {
        outputs[i].index = i;
        outputs[i].want_float = (!app_ctx->is_quant);
    }
    ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        goto out;
    }

    if (g_enable_perf)
    {
        // 先获取 perf_run 总时间
        rknn_perf_run perf_run;
        memset(&perf_run, 0, sizeof(perf_run));
        ret = rknn_query(app_ctx->rknn_ctx, RKNN_QUERY_PERF_RUN, &perf_run, sizeof(perf_run));
        long long perf_run_us = 0;
        if (ret == RKNN_SUCC)
        {
            perf_run_us = (long long)perf_run.run_duration;
            if (g_enable_perf_run_print)
            {
                printf("perf_run: %lld us (%.3f ms)\n", perf_run_us, perf_run_us / 1000.0);
            }
        }

        if (g_enable_perf_detail_print)
        {
            rknn_perf_detail perf_detail;
            memset(&perf_detail, 0, sizeof(perf_detail));
            ret = rknn_query(app_ctx->rknn_ctx, RKNN_QUERY_PERF_DETAIL, &perf_detail, sizeof(perf_detail));
            if (ret == RKNN_SUCC && perf_detail.perf_data != NULL)
            {
                // 打印原始 perf_detail 数据
                printf("%s\n", perf_detail.perf_data);

                // 解析 perf_detail 并计算各层时间总和
                char* perf_str = perf_detail.perf_data;
                long long sum_layers_us = 0;
                int layer_count = 0;

                // 使用 strtok 按行分割
                char* line = strtok(perf_str, "\n");

                // 查找 "Total Operator Elapsed Per Frame Time(us):" 获取总和时间
                char* total_time_line = NULL;
                char temp_line[4096];

                while (line != NULL)
                {
                    // 复制一行用于搜索
                    strncpy(temp_line, line, sizeof(temp_line) - 1);
                    temp_line[sizeof(temp_line) - 1] = '\0';

                    // 查找总和时间行
                    if (total_time_line == NULL)
                    {
                        char* total_ptr = strstr(temp_line, "Total Operator Elapsed Per Frame Time(us):");
                        if (total_ptr != NULL)
                        {
                            // 提取时间值
                            char* num_ptr = total_ptr + strlen("Total Operator Elapsed Per Frame Time(us):");
                            while (*num_ptr && (*num_ptr < '0' || *num_ptr > '9')) num_ptr++;
                            if (*num_ptr)
                            {
                                sum_layers_us = strtoll(num_ptr, NULL, 10);
                            }
                        }
                    }

                    // 查找层信息行（以数字开头的行，如 "1    InputOperator"）
                    // 同时统计有效的 NPU 层
                    int len = strlen(temp_line);
                    if (len > 10 && temp_line[0] >= '0' && temp_line[0] <= '9')
                    {
                        // 跳过 Total 开头的行
                        if (strstr(temp_line, "Total") == NULL && strstr(temp_line, "Operator") == NULL)
                        {
                            // 检查是否是有效的层行（包含 NPU 或 CPU）
                            if (strstr(temp_line, "NPU") != NULL || strstr(temp_line, "CPU") != NULL)
                            {
                                layer_count++;
                            }
                        }
                    }

                    line = strtok(NULL, "\n");
                }

                // 打印对比结果（通过环境变量 RKNN_PERF_COMPARE 控制）
                if (getenv("RKNN_PERF_COMPARE") != NULL)
                {
                    printf("\n========== perf_run vs perf_detail 对比 ==========\n");
                    printf("perf_run (总推理时间):       %lld us (%.3f ms)\n", perf_run_us, perf_run_us / 1000.0);
                    printf("perf_detail (各层总和):      %lld us (%.3f ms)\n", sum_layers_us, sum_layers_us / 1000.0);
                    printf("层数:                        %d\n", layer_count);
                    if (sum_layers_us > 0)
                    {
                        double diff_percent = ((double)(perf_run_us - sum_layers_us) / sum_layers_us) * 100.0;
                        printf("差异:                        %lld us (%.2f%%)\n", perf_run_us - sum_layers_us, diff_percent);
                    }
                    printf("==================================================\n");
                }
            }
        }
    }

    // Post Process
    post_process(app_ctx, outputs, &letter_box, box_conf_threshold, nms_threshold, od_results);

    // Remeber to release rknn output
    rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);

out:
    if (dst_img.virt_addr != NULL)
    {
        free(dst_img.virt_addr);
    }

    return ret;
}