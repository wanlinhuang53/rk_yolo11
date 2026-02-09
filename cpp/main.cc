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

/*-------------------------------------------
|  YOLO11 Demo - 支持守护模式 (模型只加载一次)
|
|  普通模式: ./rknn_yolo11_demo <model> <image> [warmup] [repeat]
|  守护模式: RKNN_DAEMON=1 ./rknn_yolo11_demo <model> <image>
|           然后通过 stdin 发送命令:
|             run <warmup> <repeat>  - 执行测试
|             quit                    - 退出
|-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <unistd.h>
#include <fcntl.h>

#include "yolo11.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

#if defined(RV1106_1103) 
#include "dma_alloc.hpp"
#endif

// 守护模式全局状态
static bool g_daemon_mode = false;
static rknn_app_context_t g_rknn_app_ctx;
static image_buffer_t g_src_image;
static bool g_model_loaded = false;
static bool g_image_loaded = false;

// 守护模式: 解析命令
static int daemon_parse_command(const char* cmd, int* warmup, int* repeat) {
    if (strcmp(cmd, "quit") == 0 || strcmp(cmd, "exit") == 0) {
        return -1;  // 退出信号
    }
    
    if (strncmp(cmd, "run", 3) == 0) {
        // 格式: run <warmup> <repeat>
        *warmup = 10;
        *repeat = 100;
        sscanf(cmd + 3, "%d %d", warmup, repeat);
        if (*warmup < 0) *warmup = 0;
        if (*repeat < 0) *repeat = 0;
        return 0;  // 执行测试
    }
    
    if (strcmp(cmd, "status") == 0) {
        fprintf(stdout, "OK model_loaded=%d image_loaded=%d\n", 
                g_model_loaded, g_image_loaded);
        fflush(stdout);
        return 2;  // 仅查询状态
    }
    
    fprintf(stderr, "ERROR unknown command: %s\n", cmd);
    fflush(stderr);
    return 2;
}

// 守护模式: 执行测试
static int daemon_run_benchmark(int warmup, int repeat) {
    if (!g_model_loaded || !g_image_loaded) {
        fprintf(stdout, "ERROR model or image not loaded\n");
        fflush(stdout);
        return -1;
    }
    
    int ret;
    object_detect_result_list od_results;
    
    // 预热
    if (warmup > 0) {
        for (int i = 0; i < warmup; i++) {
            object_detect_result_list tmp_results;
            ret = inference_yolo11_model(&g_rknn_app_ctx, &g_src_image, &tmp_results);
            if (ret != 0) {
                fprintf(stdout, "ERROR warmup failed ret=%d\n", ret);
                fflush(stdout);
                return -1;
            }
        }
    }
    
    // 正式测试
    if (repeat > 0) {
        long long sum_us = 0;
        long long min_us = LLONG_MAX;
        long long max_us = 0;
        int ok = 0;
        
        std::vector<long long> run_us_list;
        run_us_list.reserve((size_t)repeat);
        
        for (int i = 0; i < repeat; i++) {
            ret = inference_yolo11_model(&g_rknn_app_ctx, &g_src_image, &od_results);
            if (ret != 0) {
                fprintf(stderr, "ERROR inference failed ret=%d iter=%d\n", ret, i);
                fflush(stderr);
                continue;
            }
            
            rknn_perf_run perf_run;
            memset(&perf_run, 0, sizeof(perf_run));
            ret = rknn_query(g_rknn_app_ctx.rknn_ctx, RKNN_QUERY_PERF_RUN, &perf_run, sizeof(perf_run));
            if (ret == RKNN_SUCC) {
                long long us = perf_run.run_duration;
                run_us_list.push_back(us);
                sum_us += us;
                if (us < min_us) min_us = us;
                if (us > max_us) max_us = us;
                ok++;
                
                // 输出每次迭代 (CSV 格式: iter,<idx>,<us>,<ms>)
                // 输出到 stdout（会被脚本捕获）
                fprintf(stdout, "iter,%d,%lld,%.3f\n", i, us, us / 1000.0);
                fflush(stdout);
            }
        }
        
        if (ok > 0) {
            double avg_ms = (sum_us / (double)ok) / 1000.0;
            double min_ms = min_us / 1000.0;
            double max_ms = max_us / 1000.0;
            
            // 输出汇总 - 输出到 stdout
            fprintf(stdout, "done,%d,%.3f,%.3f,%.3f\n", ok, avg_ms, min_ms, max_ms);
            fflush(stdout);
        } else {
            fprintf(stdout, "done,0,0,0,0\n");
            fflush(stdout);
        }
    }
    
    return 0;
}

// 守护模式: 主循环
static int daemon_main_loop() {
    // READY 输出到 stdout
    fprintf(stdout, "READY\n");
    fflush(stdout);
    
    char cmd[256];
    while (fgets(cmd, sizeof(cmd), stdin) != nullptr) {
        // 去除换行符
        cmd[strcspn(cmd, "\r\n")] = 0;
        
        int warmup = 10, repeat = 100;
        int cmd_type = daemon_parse_command(cmd, &warmup, &repeat);
        
        if (cmd_type == -1) {
            // quit
            fprintf(stdout, "BYE\n");
            fflush(stdout);
            break;
        }
        
        if (cmd_type == 2) {
            // 仅状态查询
            continue;
        }
        
        // 执行测试
        daemon_run_benchmark(warmup, repeat);
    }
    
    return 0;
}

// 守护模式: 初始化
static int daemon_init(const char* model_path, const char* image_path) {
    int ret;
    
    memset(&g_rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    memset(&g_src_image, 0, sizeof(image_buffer_t));
    
    init_post_process();
    
    // 加载模型
    ret = init_yolo11_model(model_path, &g_rknn_app_ctx);
    if (ret != 0) {
        fprintf(stderr, "init_yolo11_model fail! ret=%d\n", ret);
        return -1;
    }
    g_model_loaded = true;
    
    fprintf(stdout, "MODEL_LOADED %s\n", model_path);
    fflush(stdout);
    
    // 读取图片
    ret = read_image(image_path, &g_src_image);
    if (ret != 0) {
        fprintf(stderr, "read_image fail! ret=%d\n", ret);
        return -1;
    }
    g_image_loaded = true;
    
    fprintf(stdout, "IMAGE_LOADED %s\n", image_path);
    fflush(stdout);
    
    return 0;
}

// 守护模式: 清理
static void daemon_cleanup() {
    if (g_image_loaded && g_src_image.virt_addr != nullptr) {
#if defined(RV1106_1103) 
        dma_buf_free(g_rknn_app_ctx.img_dma_buf.size, &g_rknn_app_ctx.img_dma_buf.dma_buf_fd,
                g_rknn_app_ctx.img_dma_buf.dma_buf_virt_addr);
#else
        free(g_src_image.virt_addr);
#endif
        g_src_image.virt_addr = nullptr;
    }
    
    if (g_model_loaded) {
        release_yolo11_model(&g_rknn_app_ctx);
        g_model_loaded = false;
    }
    
    deinit_post_process();
}

/*-------------------------------------------
| 普通模式: 原始逻辑
|-------------------------------------------*/
int main_normal(int argc, char **argv)
{
    if (argc < 3 || argc > 5)
    {
        printf("%s <model_path> <image_path> [warmup(optional)] [repeat(optional)]\n", argv[0]);
        printf("  warmup: number of warmup runs before benchmarking (default 10)\n");
        printf("  repeat: number of benchmark runs to average (default 100)\n");
        return -1;
    }

    const char *model_path = argv[1];
    const char *image_path = argv[2];
    int warmup = 10;
    int repeat = 100;
    if (argc >= 4)
    {
        warmup = atoi(argv[3]);
        if (warmup < 0)
            warmup = 0;
    }
    if (argc >= 5)
    {
        repeat = atoi(argv[4]);
        if (repeat < 0)
            repeat = 0;
    }

    // For benchmarking loops, disable verbose per-run perf prints.
    // Perf collection is still controlled by RKNN_PERF=1.
    if (repeat > 2)
    {
        setenv("RKNN_PERF_RUN", "0", 0);
        setenv("RKNN_PERF_DETAIL", "0", 0);
    }

    const char* dump_minmax_env = getenv("RKNN_BENCH_DUMP_MINMAX_DETAIL");
    bool dump_minmax_detail = (dump_minmax_env != NULL) && (dump_minmax_env[0] == '1');
    std::string min_detail;
    std::string max_detail;
    int min_i = -1;
    int max_i = -1;

    const char* perf_detail_every_env = getenv("RKNN_BENCH_PERF_DETAIL_EVERY");
    int perf_detail_every = 0;
    if (perf_detail_every_env != NULL && perf_detail_every_env[0] != '\0')
    {
        perf_detail_every = atoi(perf_detail_every_env);
        if (perf_detail_every < 0)
            perf_detail_every = 0;
    }
    const char* perf_detail_max_env = getenv("RKNN_BENCH_PERF_DETAIL_MAX");
    int perf_detail_max = 0;
    if (perf_detail_max_env != NULL && perf_detail_max_env[0] != '\0')
    {
        perf_detail_max = atoi(perf_detail_max_env);
        if (perf_detail_max < 0)
            perf_detail_max = 0;
    }
    int perf_detail_dumped = 0;

    const char* save_img_env = getenv("RKNN_DEMO_SAVE_IMAGE");
    bool save_image = (save_img_env == NULL) || (save_img_env[0] != '0');

    const char* extreme_topk_env = getenv("RKNN_BENCH_EXTREME_TOPK");
    int extreme_topk = 0;
    if (extreme_topk_env != NULL && extreme_topk_env[0] != '\0')
    {
        extreme_topk = atoi(extreme_topk_env);
        if (extreme_topk < 0)
            extreme_topk = 0;
    }
    const char* extremes_csv_env = getenv("RKNN_BENCH_EXTREMES_CSV");
    std::string extremes_csv_path = (extremes_csv_env != NULL && extremes_csv_env[0] != '\0')
        ? std::string(extremes_csv_env)
        : std::string("bench_extremes.csv");

    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    init_post_process();

    ret = init_yolo11_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolo11_model fail! ret=%d model_path=%s\n", ret, model_path);
        goto out;
    }

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    ret = read_image(image_path, &src_image);

#if defined(RV1106_1103) 
    //RV1106 rga requires that input and output bufs are memory allocated by dma
    ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, src_image.size, &rknn_app_ctx.img_dma_buf.dma_buf_fd, 
                       (void **) & (rknn_app_ctx.img_dma_buf.dma_buf_virt_addr));
    memcpy(rknn_app_ctx.img_dma_buf.dma_buf_virt_addr, src_image.virt_addr, src_image.size);
    dma_sync_cpu_to_device(rknn_app_ctx.img_dma_buf.dma_buf_fd);
    free(src_image.virt_addr);
    src_image.virt_addr = (unsigned char *)rknn_app_ctx.img_dma_buf.dma_buf_virt_addr;
    src_image.fd = rknn_app_ctx.img_dma_buf.dma_buf_fd;
    rknn_app_ctx.img_dma_buf.size = src_image.size;
#endif
    
    if (ret != 0)
    {
        printf("read image fail! ret=%d image_path=%s\n", ret, image_path);
        goto out;
    }

    object_detect_result_list od_results;

    if (warmup > 0)
    {
        for (int i = 0; i < warmup; i++)
        {
            object_detect_result_list tmp_results;
            ret = inference_yolo11_model(&rknn_app_ctx, &src_image, &tmp_results);
            if (ret != 0)
            {
                printf("inference_yolo11_model warmup fail! ret=%d\n", ret);
                goto out;
            }
        }
    }

    if (repeat > 0)
    {
        long long sum_us = 0;
        long long min_us = LLONG_MAX;
        long long max_us = 0;
        int ok = 0;

        const char* bench_csv_env = getenv("RKNN_BENCH_CSV");
        bool dump_csv = (bench_csv_env != NULL) && (bench_csv_env[0] != '\0');
        std::string bench_csv_path = dump_csv ? std::string(bench_csv_env) : std::string();
        std::vector<long long> run_us_list;
        std::vector<long long> e2e_us_list;
        if (dump_csv)
        {
            run_us_list.reserve((size_t)repeat);
            e2e_us_list.reserve((size_t)repeat);
        }

        // Record TopK fastest/slowest points for debugging jitter tails.
        std::vector<std::pair<long long, int>> slowest;
        std::vector<std::pair<long long, int>> fastest;
        if (extreme_topk > 0)
        {
            slowest.reserve((size_t)extreme_topk + 1);
            fastest.reserve((size_t)extreme_topk + 1);
        }

        for (int i = 0; i < repeat; i++)
        {
            std::chrono::steady_clock::time_point t0;
            if (dump_csv)
            {
                t0 = std::chrono::steady_clock::now();
            }
            ret = inference_yolo11_model(&rknn_app_ctx, &src_image, &od_results);
            if (ret != 0)
            {
                printf("inference_yolo11_model bench fail! ret=%d\n", ret);
                goto out;
            }
            long long e2e_us = -1;
            if (dump_csv)
            {
                std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
                e2e_us = (long long)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            }

            rknn_perf_run perf_run;
            memset(&perf_run, 0, sizeof(perf_run));
            ret = rknn_query(rknn_app_ctx.rknn_ctx, RKNN_QUERY_PERF_RUN, &perf_run, sizeof(perf_run));
            if (ret == RKNN_SUCC)
            {
                long long us = (long long)perf_run.run_duration;
                sum_us += us;
                if (dump_csv)
                {
                    run_us_list.push_back(us);
                    e2e_us_list.push_back(e2e_us);
                }

                if (extreme_topk > 0)
                {
                    slowest.emplace_back(us, i);
                    std::sort(slowest.begin(), slowest.end(),
                              [](const std::pair<long long, int>& a, const std::pair<long long, int>& b) {
                                  return a.first > b.first;
                              });
                    if ((int)slowest.size() > extreme_topk)
                        slowest.resize((size_t)extreme_topk);

                    fastest.emplace_back(us, i);
                    std::sort(fastest.begin(), fastest.end(),
                              [](const std::pair<long long, int>& a, const std::pair<long long, int>& b) {
                                  return a.first < b.first;
                              });
                    if ((int)fastest.size() > extreme_topk)
                        fastest.resize((size_t)extreme_topk);
                }

                bool dump_iter_detail = false;
                if (perf_detail_every > 0)
                {
                    if ((i % perf_detail_every) == 0)
                    {
                        dump_iter_detail = true;
                    }
                }
                if (dump_iter_detail)
                {
                    if (perf_detail_max == 0 || perf_detail_dumped < perf_detail_max)
                    {
                        rknn_perf_detail perf_detail;
                        memset(&perf_detail, 0, sizeof(perf_detail));
                        int r = rknn_query(rknn_app_ctx.rknn_ctx, RKNN_QUERY_PERF_DETAIL, &perf_detail, sizeof(perf_detail));
                        if (r == RKNN_SUCC && perf_detail.perf_data != NULL)
                        {
                            char fn[64];
                            snprintf(fn, sizeof(fn), "perf_detail_iter_%06d.txt", i);
                            FILE* f = fopen(fn, "wb");
                            if (f)
                            {
                                fwrite(perf_detail.perf_data, 1, strlen(perf_detail.perf_data), f);
                                fclose(f);
                                perf_detail_dumped++;
                            }
                        }
                    }
                }

                bool new_min = false;
                bool new_max = false;
                if (us < min_us)
                {
                    min_us = us;
                    min_i = i;
                    new_min = true;
                }
                if (us > max_us)
                {
                    max_us = us;
                    max_i = i;
                    new_max = true;
                }

                if (dump_minmax_detail && (new_min || new_max))
                {
                    rknn_perf_detail perf_detail;
                    memset(&perf_detail, 0, sizeof(perf_detail));
                    int r = rknn_query(rknn_app_ctx.rknn_ctx, RKNN_QUERY_PERF_DETAIL, &perf_detail, sizeof(perf_detail));
                    if (r == RKNN_SUCC && perf_detail.perf_data != NULL)
                    {
                        if (new_min)
                            min_detail.assign(perf_detail.perf_data);
                        if (new_max)
                            max_detail.assign(perf_detail.perf_data);
                    }
                }
                ok++;
            }
            else
            {
                if (dump_csv)
                {
                    run_us_list.push_back(-1);
                    e2e_us_list.push_back(e2e_us);
                }
            }
        }

        if (ok > 0)
        {
            double avg_ms = (sum_us / (double)ok) / 1000.0;
            double min_ms = min_us / 1000.0;
            double max_ms = max_us / 1000.0;
            printf("Benchmark (perf_run) warmup=%d repeat=%d ok=%d\n", warmup, repeat, ok);
            printf("  avg=%.3f ms, min=%.3f ms, max=%.3f ms\n", avg_ms, min_ms, max_ms);

            // Always show where the extremes happen.
            printf("  min_idx=%d, max_idx=%d\n", min_i, max_i);

            if (dump_minmax_detail)
            {
                if (!min_detail.empty())
                {
                    FILE* f = fopen("perf_detail_min.txt", "wb");
                    if (f)
                    {
                        fwrite(min_detail.data(), 1, min_detail.size(), f);
                        fclose(f);
                    }
                }
                if (!max_detail.empty())
                {
                    FILE* f = fopen("perf_detail_max.txt", "wb");
                    if (f)
                    {
                        fwrite(max_detail.data(), 1, max_detail.size(), f);
                        fclose(f);
                    }
                }

                printf("  perf_detail dumped to perf_detail_min.txt / perf_detail_max.txt\n");
            }

            if (extreme_topk > 0)
            {
                FILE* f = fopen(extremes_csv_path.c_str(), "wb");
                if (f)
                {
                    fprintf(f, "type,rank,iter,perf_run_us,perf_run_ms\n");
                    for (size_t r = 0; r < slowest.size(); r++)
                    {
                        fprintf(f, "slowest,%zu,%d,%lld,%.3f\n", r + 1, slowest[r].second, slowest[r].first, slowest[r].first / 1000.0);
                    }
                    for (size_t r = 0; r < fastest.size(); r++)
                    {
                        fprintf(f, "fastest,%zu,%d,%lld,%.3f\n", r + 1, fastest[r].second, fastest[r].first, fastest[r].first / 1000.0);
                    }
                    fclose(f);
                    printf("  extremes_topk=%d dumped to %s\n", extreme_topk, extremes_csv_path.c_str());
                }
                else
                {
                    printf("Benchmark: failed to open RKNN_BENCH_EXTREMES_CSV=%s\n", extremes_csv_path.c_str());
                }
            }

            if (dump_csv)
            {
                FILE* f = fopen(bench_csv_path.c_str(), "wb");
                if (f)
                {
                    fprintf(f, "iter,perf_run_us,e2e_us\n");
                    for (size_t i = 0; i < run_us_list.size(); i++)
                    {
                        fprintf(f, "%zu,%lld,%lld\n", i, run_us_list[i], e2e_us_list[i]);
                    }
                    fclose(f);
                }
                else
                {
                    printf("Benchmark: failed to open RKNN_BENCH_CSV=%s\n", bench_csv_path.c_str());
                }
            }
        }
        else
        {
            printf("Benchmark: perf_run not available. Please set env RKNN_PERF=1 before running to enable perf collection.\n");
        }
    }
    else
    {
        ret = inference_yolo11_model(&rknn_app_ctx, &src_image, &od_results);
        if (ret != 0)
        {
            printf("inference_yolo11_model fail! ret=%d\n", ret);
            goto out;
        }
    }

    // 画框和概率
    char text[256];
    for (int i = 0; i < od_results.count; i++)
    {
        object_detect_result *det_result = &(od_results.results[i]);
        printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
               det_result->box.left, det_result->box.top,
               det_result->box.right, det_result->box.bottom,
               det_result->prop);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;

        draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);

        sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
        draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);
    }

    if (save_image)
    {
        write_image("out.png", &src_image);
    }

out:
    deinit_post_process();

    ret = release_yolo11_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_yolo11_model fail! ret=%d\n", ret);
    }

    if (src_image.virt_addr != NULL)
    {
#if defined(RV1106_1103) 
        dma_buf_free(rknn_app_ctx.img_dma_buf.size, &rknn_app_ctx.img_dma_buf.dma_buf_fd, 
                rknn_app_ctx.img_dma_buf.dma_buf_virt_addr);
#else
        free(src_image.virt_addr);
#endif
    }

    return 0;
}

/*-------------------------------------------
| 主入口 - 判断模式
|-------------------------------------------*/
int main(int argc, char **argv)
{
    const char* daemon_env = getenv("RKNN_DAEMON");
    if (daemon_env != NULL && daemon_env[0] == '1') {
        // 守护模式
        if (argc < 3) {
            fprintf(stderr, "DAEMON: %s <model_path> <image_path>\n", argv[0]);
            return -1;
        }
        
        // 初始化
        if (daemon_init(argv[1], argv[2]) != 0) {
            return -1;
        }
        
        // 主循环
        daemon_main_loop();
        
        // 清理
        daemon_cleanup();
        
        return 0;
    }
    
    // 普通模式
    return main_normal(argc, argv);
}
