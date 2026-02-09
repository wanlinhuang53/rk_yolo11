#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# YOLO11 Benchmark - 指定配置测试 (支持多模型)
# 用法: ./run_specific_bench.sh [模型路径1] [模型路径2] ...
# =============================================================================

# 默认模型列表 (如果没有提供参数)
if [ "$#" -eq 0 ]; then
    MODELS=(
        "/home/forlinx/Desktop/eval/model/best_i8_drone.rknn"
    )
else
    MODELS=("$@")
fi

IMAGE_PATH="${IMAGE_PATH:-/home/forlinx/Desktop/eval/model/00027.jpg}"
WARMUP="${WARMUP:-20}"
REPEAT="${REPEAT:-200}"

# 硬件配置
NPU_MASK="${NPU_MASK:-keep}"
CPU_AFFINITY="${CPU_AFFINITY:-all}"
NPU_FREQ="${NPU_FREQ:-keep}"
DDR_FREQ="${DDR_FREQ:-keep}"
CPU_FREQ="${CPU_FREQ:-keep}"
PRIO_MODE="${PRIO_MODE:-normal}"

OUTPUT_DIR="${OUTPUT_DIR:-./bench_specific_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_DIR"
DEMO_BIN="${DEMO_BIN:-/home/forlinx/Desktop/eval/rknn_yolo11_demo/rknn_yolo11_demo}"

# -------------------- 检测系统配置 --------------------
if [ -d "/sys/class/devfreq/fdab0000.npu" ]; then
    NPU_NODE="/sys/class/devfreq/fdab0000.npu"
elif [ -d "/sys/class/devfreq/fb000000.npu" ]; then
    NPU_NODE="/sys/class/devfreq/fb000000.npu"
else
    NPU_NODE="/sys/class/devfreq/fdab0000.npu"
fi

if [ -d "/sys/class/devfreq/dmc" ]; then
    DMC_NODE="/sys/class/devfreq/dmc"
elif [ -d "/sys/class/devfreq/fb000000.dmc" ]; then
    DMC_NODE="/sys/class/devfreq/fb000000.dmc"
else
    DMC_NODE="/sys/class/devfreq/dmc"
fi

# -------------------- 帮助信息 --------------------
show_help() {
    cat << EOF
用法: $0 [模型路径1] [模型路径2] ...

环境变量配置:
  NPU_MASK        NPU核掩码 (1=Core0, 2=Core1, 4=Core2, 7=All) [默认: keep]
  CPU_AFFINITY    CPU亲和性 (0-3=小核, 4-7=大核, 4/6=单核, all=所有) [默认: all]
  NPU_FREQ        NPU频率 (数值/performance/keep) [默认: keep]
  DDR_FREQ        DDR频率 (数值/performance/keep) [默认: keep]
  CPU_FREQ        CPU频率 (数值/performance/keep) [默认: keep]
  PRIO_MODE       优先级 (normal/nice-20/chrt90 等) [默认: normal]
  DEMO_BIN        Demo程序路径 [默认: /home/forlinx/Desktop/eval/rknn_yolo11_demo/rknn_yolo11_demo]
  OUTPUT_DIR      输出目录 [默认: ./bench_日期_时间]
  IMAGE_PATH      测试图片 [默认: /home/forlinx/Desktop/eval/model/00027.jpg]
  WARMUP          预热次数 [默认: 20]
  REPEAT          推理次数 [默认: 200]

示例:
  # 单模型 (默认 best_i8_drone.rknn)
  ./run_specific_bench.sh

  # 多模型测试
  ./run_specific_bench.sh /path/to/model1.rknn /path/to/model2.rknn /path/to/model3.rknn

  # 指定多个模型 + NPU配置
  NPU_MASK=4 CPU_AFFINITY=4-7 ./run_specific_bench.sh /path/m1.rknn /path/m2.rknn

  # 锁定最高频率测试多个模型
  NPU_FREQ=performance DDR_FREQ=performance CPU_FREQ=performance ./run_specific_bench.sh model1.rknn model2.rknn model3.rknn

  # 优先级对比测试
  PRIO_MODE=normal   ./run_specific_bench.sh model.rknn
  PRIO_MODE=chrt90   ./run_specific_bench.sh model.rknn

EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    show_help
    exit 0
fi

# -------------------- 频率设置函数 --------------------
set_devfreq() {
    local node="$1"
    local freq="$2"
    local name="$3"

    if [ "$freq" = "keep" ]; then
        echo "[$name] 保持当前频率"
        return
    fi

    if [ ! -d "$node" ]; then
        echo "[$name] 节点不存在，跳过"
        return
    fi

    if [ "$freq" = "performance" ]; then
        if [ -f "$node/available_frequencies" ]; then
            freq=$(tr ' ' '\n' < "$node/available_frequencies" | sort -n | tail -1)
            echo "[$name] 可用最高频率: $freq Hz"
        else
            freq=1000000000
            echo "[$name] 使用默认频率: $freq Hz"
        fi
    else
        echo "[$name] 设置频率: $freq Hz"
    fi

    if [ -w "$node/governor" ]; then
        echo "userspace" > "$node/governor" 2>/dev/null || true
    fi

    if [ -w "$node/userspace/set_freq" ]; then
        echo "$freq" > "$node/userspace/set_freq" 2>/dev/null || true
    elif [ -w "$node/min_freq" ]; then
        echo "$freq" > "$node/min_freq" 2>/dev/null || true
        echo "$freq" > "$node/max_freq" 2>/dev/null || true
    fi
}

set_cpu_freq() {
    local freq="$1"

    if [ "$freq" = "keep" ]; then
        echo "[CPU] 保持当前频率"
        return
    fi

    for pol in /sys/devices/system/cpu/cpufreq/policy*; do
        [ -d "$pol" ] || continue

        if [ "$freq" = "performance" ]; then
            if [ -w "$pol/scaling_governor" ]; then
                echo "performance" > "$pol/scaling_governor" 2>/dev/null || true
            fi
        else
            if [ -w "$pol/scaling_governor" ]; then
                echo "userspace" > "$pol/scaling_governor" 2>/dev/null || true
            fi
            if [ -w "$pol/scaling_setspeed" ]; then
                echo "$freq" > "$pol/scaling_setspeed" 2>/dev/null || true
            fi
        fi
    done
    echo "[CPU] 频率设置完成"
}

# -------------------- 结果汇总文件 --------------------
# 保存输出目录的绝对路径（cd MODEL_DIR 后需要用到）
OUTPUT_DIR_ABS="$(cd "$OUTPUT_DIR" 2>/dev/null && pwd)"
SUMMARY_FILE="$OUTPUT_DIR_ABS/all_models_summary.txt"
> "$SUMMARY_FILE"

# -------------------- 主循环: 遍历每个模型 --------------------
MODEL_COUNT=0
for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_COUNT=$((MODEL_COUNT + 1))

    # 检查模型文件是否存在
    if [ ! -f "$MODEL_PATH" ]; then
        echo "⚠️  模型文件不存在: $MODEL_PATH"
        echo "跳过..."
        continue
    fi

    # 提取模型名称作为子目录
    MODEL_NAME=$(basename "$MODEL_PATH" .rknn)
    MODEL_DIR="$OUTPUT_DIR/${MODEL_COUNT}_${MODEL_NAME}"
    # 保存 MODEL_DIR 的绝对路径（因为后续可能 cd 到别的地方）
    MODEL_DIR_ABS="$(mkdir -p "$MODEL_DIR" && cd "$MODEL_DIR" && pwd)"
    MODEL_DIR="$MODEL_DIR_ABS"

    echo ""
    echo "========================================"
    echo " 测试模型 $MODEL_COUNT / ${#MODELS[@]}"
    echo "========================================"
    echo "  模型: $MODEL_PATH"
    echo "  输出: $MODEL_DIR"
    echo ""

    # 应用频率设置
    set_devfreq "$NPU_NODE" "$NPU_FREQ" "NPU"
    set_devfreq "$DMC_NODE" "$DDR_FREQ" "DDR"
    set_cpu_freq "$CPU_FREQ"

    # 读取当前频率
    npu_freq=$(cat "$NPU_NODE/cur_freq" 2>/dev/null || echo "N/A")
    dmc_freq=$(cat "$DMC_NODE/cur_freq" 2>/dev/null || echo "N/A")
    echo ""
    echo "[当前频率]"
    echo "  NPU: $npu_freq Hz"
    echo "  DDR: $dmc_freq Hz"

    # 设置 NPU 核
    if [ "$NPU_MASK" = "keep" ]; then
        unset RKNN_NPU_CORE_MASK || true
        unset RKNN_NPU_CORE || true
        echo "[NPU核] 保持当前配置"
    else
        export RKNN_NPU_CORE_MASK="$NPU_MASK"
        if [ "$NPU_MASK" = "1" ]; then
            export RKNN_NPU_CORE="0"
            echo "[NPU核] 使用 Core0"
        elif [ "$NPU_MASK" = "2" ]; then
            export RKNN_NPU_CORE="1"
            echo "[NPU核] 使用 Core1"
        elif [ "$NPU_MASK" = "4" ]; then
            export RKNN_NPU_CORE="2"
            echo "[NPU核] 使用 Core2"
        else
            unset RKNN_NPU_CORE || true
            echo "[NPU核] 使用多核 (mask=$NPU_MASK)"
        fi
    fi

    # 构建命令
    cmd=("$DEMO_BIN" "$MODEL_PATH" "$IMAGE_PATH" "$WARMUP" "$REPEAT")
    prefix=()

    if [ "$CPU_AFFINITY" != "all" ]; then
        prefix+=(taskset -c "$CPU_AFFINITY")
        echo "[CPU亲和性] 绑定到 CPU $CPU_AFFINITY"
    fi

    if [[ "$PRIO_MODE" =~ ^nice(-?[0-9]+)$ ]]; then
        prefix=(nice -n "${BASH_REMATCH[1]}" "${prefix[@]}")
        echo "[优先级] nice ${BASH_REMATCH[1]}"
    elif [[ "$PRIO_MODE" =~ ^chrt([0-9]+)$ ]]; then
        if [ "$(id -u)" = "0" ]; then
            prefix=(chrt -f "${BASH_REMATCH[1]}" "${prefix[@]}")
        else
            prefix=(sudo chrt -f "${BASH_REMATCH[1]}" "${prefix[@]}")
        fi
        echo "[优先级] chrt ${BASH_REMATCH[1]}"
    else
        echo "[优先级] normal"
    fi

    # 关闭性能收集 (避免 ~5ms 开销) - 测真实性能
    export RKNN_PERF=0
    export RKNN_PERF_RUN=0
    export RKNN_PERF_DETAIL=0
    # 关闭算子详细信息
    export RKNN_BENCH_DUMP_MINMAX_DETAIL=0
    export RKNN_DEMO_SAVE_IMAGE=0
    export RKNN_DEMO_PRINT_IO=0
    export RKNN_DEMO_PRINT_PREPROCESS=0
    # 保存每次迭代数据到 CSV (用于计算方差)
    export RKNN_BENCH_CSV=bench.csv

    # 运行测试
    cd "$MODEL_DIR"
    echo ""
    echo "[运行测试]"
    echo "命令: ${prefix[@]:-} ${cmd[*]}"
    echo ""

    "${prefix[@]}" "${cmd[@]}" 2>&1 | tee "run.log"

    # 解析结果并添加到汇总
    echo ""
    echo "========================================"
    echo " 模型 $MODEL_NAME 结果摘要"
    echo "========================================"

    if [ -f "bench.csv" ]; then
        python3 - << PY >> "$SUMMARY_FILE"
import csv
import statistics

try:
    with open('bench.csv', 'r') as f:
        reader = csv.DictReader(f)
        times = []
        for row in reader:
            if 'perf_run_us' in row and row['perf_run_us']:
                times.append(float(row['perf_run_us']) / 1000.0)

    if times:
        mean = statistics.mean(times)
        stdev = statistics.stdev(times) if len(times) > 1 else 0
        cv = (stdev / mean * 100) if mean > 0 else 0

        print(f"模型: $MODEL_NAME")
        print(f"  样本数: {len(times)}")
        print(f"  平均值: {mean:.3f} ms")
        print(f"  标准差: {stdev:.3f} ms")
        print(f"  CV: {cv:.2f} %")
        print(f"  范围: {min(times):.3f} - {max(times):.3f} ms")
        print()
except Exception as e:
    print(f"解析失败: {e}")
PY
    fi

done

# -------------------- 最终汇总 --------------------
echo ""
echo "========================================"
echo " 所有模型测试完成"
echo "========================================"
echo ""
echo "[测试配置]"
echo "  NPU Mask: $NPU_MASK"
echo "  CPU亲和性: $CPU_AFFINITY"
echo "  NPU频率: $NPU_FREQ"
echo "  DDR频率: $DDR_FREQ"
echo "  CPU频率: $CPU_FREQ"
echo "  优先级: $PRIO_MODE"
echo ""
echo "[汇总结果]"
if [ -f "$SUMMARY_FILE" ]; then
    cat "$SUMMARY_FILE"
fi
echo ""
echo "详细结果保存在: $OUTPUT_DIR"
echo "========================================"
