#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <demo_bin> <model_path> <image_path> [out_dir] [warmup] [repeat]"
  exit 1
fi

DEMO_BIN="$1"
MODEL_PATH="$2"
IMAGE_PATH="$3"
OUT_DIR="${4:-bench_out_$(date +%Y%m%d_%H%M%S)}"
WARMUP="${5:-20}"
REPEAT="${6:-200}"

mkdir -p "$OUT_DIR"

NPU_MASKS_DEFAULT=(1 2 3 4 5 6 7) # 1=Core0, 2=Core1, 3=Core0+1, 4=Core2, 5=Core0+2, 6=Core1+2, 7=All
# RK3588 Clusters: 0-3 (Small), 4-5 (Big0), 6-7 (Big1), 4-7 (All Big)
CPU_AFFINITIES_DEFAULT=(all 0-3 4-7 4-5 6-7 4 6)

NPU_FREQS_DEFAULT=(keep performance)
DMC_FREQS_DEFAULT=(keep performance)
CPU_FREQS_DEFAULT=(keep performance)
MEM_STRESS_MODES_DEFAULT=(off)

if [ -n "${NPU_MASKS:-}" ]; then
  read -r -a NPU_MASKS_DEFAULT <<<"$NPU_MASKS"
fi

if [ -n "${CPU_AFFINITIES:-}" ]; then
  read -r -a CPU_AFFINITIES_DEFAULT <<<"$CPU_AFFINITIES"
fi

if [ -n "${NPU_FREQS:-}" ]; then
  read -r -a NPU_FREQS_DEFAULT <<<"$NPU_FREQS"
fi

if [ -n "${DMC_FREQS:-}" ]; then
  read -r -a DMC_FREQS_DEFAULT <<<"$DMC_FREQS"
fi

if [ -n "${CPU_FREQS:-}" ]; then
  read -r -a CPU_FREQS_DEFAULT <<<"$CPU_FREQS"
fi

if [ -n "${MEM_STRESS_MODES:-}" ]; then
  read -r -a MEM_STRESS_MODES_DEFAULT <<<"$MEM_STRESS_MODES"
fi

# 设置默认值
STABILITY_OPT="${STABILITY_OPT:-0}"
RK3588_FREQ_MAX="${RK3588_FREQ_MAX:-0}"

# Auto-detect nodes
if [ -d "/sys/class/devfreq/fdab0000.npu" ]; then
    NPU_NODE="/sys/class/devfreq/fdab0000.npu"
elif [ -d "/sys/class/devfreq/fb000000.npu" ]; then
    NPU_NODE="/sys/class/devfreq/fb000000.npu"
else
    NPU_NODE="${NPU_NODE:-/sys/class/devfreq/fdab0000.npu}"
fi

if [ -d "/sys/class/devfreq/dmc" ]; then
    DMC_NODE="/sys/class/devfreq/dmc"
elif [ -d "/sys/class/devfreq/fb000000.dmc" ]; then
    DMC_NODE="/sys/class/devfreq/fb000000.dmc"
else
    DMC_NODE="${DMC_NODE:-/sys/class/devfreq/dmc}"
fi

# Optimization: Disable DDR polling for lower latency (DMC polling_ms)
disable_dmc_polling() {
  if [ -f "$DMC_NODE/polling_ms" ]; then
    if [ -w "$DMC_NODE/polling_ms" ]; then
      echo 0 > "$DMC_NODE/polling_ms" 2>/dev/null || true
    else
      if [ "$(id -u)" = "0" ]; then
        sh -c "echo 0 > $DMC_NODE/polling_ms" 2>/dev/null || true
      else
        sudo sh -c "echo 0 > $DMC_NODE/polling_ms" 2>/dev/null || true
      fi
    fi
    echo "DMC Polling disabled (0ms)"
  fi
}

NPU_GOV="${NPU_GOV:-userspace}"
DMC_GOV="${DMC_GOV:-userspace}"

MEM_STRESS_MB="${MEM_STRESS_MB:-512}"
MEM_STRESS_WORKERS="${MEM_STRESS_WORKERS:-1}"
COOL_DOWN_SEC="${COOL_DOWN_SEC:-0}"

PERF_DETAIL_EVERY="${PERF_DETAIL_EVERY:-0}"
PERF_DETAIL_MAX="${PERF_DETAIL_MAX:-0}"

EXP_DMC_FREQ="${EXP_DMC_FREQ:-2112000000}"
EXP_NPU_FREQ="${EXP_NPU_FREQ:-950000000}"

# OFAT (one-factor-at-a-time) defaults
OFAT_BASE_NPU_MASK="${OFAT_BASE_NPU_MASK:-keep}"
OFAT_BASE_CPU_AFF="${OFAT_BASE_CPU_AFF:-all}"
OFAT_BASE_PRIO="${OFAT_BASE_PRIO:-normal}"
OFAT_BASE_NPU_FREQ="${OFAT_BASE_NPU_FREQ:-keep}"
OFAT_BASE_DMC_FREQ="${OFAT_BASE_DMC_FREQ:-keep}"
OFAT_BASE_CPU_FREQ="${OFAT_BASE_CPU_FREQ:-keep}"
OFAT_MEM_STRESS="${OFAT_MEM_STRESS:-off}"

# Frequency sweep lists (space-separated). Each entry can be a number, keep, or performance.
OFAT_CPU_FREQS_DEFAULT=(keep performance)
OFAT_DMC_FREQS_DEFAULT=(keep performance)
OFAT_NPU_FREQS_DEFAULT=(keep performance)
if [ -n "${OFAT_CPU_FREQS:-}" ]; then
  read -r -a OFAT_CPU_FREQS_DEFAULT <<<"$OFAT_CPU_FREQS"
fi
if [ -n "${OFAT_DMC_FREQS:-}" ]; then
  read -r -a OFAT_DMC_FREQS_DEFAULT <<<"$OFAT_DMC_FREQS"
fi
if [ -n "${OFAT_NPU_FREQS:-}" ]; then
  read -r -a OFAT_NPU_FREQS_DEFAULT <<<"$OFAT_NPU_FREQS"
fi

OFAT_PRIO_MODES_DEFAULT=(normal)
if [ -n "${OFAT_PRIO_MODES:-}" ]; then
  read -r -a OFAT_PRIO_MODES_DEFAULT <<<"$OFAT_PRIO_MODES"
fi

# Core sweep lists (space-separated). Each entry can be keep, 1, 2, 4, 7 ...
OFAT_NPU_MASKS_DEFAULT=(keep 1 2 4 7)
OFAT_CPU_AFFINITIES_DEFAULT=(all 0-3 4-7 4-5 6-7 4 6)
if [ -n "${OFAT_NPU_MASKS:-}" ]; then
  read -r -a OFAT_NPU_MASKS_DEFAULT <<<"$OFAT_NPU_MASKS"
fi
if [ -n "${OFAT_CPU_AFFINITIES:-}" ]; then
  read -r -a OFAT_CPU_AFFINITIES_DEFAULT <<<"$OFAT_CPU_AFFINITIES"
fi

sanitize_id() {
  local s="$1"
  # keep letters/numbers/_/- only
  echo "$s" | tr -c 'A-Za-z0-9_\n\r\t-' '_' | tr -s '_'
}

read_first_line() {
  local p="$1"
  if [ -f "$p" ]; then
    head -n 1 "$p" 2>/dev/null || true
  else
    echo ""
  fi
}

read_all_thermal() {
  local out=""
  for z in /sys/class/thermal/thermal_zone*/temp; do
    if [ -f "$z" ]; then
      out+="$(basename "$(dirname "$z")")=$(cat "$z" 2>/dev/null) "
    fi
  done
  echo "$out"
}

now_ms() {
  python3 - <<'PY'
import time
print(int(time.time()*1000))
PY
}

RUNS_CSV="$OUT_DIR/runs.csv"
if [ ! -f "$RUNS_CSV" ]; then
  echo "run_id,ts_ms,npu_mask,cpu_affinity,prio_mode,npu_freq_target,dmc_freq_target,cpu_freq_target,mem_stress,npu_gov,npu_freq,dmc_gov,dmc_freq,cpu_policies_freq,temps_pre,temps_post,csv_path,log_path,perf_detail_min,perf_detail_max" > "$RUNS_CSV"
fi

# 关闭性能收集（避免 ~5ms 开销）- 测真实性能
export RKNN_PERF=0
# 关闭所有打印
export RKNN_PERF_RUN=0
export RKNN_PERF_DETAIL=0
# 关闭算子详细信息
export RKNN_BENCH_DUMP_MINMAX_DETAIL=0
export RKNN_DEMO_SAVE_IMAGE=0
export RKNN_DEMO_PRINT_IO=0
export RKNN_DEMO_PRINT_PREPROCESS=0
# 保存每次迭代数据到 CSV（用于计算方差）
export RKNN_BENCH_CSV=bench.csv

PRIO_MODES=(normal)
if [ "${USE_CHRT:-0}" = "1" ]; then
  PRIO_MODES+=(chrt90)
fi

set_devfreq_userspace_lock() {
  local node="$1"
  local gov="$2"
  local freq="$3"
  if [ "$freq" = "keep" ]; then
    return 0
  fi
  
  if [ ! -d "$node" ]; then
    return 0
  fi

  # Support "performance" as a shortcut
  if [ "$freq" = "performance" ]; then
    if [ -f "$node/available_frequencies" ]; then
      freq=$(tr ' ' '\n' < "$node/available_frequencies" | sort -n | tail -1)
    else
      # Default to a safe high frequency for RK3588 if available_frequencies is missing
      freq=1000000000
    fi
  fi

  if [ -w "$node/governor" ]; then
    echo "$gov" > "$node/governor" 2>/dev/null || true
  else
    if [ "$(id -u)" = "0" ]; then
      sh -c "echo '$gov' > '$node/governor'" 2>/dev/null || true
    else
      sudo sh -c "echo '$gov' > '$node/governor'" 2>/dev/null || true
    fi
  fi

  if [ -w "$node/userspace/set_freq" ]; then
     echo "$freq" > "$node/userspace/set_freq" 2>/dev/null || true
  elif [ -w "$node/min_freq" ]; then
    echo "$freq" > "$node/min_freq" 2>/dev/null || true
    echo "$freq" > "$node/max_freq" 2>/dev/null || true
  else
    if [ "$(id -u)" = "0" ]; then
      sh -c "echo '$freq' > '$node/min_freq' && echo '$freq' > '$node/max_freq'" 2>/dev/null || true
    else
      sudo sh -c "echo '$freq' > '$node/min_freq' && echo '$freq' > '$node/max_freq'" 2>/dev/null || true
    fi
  fi
}

set_cpu_policies_freq() {
  local freq="$1"
  if [ "$freq" = "keep" ]; then
    return 0
  fi
  
  for pol in /sys/devices/system/cpu/cpufreq/policy*; do
    if [ ! -d "$pol" ]; then
      continue
    fi

    local target_gov="userspace"
    local target_freq="$freq"

    if [ "$freq" = "performance" ]; then
        target_gov="performance"
        # In performance mode, we don't need to setspeed
    fi

    if [ -w "$pol/scaling_governor" ]; then
      echo "$target_gov" > "$pol/scaling_governor" 2>/dev/null || true
    else
      if [ "$(id -u)" = "0" ]; then
        sh -c "echo '$target_gov' > '$pol/scaling_governor'" 2>/dev/null || true
      else
        sudo sh -c "echo '$target_gov' > '$pol/scaling_governor'" 2>/dev/null || true
      fi
    fi

    if [ "$target_gov" = "userspace" ] && [ -f "$pol/scaling_setspeed" ]; then
      if [ -w "$pol/scaling_setspeed" ]; then
        echo "$target_freq" > "$pol/scaling_setspeed" 2>/dev/null || true
      else
        if [ "$(id -u)" = "0" ]; then
          sh -c "echo '$target_freq' > '$pol/scaling_setspeed'" 2>/dev/null || true
        else
          sudo sh -c "echo '$target_freq' > '$pol/scaling_setspeed'" 2>/dev/null || true
        fi
      fi
    fi
  done
}

start_mem_stress() {
  local mode="$1"
  if [ "$mode" != "on" ]; then
    echo ""
    return 0
  fi
  python3 - <<PY >/dev/null 2>&1 &
import os, time
workers = int(os.environ.get('MEM_STRESS_WORKERS','1'))
mb = int(os.environ.get('MEM_STRESS_MB','512'))
sz = mb*1024*1024
bufs = []
for _ in range(max(1,workers)):
    a = bytearray(sz)
    b = bytearray(sz)
    for i in range(0, sz, 4096):
        a[i] = (i//4096) & 0xFF
    bufs.append((a,b))
v = 0
while True:
    for a,b in bufs:
        b[:] = a
        v ^= b[0]
    time.sleep(0)
PY
  echo "$!"
}

stop_mem_stress() {
  local pid="$1"
  if [ -z "$pid" ]; then
    return 0
  fi
  kill "$pid" >/dev/null 2>&1 || true
  wait "$pid" >/dev/null 2>&1 || true
}

run_one() {
  local run_id="$1"
  local npu_mask="$2"
  local cpu_aff="$3"
  local prio_mode="$4"
  local npu_freq_target="$5"
  local dmc_freq_target="$6"
  local cpu_freq_target="$7"
  local mem_stress="$8"

  local run_dir="$OUT_DIR/$run_id"
  mkdir -p "$run_dir"

  set_devfreq_userspace_lock "$NPU_NODE" "$NPU_GOV" "$npu_freq_target"
  set_devfreq_userspace_lock "$DMC_NODE" "$DMC_GOV" "$dmc_freq_target"
  set_cpu_policies_freq "$cpu_freq_target"

  local npu_gov="$(read_first_line "$NPU_NODE/governor")"
  local npu_freq="$(read_first_line "$NPU_NODE/cur_freq")"
  local dmc_gov="$(read_first_line "$DMC_NODE/governor")"
  local dmc_freq="$(read_first_line "$DMC_NODE/cur_freq")"
  local temps_pre="$(read_all_thermal)"

  local cpu_policies_freq=""
  for p in /sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq; do
    if [ -f "$p" ]; then
      cpu_policies_freq+="$(basename "$(dirname "$p")")=$(cat "$p" 2>/dev/null) "
    fi
  done

  if [ "$npu_mask" = "keep" ]; then
    unset RKNN_NPU_CORE_MASK || true
    unset RKNN_NPU_CORE || true
  else
    export RKNN_NPU_CORE_MASK="$npu_mask"
    if [ "$npu_mask" = "1" ]; then
      export RKNN_NPU_CORE="0"
    elif [ "$npu_mask" = "2" ]; then
      export RKNN_NPU_CORE="1"
    elif [ "$npu_mask" = "4" ]; then
      export RKNN_NPU_CORE="2"
    else
      unset RKNN_NPU_CORE || true
    fi
  fi

  local csv_path="bench.csv"
  local log_path="$run_dir/run.log"
  local perf_min="$run_dir/perf_detail_min.txt"
  local perf_max="$run_dir/perf_detail_max.txt"
  local temps_post_path="$run_dir/temps_post.txt"

  local cmd=("$DEMO_BIN" "$MODEL_PATH" "$IMAGE_PATH" "$WARMUP" "$REPEAT")
  local prefix=()

  if [ "$cpu_aff" != "all" ]; then
    prefix+=(taskset -c "$cpu_aff")
  fi

  if [[ "$prio_mode" =~ ^nice(-?[0-9]+)$ ]]; then
    prefix=(nice -n "${BASH_REMATCH[1]}" "${prefix[@]}")
  fi

  if [[ "$prio_mode" =~ ^chrt([0-9]+)$ ]]; then
    local chrt_prio="${BASH_REMATCH[1]}"
    if [ "$(id -u)" = "0" ]; then
      prefix=(chrt -f "$chrt_prio" "${prefix[@]}")
    else
      prefix=(sudo chrt -f "$chrt_prio" "${prefix[@]}")
    fi
  fi

  local stress_pid
  stress_pid="$(start_mem_stress "$mem_stress")"

  ( 
    cd "$run_dir"
    RKNN_BENCH_CSV="$csv_path" RKNN_BENCH_PERF_DETAIL_EVERY="$PERF_DETAIL_EVERY" RKNN_BENCH_PERF_DETAIL_MAX="$PERF_DETAIL_MAX" "${prefix[@]}" "${cmd[@]}" 
  ) >"$log_path" 2>&1 || true

  stop_mem_stress "$stress_pid"
  read_all_thermal > "$temps_post_path"

  true

  local ts_ms
  ts_ms="$(now_ms)"

  local temps_post
  temps_post="$(cat "$temps_post_path" 2>/dev/null || echo "N/A")"

  echo "$run_id,$ts_ms,$npu_mask,$cpu_aff,$prio_mode,$npu_freq_target,$dmc_freq_target,$cpu_freq_target,$mem_stress,$npu_gov,$npu_freq,$dmc_gov,$dmc_freq,\"$cpu_policies_freq\",\"$temps_pre\",\"$temps_post\",$csv_path,$log_path,$perf_min,$perf_max" >> "$RUNS_CSV"

  if [ "$COOL_DOWN_SEC" != "0" ]; then
    sleep "$COOL_DOWN_SEC" || true
  fi
}

# --- Mode Selection ---
# Set ABLATION=1 to run the 8-step factor isolation
# Set CORE_SWEEP=1 to run the exhaustive core combinations
ABLATION="${ABLATION:-0}"
CORE_SWEEP="${CORE_SWEEP:-1}"
OFAT="${OFAT:-0}"

if [ "$OFAT" = "1" ]; then
    echo "🧪 [Mode] OFAT (One-Factor-At-A-Time) with multi-frequency sweeps"

    # Baseline
    run_one "OFAT_0_Base" "$OFAT_BASE_NPU_MASK" "$OFAT_BASE_CPU_AFF" "$OFAT_BASE_PRIO" \
      "$OFAT_BASE_NPU_FREQ" "$OFAT_BASE_DMC_FREQ" "$OFAT_BASE_CPU_FREQ" "$OFAT_MEM_STRESS"

    # CPU frequency sweep
    for f in "${OFAT_CPU_FREQS_DEFAULT[@]}"; do
      rid="OFAT_CPUFreq__$(sanitize_id "$f")"
      run_one "$rid" "$OFAT_BASE_NPU_MASK" "$OFAT_BASE_CPU_AFF" "$OFAT_BASE_PRIO" \
        "$OFAT_BASE_NPU_FREQ" "$OFAT_BASE_DMC_FREQ" "$f" "$OFAT_MEM_STRESS"
    done

    # DDR/DMC frequency sweep
    for f in "${OFAT_DMC_FREQS_DEFAULT[@]}"; do
      rid="OFAT_DDRFreq__$(sanitize_id "$f")"
      run_one "$rid" "$OFAT_BASE_NPU_MASK" "$OFAT_BASE_CPU_AFF" "$OFAT_BASE_PRIO" \
        "$OFAT_BASE_NPU_FREQ" "$f" "$OFAT_BASE_CPU_FREQ" "$OFAT_MEM_STRESS"
    done

    # NPU frequency sweep
    for f in "${OFAT_NPU_FREQS_DEFAULT[@]}"; do
      rid="OFAT_NPUFreq__$(sanitize_id "$f")"
      run_one "$rid" "$OFAT_BASE_NPU_MASK" "$OFAT_BASE_CPU_AFF" "$OFAT_BASE_PRIO" \
        "$f" "$OFAT_BASE_DMC_FREQ" "$OFAT_BASE_CPU_FREQ" "$OFAT_MEM_STRESS"
    done

    # NPU core mask sweep
    for m in "${OFAT_NPU_MASKS_DEFAULT[@]}"; do
      rid="OFAT_NPUCore__mask_$(sanitize_id "$m")"
      run_one "$rid" "$m" "$OFAT_BASE_CPU_AFF" "$OFAT_BASE_PRIO" \
        "$OFAT_BASE_NPU_FREQ" "$OFAT_BASE_DMC_FREQ" "$OFAT_BASE_CPU_FREQ" "$OFAT_MEM_STRESS"
    done

    # CPU affinity sweep
    for a in "${OFAT_CPU_AFFINITIES_DEFAULT[@]}"; do
      rid="OFAT_CPUAff__$(sanitize_id "$a")"
      run_one "$rid" "$OFAT_BASE_NPU_MASK" "$a" "$OFAT_BASE_PRIO" \
        "$OFAT_BASE_NPU_FREQ" "$OFAT_BASE_DMC_FREQ" "$OFAT_BASE_CPU_FREQ" "$OFAT_MEM_STRESS"
    done

    # Priority sweep
    for p in "${OFAT_PRIO_MODES_DEFAULT[@]}"; do
      rid="OFAT_Prio__$(sanitize_id "$p")"
      run_one "$rid" "$OFAT_BASE_NPU_MASK" "$OFAT_BASE_CPU_AFF" "$p" \
        "$OFAT_BASE_NPU_FREQ" "$OFAT_BASE_DMC_FREQ" "$OFAT_BASE_CPU_FREQ" "$OFAT_MEM_STRESS"
    done
fi

if [ "$ABLATION" = "1" ]; then
    echo "🧪 [Mode] Ablation Study (Factor Isolation)"
    # EXP_0: Baseline (All dynamic)
    run_one "EXP_0_Baseline" "keep" "all" "normal" "keep" "keep" "keep" "off"
    # EXP_1: +CPU Freq (Fixed Performance)
    run_one "EXP_1_CPU_Freq" "keep" "all" "normal" "keep" "keep" "performance" "off"
    # EXP_2: +DDR Freq (Fixed Performance)
    run_one "EXP_2_DDR_Freq" "keep" "all" "normal" "keep" "$EXP_DMC_FREQ" "performance" "off"
    # EXP_3: +NPU Freq (Fixed Performance)
    run_one "EXP_3_NPU_Freq" "keep" "all" "normal" "$EXP_NPU_FREQ" "$EXP_DMC_FREQ" "performance" "off"
    # EXP_4: +DDR Polling Opt (polling_ms=0)
    disable_dmc_polling
    run_one "EXP_4_DMC_Poll" "keep" "all" "normal" "$EXP_NPU_FREQ" "$EXP_DMC_FREQ" "performance" "off"
    # EXP_5: +NPU Core (Fix to Core 0 for minimum inter-core sync jitter)
    run_one "EXP_5_NPU_Core0" "1" "all" "normal" "$EXP_NPU_FREQ" "$EXP_DMC_FREQ" "performance" "off"
    # EXP_6: +CPU Affinity (Fix to Big Cores for minimum task submission jitter)
    run_one "EXP_6_CPU_Big" "1" "4-7" "normal" "$EXP_NPU_FREQ" "$EXP_DMC_FREQ" "performance" "off"
    # EXP_7: +Realtime Priority (chrt -f 90)
    run_one "EXP_7_Priority" "1" "4-7" "chrt90" "$EXP_NPU_FREQ" "$EXP_DMC_FREQ" "performance" "off"
fi

# =============================================================================
#                           稳定性策略优化模式
# 目标：系统测试所有影响因素，以方差为指标找到最优策略
# 影响因素：NPU/CPU/DDR频率、多核、组合、调度、优先级
# =============================================================================
if [ "$STABILITY_OPT" = "1" ]; then
    echo "🎯 [Mode] Stability Strategy Optimization (Variance-Based)"
    
    # 基础配置：所有频率锁定为最高
    echo "📌 Locking all frequencies to maximum for controlled testing..."
    set_devfreq_userspace_lock "$NPU_NODE" "$NPU_GOV" "performance"
    set_devfreq_userspace_lock "$DMC_NODE" "$DMC_GOV" "performance"
    set_cpu_policies_freq "performance"
    
    # 策略1：基准测试（全动态）
    echo "=== Strategy 1: Baseline (All Dynamic) ==="
    run_one "STAB_01_Baseline" "keep" "all" "normal" "keep" "keep" "keep" "off"
    
    # 策略2：全锁定频率
    echo "=== Strategy 2: All Frequencies Locked ==="
    run_one "STAB_02_AllLocked" "keep" "all" "normal" "performance" "performance" "performance" "off"
    
    # 策略3：单NPU核（避免多核同步抖动）
    echo "=== Strategy 3: Single NPU Core ==="
    for mask in 1 2 4; do
        core=$((mask - 1))
        run_one "STAB_03_NPU${core}" "$mask" "all" "normal" "performance" "performance" "performance" "off"
    done
    
    # 策略4：单CPU核（避免任务切换抖动）
    echo "=== Strategy 4: Single CPU Core ==="
    for cpu in 0 4 6; do
        run_one "STAB_04_CPU${cpu}" "keep" "$cpu" "normal" "performance" "performance" "performance" "off"
    done
    
    # 策略5：CPU+NPU组合（测试关联性）
    echo "=== Strategy 5: CPU+NPU Combinations ==="
    # 小核+单NPU
    for cpu in 0 1 2 3; do
        for mask in 1 2 4; do
            run_one "STAB_05_LC${cpu}_NPU$((mask-1))" "$mask" "$cpu" "normal" "performance" "performance" "performance" "off"
        done
    done
    # 大核+单NPU
    for cpu in 4 5 6 7; do
        for mask in 1 2 4; do
            run_one "STAB_05_BC${cpu}_NPU$((mask-1))" "$mask" "$cpu" "normal" "performance" "performance" "performance" "off"
        done
    done
    
    # 策略6：进程优先级优化
    echo "=== Strategy 6: Priority Optimization ==="
    # Nice值优化
    for nice in -20 -10 0 10; do
        run_one "STAB_06_Nice${nice}" "keep" "6" "nice${nice}" "performance" "performance" "performance" "off"
    done
    # 实时优先级
    for rt in 50 90 99; do
        run_one "STAB_06_RT${rt}" "keep" "6" "chrt${rt}" "performance" "performance" "performance" "off"
    done
    
    # 策略7：DDR调度优化（关闭轮询）
    echo "=== Strategy 7: DDR Scheduling Optimization ==="
    disable_dmc_polling
    run_one "STAB_07_NoPolling" "keep" "6" "normal" "performance" "performance" "performance" "off"
    
    # 策略8：最优组合（基于前期结果的最佳配置）
    echo "=== Strategy 8: Best Combination (Empirical) ==="
    # 假设前期实验表明：大核+单NPU+实时优先级最稳定
    run_one "STAB_08_BestCombo" "1" "6" "chrt90" "performance" "performance" "performance" "off"
    
    echo "✅ Stability Strategy Optimization Complete"
    echo "📊 Use analyze_bench.py on $OUT_DIR to compare variances"
fi

if [ "$CORE_SWEEP" = "1" ]; then
    echo "🏁 [Mode] Core Sweep (All CPU/NPU Core Combinations)"
    # In this mode, we lock frequencies for stability
    set_devfreq_userspace_lock "$NPU_NODE" "$NPU_GOV" "performance"
    set_devfreq_userspace_lock "$DMC_NODE" "$DMC_GOV" "performance"
    set_cpu_policies_freq "performance"
    
    idx=0
    for npu_mask in "${NPU_MASKS_DEFAULT[@]}"; do
        for cpu_aff in "${CPU_AFFINITIES_DEFAULT[@]}"; do
            run_id=$(printf "CORE_%02d__npu_%s__cpu_%s" "$idx" "$npu_mask" "$cpu_aff")
            echo "[RUN] $run_id"
            run_one "$run_id" "$npu_mask" "$cpu_aff" "normal" "keep" "keep" "keep" "off"
            idx=$((idx+1))
        done
    done
fi

echo "Done. Output: $OUT_DIR"
