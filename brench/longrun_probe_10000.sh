#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "" ] || [ "${2:-}" = "" ] || [ "${3:-}" = "" ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  echo "USAGE: $0 <DEMO_BIN> <MODEL.rknn> <IMAGE> [OUT_DIR]"
  echo "ENV: WARMUP=20 REPEAT=10000 CPU_AFF=6-7 NPU_MASK=1 PRIO_MODE=nice0 PROBE_INTERVAL=1 PROBE_TOP_EVERY=5"
  exit 1
fi

DEMO_BIN="$1"
MODEL_PATH="$2"
IMAGE_PATH="$3"
OUT_DIR="${4:-./results_longrun}"

WARMUP="${WARMUP:-20}"
REPEAT="${REPEAT:-10000}"
CPU_AFF="${CPU_AFF:-6-7}"
NPU_MASK="${NPU_MASK:-1}"
PRIO_MODE="${PRIO_MODE:-nice0}"
PROBE_INTERVAL="${PROBE_INTERVAL:-1}"
PROBE_TOP_EVERY="${PROBE_TOP_EVERY:-5}"

DEMO_PRINT_PREPROCESS="${DEMO_PRINT_PREPROCESS:-0}"
WORK_DIR="${WORK_DIR:-}"

if [ "$WORK_DIR" = "" ]; then
  model_dir="$(dirname "$MODEL_PATH")"
  if [ -d "$model_dir/.." ]; then
    WORK_DIR="$(cd "$model_dir/.." && pwd)"
  else
    WORK_DIR="$(pwd)"
  fi
fi

now_tag="$(date +%Y%m%d_%H%M%S)"
run_dir="$OUT_DIR/LONGRUN_${now_tag}"
mkdir -p "$run_dir"

info_path="$run_dir/env_info.txt"
probe_path="$run_dir/probe.csv"
bench_csv="$run_dir/bench.csv"
run_log="$run_dir/run.log"
stats_path="$run_dir/stats.txt"

read_first_line() { head -n 1 "$1" 2>/dev/null || true; }

find_npu_node() {
  if [ -d /sys/class/devfreq/fdab0000.npu ]; then
    echo "/sys/class/devfreq/fdab0000.npu"
    return 0
  fi
  if [ -d /sys/class/devfreq/devfreq0 ]; then
    echo "/sys/class/devfreq/devfreq0"
    return 0
  fi
  echo ""
}

NPU_NODE="$(find_npu_node)"
DMC_NODE="/sys/class/devfreq/dmc"

max_from_list() {
  local s="$1"
  echo "$s" | awk '{print $NF}'
}

set_cpu_policy_userspace() {
  local policy="$1"
  local freq="$2"
  local pol_dir="/sys/devices/system/cpu/cpufreq/${policy}"
  if [ ! -d "$pol_dir" ]; then
    return 0
  fi
  echo userspace >"$pol_dir/scaling_governor" || true
  if [ -f "$pol_dir/scaling_setspeed" ]; then
    echo "$freq" >"$pol_dir/scaling_setspeed" || true
  fi
}

lock_cpu_max() {
  for policy in policy0 policy4 policy6; do
    local pol_dir="/sys/devices/system/cpu/cpufreq/${policy}"
    if [ ! -d "$pol_dir" ]; then
      continue
    fi
    local avail
    avail="$(cat "$pol_dir/scaling_available_frequencies" 2>/dev/null || true)"
    if [ "$avail" = "" ]; then
      continue
    fi
    local maxf
    maxf="$(max_from_list "$avail")"
    set_cpu_policy_userspace "$policy" "$maxf"
  done
}

pick_child_pid() {
  local parent_pid="$1"
  local demo_base
  demo_base="$(basename "$DEMO_BIN")"

  local kids
  kids="$(pgrep -P "$parent_pid" 2>/dev/null || true)"
  if [ "$kids" = "" ]; then
    echo ""
    return 0
  fi

  local c
  for c in $kids; do
    local comm
    comm="$(ps -p "$c" -o comm= 2>/dev/null | awk '{print $1}' || true)"
    if [ "$comm" = "$demo_base" ]; then
      echo "$c"
      return 0
    fi
  done

  echo "$(echo "$kids" | head -n 1)"
}

find_leaf_pid() {
  local pid="$1"
  local next
  while true; do
    next="$(pick_child_pid "$pid")"
    if [ "$next" = "" ]; then
      echo "$pid"
      return 0
    fi
    pid="$next"
  done
}

lock_devfreq_userspace_max() {
  local node="$1"
  if [ "$node" = "" ] || [ ! -d "$node" ]; then
    return 0
  fi
  local avail
  avail="$(cat "$node/available_frequencies" 2>/dev/null || true)"
  if [ "$avail" != "" ]; then
    local maxf
    maxf="$(max_from_list "$avail")"
    if [ -f "$node/governor" ]; then
      echo userspace >"$node/governor" || true
    fi
    if [ -f "$node/userspace/set_freq" ]; then
      echo "$maxf" >"$node/userspace/set_freq" || true
    fi
  else
    if [ -f "$node/governor" ]; then
      echo performance >"$node/governor" || true
    fi
  fi
}

read_thermal_one_line() {
  for z in /sys/class/thermal/thermal_zone*/temp; do
    if [ -f "$z" ]; then
      local name="$(cat "${z%/temp}/type" 2>/dev/null || echo "zone")"
      local t="$(cat "$z" 2>/dev/null || echo "")"
      echo -n "${name}=${t} "
    fi
  done
}

write_env_info() {
  {
    echo "date: $(date -R)"
    echo "uname: $(uname -a 2>/dev/null || true)"
    echo "whoami: $(id 2>/dev/null || true)"
    echo "cmd: $DEMO_BIN $MODEL_PATH $IMAGE_PATH $WARMUP $REPEAT"
    echo "WORK_DIR: $WORK_DIR"
    echo "CPU_AFF: $CPU_AFF"
    echo "NPU_MASK: $NPU_MASK"
    echo "PRIO_MODE: $PRIO_MODE"
    echo "RKNN_DEMO_PRINT_PREPROCESS: $DEMO_PRINT_PREPROCESS"
    echo "NPU_NODE: $NPU_NODE"
    echo "DMC_NODE: $DMC_NODE"
    echo "cpu policies cur:"
    for p in /sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq; do
      if [ -f "$p" ]; then
        echo "  $(basename "$(dirname "$p")")=$(cat "$p" 2>/dev/null)"
      fi
    done
    if [ "$NPU_NODE" != "" ] && [ -f "$NPU_NODE/governor" ]; then
      echo "npu governor: $(read_first_line "$NPU_NODE/governor")"
      echo "npu cur_freq: $(read_first_line "$NPU_NODE/cur_freq")"
      echo "npu set_freq: $(read_first_line "$NPU_NODE/userspace/set_freq" 2>/dev/null || true)"
    fi
    if [ -d "$DMC_NODE" ]; then
      echo "dmc governor: $(read_first_line "$DMC_NODE/governor")"
      echo "dmc cur_freq: $(read_first_line "$DMC_NODE/cur_freq")"
      echo "dmc set_freq: $(read_first_line "$DMC_NODE/userspace/set_freq" 2>/dev/null || true)"
    fi
    echo -n "thermal(pre): "
    read_thermal_one_line
    echo
  } >"$info_path"
}

probe_loop() {
  local pid="$1"
  local i=0
  echo "ts,npu_gov,npu_freq,dmc_gov,dmc_freq,cpu_policy6,loadavg,temps,top" >"$probe_path"
  while kill -0 "$pid" 2>/dev/null; do
    i=$((i+1))
    local ts
    ts="$(date +%s.%N)"

    local npu_gov=""; local npu_freq=""
    if [ "$NPU_NODE" != "" ]; then
      npu_gov="$(read_first_line "$NPU_NODE/governor")"
      npu_freq="$(read_first_line "$NPU_NODE/cur_freq")"
    fi

    local dmc_gov=""; local dmc_freq=""
    if [ -d "$DMC_NODE" ]; then
      dmc_gov="$(read_first_line "$DMC_NODE/governor")"
      dmc_freq="$(read_first_line "$DMC_NODE/cur_freq")"
    fi

    local cpu6=""
    if [ -f /sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq ]; then
      cpu6="$(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq 2>/dev/null || true)"
    fi

    local loadavg
    loadavg="$(cat /proc/loadavg 2>/dev/null | awk '{print $1" "$2" "$3}' || true)"

    local temps
    temps="$(read_thermal_one_line | sed 's/ *$//')"

    local top_line=""
    if [ "$((i % PROBE_TOP_EVERY))" = "0" ]; then
      top_line="$(ps -eo pid,comm,pcpu --sort=-pcpu 2>/dev/null | head -n 6 | tr '\n' ';' | tr ',' '.' | sed 's/"/''/g')"
    fi

    echo "${ts},${npu_gov},${npu_freq},${dmc_gov},${dmc_freq},${cpu6},\"${loadavg}\",\"${temps}\",\"${top_line}\"" >>"$probe_path"
    sleep "$PROBE_INTERVAL" || true
  done
}

if [ "$(id -u)" != "0" ]; then
  echo "Please run as root (sudo -E). Output dir: $run_dir"
  exit 2
fi

lock_cpu_max
lock_devfreq_userspace_max "$DMC_NODE"
lock_devfreq_userspace_max "$NPU_NODE"

export RKNN_NPU_CORE_MASK="$NPU_MASK"
if [ "$NPU_MASK" = "1" ]; then
  export RKNN_NPU_CORE="0"
elif [ "$NPU_MASK" = "2" ]; then
  export RKNN_NPU_CORE="1"
elif [ "$NPU_MASK" = "4" ]; then
  export RKNN_NPU_CORE="2"
else
  unset RKNN_NPU_CORE || true
fi

write_env_info

prefix=()
if [ "$CPU_AFF" != "all" ]; then
  prefix+=(taskset -c "$CPU_AFF")
fi

if [[ "$PRIO_MODE" =~ ^nice(-?[0-9]+)$ ]]; then
  prefix=(nice -n "${BASH_REMATCH[1]}" "${prefix[@]}")
fi

if [[ "$PRIO_MODE" =~ ^chrt([0-9]+)$ ]]; then
  chrt_prio="${BASH_REMATCH[1]}"
  prefix=(chrt -f "$chrt_prio" "${prefix[@]}")
fi

(
  if [ -d "$WORK_DIR" ]; then
    cd "$WORK_DIR"
  fi
  RKNN_DEMO_PRINT_PREPROCESS="$DEMO_PRINT_PREPROCESS" RKNN_BENCH_CSV="$bench_csv" "${prefix[@]}" "$DEMO_BIN" "$MODEL_PATH" "$IMAGE_PATH" "$WARMUP" "$REPEAT"
) >"$run_log" 2>&1 &

wrapper_pid=$!

demo_pid=""
for _ in $(seq 1 50); do
  demo_pid="$(find_leaf_pid "$wrapper_pid")"
  if [ "$demo_pid" != "" ]; then
    break
  fi
  sleep 0.05 || true
done

if [ "$demo_pid" = "" ]; then
  demo_pid="$wrapper_pid"
fi

if [ "$CPU_AFF" != "all" ]; then
  taskset -pc "$CPU_AFF" "$demo_pid" >/dev/null 2>&1 || true
fi

{
  echo "wrapper_pid: $wrapper_pid"
  echo "demo_pid: $demo_pid"
  echo "demo affinity:"
  taskset -p "$demo_pid" 2>/dev/null || true
  echo "demo scheduler:"
  chrt -p "$demo_pid" 2>/dev/null || true
} >>"$info_path"

probe_loop "$demo_pid" &
probe_pid=$!

wait "$wrapper_pid" || true

kill "$probe_pid" 2>/dev/null || true

{
  echo -n "thermal(post): "
  read_thermal_one_line
  echo
} >>"$info_path"

if [ -f "$bench_csv" ]; then
  awk -F, 'NR==1{next} $2>=0{ x=$2/1000.0; n++; sum+=x; sumsq+=x*x } END{ if(n>0){ mean=sum/n; var=sumsq/n-mean*mean; std=(var>0)?sqrt(var):0; cv=(mean>0)?(std/mean):0; printf("n=%d\nmean_ms=%.6f\nstd_ms=%.6f\nvar_ms2=%.6f\ncv=%.6f\n", n, mean, std, var, cv) } }' "$bench_csv" >"$stats_path" || true
fi

echo "Done. $run_dir"
