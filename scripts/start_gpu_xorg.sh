#!/usr/bin/env bash
set -Eeuo pipefail

DISPLAY_NUM="${SOMAFORCE_XORG_DISPLAY_NUM:-0}"
GEOMETRY="${SOMAFORCE_XORG_GEOMETRY:-1920x1080}"
BUS_ID="${SOMAFORCE_XORG_BUS_ID:-PCI:200:0:0}"
STATE_DIR="${SOMAFORCE_XORG_STATE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.runtime/novnc}"
CONFIG_FILE="${STATE_DIR}/xorg-${DISPLAY_NUM}.conf"
LOG_FILE="${STATE_DIR}/xorg-${DISPLAY_NUM}.log"
PID_FILE="${STATE_DIR}/xorg-${DISPLAY_NUM}.pid"

usage() {
  cat <<'EOF'
Usage:
  scripts/start_gpu_xorg.sh [start|status|stop] [options]

Options:
  --display-num N      Xorg display number, default: 0
  --geometry WxH       Virtual screen size, default: 1920x1080
  --bus-id BUSID       NVIDIA Xorg BusID, default: PCI:200:0:0
  --state-dir PATH     Runtime dir, default: .runtime/novnc
  -h, --help           Show this help.
EOF
}

COMMAND="start"
if (($# > 0)); then
  case "$1" in
    start|status|stop)
      COMMAND="$1"
      shift
      ;;
  esac
fi

while (($# > 0)); do
  case "$1" in
    --display-num)
      DISPLAY_NUM="$2"
      shift 2
      ;;
    --geometry)
      GEOMETRY="$2"
      shift 2
      ;;
    --bus-id)
      BUS_ID="$2"
      shift 2
      ;;
    --state-dir)
      STATE_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

CONFIG_FILE="${STATE_DIR}/xorg-${DISPLAY_NUM}.conf"
LOG_FILE="${STATE_DIR}/xorg-${DISPLAY_NUM}.log"
PID_FILE="${STATE_DIR}/xorg-${DISPLAY_NUM}.pid"

write_config() {
  mkdir -p "$STATE_DIR"
  cat >"$CONFIG_FILE" <<EOF
Section "ServerLayout"
    Identifier "Layout0"
    Screen 0 "Screen0"
EndSection

Section "Device"
    Identifier "Device0"
    Driver "nvidia"
    VendorName "NVIDIA Corporation"
    BusID "${BUS_ID}"
    Option "AllowEmptyInitialConfiguration" "True"
    Option "UseDisplayDevice" "None"
EndSection

Section "Monitor"
    Identifier "Monitor0"
    HorizSync 28.0-80.0
    VertRefresh 48.0-75.0
EndSection

Section "Screen"
    Identifier "Screen0"
    Device "Device0"
    Monitor "Monitor0"
    DefaultDepth 24
    Option "MetaModes" "${GEOMETRY}"
    SubSection "Display"
        Depth 24
        Virtual ${GEOMETRY%x*} ${GEOMETRY#*x}
    EndSubSection
EndSection
EOF
}

running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" >/dev/null 2>&1
}

start_xorg() {
  if DISPLAY=":${DISPLAY_NUM}" xdpyinfo >/dev/null 2>&1; then
    echo "xorg_already_reachable=1"
    echo "display=:${DISPLAY_NUM}"
    exit 0
  fi
  if running; then
    echo "xorg_already_running=1"
    echo "display=:${DISPLAY_NUM}"
    exit 0
  fi

  rm -f "/tmp/.X${DISPLAY_NUM}-lock" "/tmp/.X11-unix/X${DISPLAY_NUM}"
  write_config
  setsid Xorg ":${DISPLAY_NUM}" \
    -config "$CONFIG_FILE" \
    -noreset \
    -novtswitch \
    -sharevts \
    -nolisten tcp \
    -logfile "$LOG_FILE" \
    >/dev/null 2>&1 < /dev/null &
  echo "$!" >"$PID_FILE"
  sleep 4

  if ! running || ! DISPLAY=":${DISPLAY_NUM}" xdpyinfo >/dev/null 2>&1; then
    echo "error: Xorg failed to start; see ${LOG_FILE}" >&2
    tail -120 "$LOG_FILE" >&2 || true
    exit 1
  fi

  echo "xorg_started=1"
  echo "display=:${DISPLAY_NUM}"
  echo "xorg_config=${CONFIG_FILE}"
  echo "xorg_log=${LOG_FILE}"
}

status_xorg() {
  if DISPLAY=":${DISPLAY_NUM}" xdpyinfo >/dev/null 2>&1; then
    echo "xorg_reachable=1"
  else
    echo "xorg_reachable=0"
  fi
  if running; then
    echo "xorg_running=1"
    echo "xorg_pid=$(cat "$PID_FILE")"
  else
    echo "xorg_running=0"
  fi
  echo "display=:${DISPLAY_NUM}"
  echo "xorg_log=${LOG_FILE}"
}

stop_xorg() {
  if running; then
    kill "$(cat "$PID_FILE")" || true
    echo "xorg_stopped=1"
  fi
  rm -f "$PID_FILE"
}

case "$COMMAND" in
  start)
    start_xorg
    ;;
  status)
    status_xorg
    ;;
  stop)
    stop_xorg
    ;;
esac
