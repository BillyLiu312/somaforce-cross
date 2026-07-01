#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

COMMAND="start"
DISPLAY_NUM="${SOMAFORCE_NOVNC_DISPLAY_NUM:-91}"
GEOMETRY="${SOMAFORCE_NOVNC_GEOMETRY:-1920x1080}"
DEPTH="${SOMAFORCE_NOVNC_DEPTH:-24}"
WEB_PORT="${SOMAFORCE_NOVNC_WEB_PORT:-6080}"
BIND_ADDR="${SOMAFORCE_NOVNC_BIND:-127.0.0.1}"
STATE_DIR="${SOMAFORCE_NOVNC_STATE_DIR:-${REPO_ROOT}/.runtime/novnc}"
NOVNC_WEB_ROOT="${SOMAFORCE_NOVNC_WEB_ROOT:-/usr/share/novnc}"
SESSION_CMD="${SOMAFORCE_NOVNC_SESSION:-auto}"
VNC_PASSWORD="${SOMAFORCE_NOVNC_PASSWORD:-}"
VNC_AUTH="${SOMAFORCE_NOVNC_AUTH:-1}"
BACKEND="${SOMAFORCE_NOVNC_BACKEND:-desktop}"
ATTACH_DISPLAY="${SOMAFORCE_NOVNC_ATTACH_DISPLAY:-${DISPLAY:-:0}}"
ATTACH_AUTH="${SOMAFORCE_NOVNC_ATTACH_AUTH:-/dev/null}"
RUN_COMMAND=()

usage() {
  cat <<'EOF'
Usage:
  scripts/novnc_desktop.sh start [options]
  scripts/novnc_desktop.sh run [options] -- <command...>
  scripts/novnc_desktop.sh status [options]
  scripts/novnc_desktop.sh stop [options]

Options:
  --display-num N      X display number to start, default: 91
  --web-port PORT      noVNC HTTP/WebSocket port, default: 6080
  --bind ADDR          noVNC bind address, default: 127.0.0.1
  --geometry WxH       VNC desktop size, default: 1920x1080
  --depth N            VNC color depth, default: 24
  --state-dir PATH     Runtime pid/log directory, default: .runtime/novnc
  --web-root PATH      noVNC web root, default: /usr/share/novnc
  --session COMMAND    X session command, default: auto
  --password PASSWORD  VNC password; generated and stored locally by default
  --no-password        Disable VNC password auth; only use behind trusted access
  --backend MODE       desktop or attach, default: desktop
  --attach-display D   Existing GPU-backed X display for attach mode, default: $DISPLAY or :0
  --attach-auth PATH   Xauthority for attach mode, default: /dev/null
  -h, --help           Show this help.

Examples:
  scripts/novnc_desktop.sh start --web-port 6080

  scripts/novnc_desktop.sh run --web-port 6080 -- \
    conda run -n isaaclab python scripts/record_scaffold_rollout.py \
      --task push_pull_box --interaction-mode push \
      --generate-default-box-usd --record-video --no-headless

Then forward the web port and open:
  http://127.0.0.1:6080/vnc.html?host=127.0.0.1&port=6080&autoconnect=true&resize=scale
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

parse_args() {
  if (($# > 0)); then
    case "$1" in
      start|run|status|stop)
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
      --web-port)
        WEB_PORT="$2"
        shift 2
        ;;
      --bind)
        BIND_ADDR="$2"
        shift 2
        ;;
      --geometry)
        GEOMETRY="$2"
        shift 2
        ;;
      --depth)
        DEPTH="$2"
        shift 2
        ;;
      --state-dir)
        STATE_DIR="$2"
        shift 2
        ;;
      --web-root)
        NOVNC_WEB_ROOT="$2"
        shift 2
        ;;
      --session)
        SESSION_CMD="$2"
        shift 2
        ;;
      --password)
        VNC_PASSWORD="$2"
        VNC_AUTH=1
        shift 2
        ;;
      --no-password)
        VNC_AUTH=0
        shift
        ;;
      --backend)
        BACKEND="$2"
        shift 2
        ;;
      --attach-display)
        ATTACH_DISPLAY="$2"
        shift 2
        ;;
      --attach-auth)
        ATTACH_AUTH="$2"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      --)
        shift
        RUN_COMMAND=("$@")
        break
        ;;
      *)
        die "unknown argument: $1"
        ;;
    esac
  done

  [[ "$DISPLAY_NUM" =~ ^[0-9]+$ ]] || die "--display-num must be an integer"
  [[ "$WEB_PORT" =~ ^[0-9]+$ ]] || die "--web-port must be an integer"
  [[ "$BACKEND" == "desktop" || "$BACKEND" == "attach" ]] || die "--backend must be desktop or attach"
}

vnc_port() {
  echo $((5900 + DISPLAY_NUM))
}

pid_file() {
  echo "${STATE_DIR}/display-${DISPLAY_NUM}.websockify.pid"
}

xstartup_file() {
  echo "${STATE_DIR}/display-${DISPLAY_NUM}.xstartup"
}

websockify_log() {
  echo "${STATE_DIR}/display-${DISPLAY_NUM}.websockify.log"
}

x11vnc_pid_file() {
  echo "${STATE_DIR}/display-${DISPLAY_NUM}.x11vnc.pid"
}

x11vnc_log() {
  echo "${STATE_DIR}/display-${DISPLAY_NUM}.x11vnc.log"
}

vnc_password_file() {
  echo "${STATE_DIR}/display-${DISPLAY_NUM}.vncpasswd"
}

vnc_password_text_file() {
  echo "${STATE_DIR}/display-${DISPLAY_NUM}.password.txt"
}

write_xstartup() {
  mkdir -p "$STATE_DIR"
  local xstartup
  xstartup="$(xstartup_file)"
  cat >"$xstartup" <<EOF
#!/usr/bin/env sh
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS

if [ "$SESSION_CMD" != "auto" ]; then
  exec sh -lc "$SESSION_CMD"
fi

if command -v dbus-launch >/dev/null 2>&1 && command -v startxfce4 >/dev/null 2>&1; then
  exec dbus-launch --exit-with-session startxfce4
fi
if command -v startxfce4 >/dev/null 2>&1; then
  exec startxfce4
fi
if command -v dbus-launch >/dev/null 2>&1 && command -v xfce4-session >/dev/null 2>&1; then
  exec dbus-launch --exit-with-session xfce4-session
fi
if command -v openbox >/dev/null 2>&1; then
  exec openbox
fi
if command -v fluxbox >/dev/null 2>&1; then
  exec fluxbox
fi

xsetroot -solid '#202020'
while true; do sleep 3600; done
EOF
  chmod +x "$xstartup"
}

websockify_running() {
  local file
  file="$(pid_file)"
  [[ -f "$file" ]] && kill -0 "$(cat "$file")" >/dev/null 2>&1
}

print_urls() {
  local port="$WEB_PORT"
  echo "backend=${BACKEND}"
  echo "display=:${DISPLAY_NUM}"
  if [[ "$BACKEND" == "attach" ]]; then
    echo "attach_display=${ATTACH_DISPLAY}"
    echo "attach_auth=${ATTACH_AUTH}"
  fi
  echo "vnc_target=127.0.0.1:$(vnc_port)"
  echo "novnc_bind=${BIND_ADDR}:${port}"
  echo "novnc_url_local=http://127.0.0.1:${port}/vnc.html?host=127.0.0.1&port=${port}&autoconnect=true&resize=scale"
  echo "novnc_url_forwarded=http://localhost:${port}/vnc.html?host=localhost&port=${port}&autoconnect=true&resize=scale"
  echo "novnc_url_direct=http://$(hostname):${port}/vnc.html?host=$(hostname)&port=${port}&autoconnect=true&resize=scale"
  echo "websockify_log=$(websockify_log)"
}

check_deps() {
  if [[ "$BACKEND" == "desktop" ]]; then
    need_cmd vncserver
  else
    need_cmd x11vnc
    need_cmd xdpyinfo
  fi
  need_cmd websockify
  if [[ "$VNC_AUTH" == "1" ]]; then
    need_cmd vncpasswd
  fi
  [[ -f "${NOVNC_WEB_ROOT}/vnc.html" ]] || die "missing noVNC web root: ${NOVNC_WEB_ROOT}/vnc.html"
}

generate_password() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex 4
  else
    python - <<'PY'
import secrets
import string
alphabet = string.ascii_letters + string.digits
print("".join(secrets.choice(alphabet) for _ in range(8)), end="")
PY
  fi
}

ensure_vnc_password() {
  [[ "$VNC_AUTH" == "1" ]] || return 0
  mkdir -p "$STATE_DIR"

  local pass_file pass_text_file password
  pass_file="$(vnc_password_file)"
  pass_text_file="$(vnc_password_text_file)"

  if [[ -n "$VNC_PASSWORD" ]]; then
    password="$VNC_PASSWORD"
  elif [[ -f "$pass_text_file" ]]; then
    password="$(cat "$pass_text_file")"
  else
    password="$(generate_password)"
    printf '%s\n' "$password" >"$pass_text_file"
    chmod 600 "$pass_text_file"
  fi

  printf '%s\n' "$password" | vncpasswd -f >"$pass_file"
  chmod 600 "$pass_file"
}

x11vnc_running() {
  local file
  file="$(x11vnc_pid_file)"
  [[ -f "$file" ]] && kill -0 "$(cat "$file")" >/dev/null 2>&1
}

start_vnc_desktop() {
  local rfb_port="$1"
  local -a vnc_auth_args
  if [[ "$VNC_AUTH" == "1" ]]; then
    vnc_auth_args=(-SecurityTypes VncAuth -PasswordFile "$(vnc_password_file)")
  else
    vnc_auth_args=(-SecurityTypes None)
  fi

  vncserver ":${DISPLAY_NUM}" \
    -useold \
    -localhost yes \
    -rfbport "$rfb_port" \
    "${vnc_auth_args[@]}" \
    -geometry "$GEOMETRY" \
    -depth "$DEPTH" \
    -xstartup "$(xstartup_file)" >/dev/null
}

start_x11vnc_attach() {
  local rfb_port="$1"
  DISPLAY="$ATTACH_DISPLAY" xdpyinfo >/dev/null 2>&1 || {
    die "attach display ${ATTACH_DISPLAY} is not reachable; start or select a GPU-backed Xorg display first"
  }

  if x11vnc_running; then
    echo "x11vnc_already_running=1"
    return 0
  fi

  local -a x11vnc_auth_args
  if [[ "$VNC_AUTH" == "1" ]]; then
    x11vnc_auth_args=(-rfbauth "$(vnc_password_file)")
  else
    x11vnc_auth_args=(-nopw)
  fi

  setsid x11vnc \
    -display "$ATTACH_DISPLAY" \
    -auth "$ATTACH_AUTH" \
    -localhost \
    -no6 \
    -rfbport "$rfb_port" \
    -forever \
    -shared \
    -noxdamage \
    -repeat \
    "${x11vnc_auth_args[@]}" \
    -o "$(x11vnc_log)" \
    >/dev/null 2>&1 < /dev/null &
  echo "$!" >"$(x11vnc_pid_file)"
  sleep 1
  x11vnc_running || die "x11vnc failed to start; see $(x11vnc_log)"
}

start_server() {
  check_deps
  write_xstartup
  ensure_vnc_password

  local rfb_port
  rfb_port="$(vnc_port)"

  if [[ "$BACKEND" == "desktop" ]]; then
    start_vnc_desktop "$rfb_port"
  else
    start_x11vnc_attach "$rfb_port"
  fi

  if websockify_running; then
    echo "websockify_already_running=1"
  else
    if [[ "$BIND_ADDR" != "127.0.0.1" && "$BIND_ADDR" != "localhost" ]]; then
      echo "warning: noVNC is binding to ${BIND_ADDR}; make sure an external auth/firewall layer protects this port." >&2
    fi
    setsid websockify \
      --web "$NOVNC_WEB_ROOT" \
      "${BIND_ADDR}:${WEB_PORT}" \
      "127.0.0.1:${rfb_port}" \
      >"$(websockify_log)" 2>&1 < /dev/null &
    echo "$!" >"$(pid_file)"
    sleep 1
    websockify_running || die "websockify failed to start; see $(websockify_log)"
  fi

  echo "novnc_started=1"
  if [[ "$VNC_AUTH" == "1" ]]; then
    echo "vnc_password=$(cat "$(vnc_password_text_file)")"
  fi
  print_urls
}

stop_server() {
  local file
  file="$(pid_file)"
  if [[ -f "$file" ]]; then
    local pid
    pid="$(cat "$file")"
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" || true
      echo "websockify_stopped=1"
    fi
    rm -f "$file"
  fi

  file="$(x11vnc_pid_file)"
  if [[ -f "$file" ]]; then
    local pid
    pid="$(cat "$file")"
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" || true
      echo "x11vnc_stopped=1"
    fi
    rm -f "$file"
  fi

  vncserver -kill ":${DISPLAY_NUM}" >/dev/null 2>&1 && echo "vnc_stopped=1" || true
}

status_server() {
  echo "vncserver_list:"
  vncserver -list || true
  if x11vnc_running; then
    echo "x11vnc_running=1"
    echo "x11vnc_pid=$(cat "$(x11vnc_pid_file)")"
  else
    echo "x11vnc_running=0"
  fi
  if websockify_running; then
    echo "websockify_running=1"
    echo "websockify_pid=$(cat "$(pid_file)")"
  else
    echo "websockify_running=0"
  fi
  print_urls
}

run_command() {
  ((${#RUN_COMMAND[@]} > 0)) || die "run requires a command after --"
  start_server
  local run_display
  if [[ "$BACKEND" == "attach" ]]; then
    run_display="$ATTACH_DISPLAY"
  else
    run_display=":${DISPLAY_NUM}"
  fi
  echo "running_on_display=${run_display}"
  DISPLAY="$run_display" QT_X11_NO_MITSHM=1 "${RUN_COMMAND[@]}"
}

parse_args "$@"

case "$COMMAND" in
  start)
    start_server
    ;;
  run)
    run_command
    ;;
  status)
    status_server
    ;;
  stop)
    stop_server
    ;;
  *)
    die "unknown command: $COMMAND"
    ;;
esac
