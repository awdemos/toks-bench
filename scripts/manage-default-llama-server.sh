#!/bin/bash
# Manage the permanent default llama-server systemd user service.
# Usage:
#   manage-default-llama-server.sh [start|stop|restart|status]
# Or source it and call default_server_start / default_server_stop.
set -euo pipefail

SERVICE_NAME="llama-server.service"

_default_server_status() {
  systemctl --user is-active "$SERVICE_NAME" >/dev/null 2>&1 && echo "active" || echo "inactive"
}

default_server_start() {
  echo "=== Starting default llama-server ($SERVICE_NAME) ==="
  systemctl --user daemon-reload
  systemctl --user start "$SERVICE_NAME"
  local deadline=$(($(date +%s) + 120))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if curl -s -o /dev/null --max-time 2 http://localhost:8080/v1/models; then
      echo "Default llama-server is up on port 8080"
      return 0
    fi
    sleep 2
  done
  echo "Default llama-server did not become healthy on port 8080"
  return 1
}

default_server_stop() {
  echo "=== Stopping default llama-server ($SERVICE_NAME) ==="
  systemctl --user stop "$SERVICE_NAME" 2>/dev/null || true
  pkill -f "llama-server.*--port 8080" 2>/dev/null || true
  sleep 3
}

default_server_restart() {
  default_server_stop
  default_server_start
}

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  case "${1:-status}" in
    start) default_server_start ;;
    stop) default_server_stop ;;
    restart) default_server_restart ;;
    status) _default_server_status ;;
    *) echo "Usage: $0 {start|stop|restart|status}"; exit 1 ;;
  esac
fi
