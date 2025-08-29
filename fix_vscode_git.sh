#!/usr/bin/env bash
# fix_vscode_git.sh
# Purpose:
#   1) Create a "clean" git wrapper that ignores LD_LIBRARY_PATH/LD_PRELOAD
#   2) Configure VS Code (workspace) to use that wrapper
#   3) (Optional) Run a safe sync: backup branch + fetch + pull --rebase + push
#
# Usage:
#   bash fix_vscode_git.sh --setup          # create wrapper + write .vscode/settings.json (if absent)
#   bash fix_vscode_git.sh --safe-sync      # perform backup + fetch + pull --rebase + push
#   bash fix_vscode_git.sh --all            # do both
#   bash fix_vscode_git.sh --help
#
set -euo pipefail

WORKSPACE_DIR="$(pwd)"
VSCODE_DIR="${WORKSPACE_DIR}/.vscode"
SETTINGS_JSON="${VSCODE_DIR}/settings.json"

timestamp() { date +"%Y-%m-%d-%H%M%S"; }

detect_system_git() {
  # Prefer system git at /usr/bin/git if present; otherwise fallback to whatever 'command -v git' says
  local sys_git=""
  if [[ -x "/usr/bin/git" ]]; then
    sys_git="/usr/bin/git"
  else
    sys_git="$(command -v git || true)"
  fi
  if [[ -z "${sys_git}" ]]; then
    echo "[ERROR] git not found on this system." >&2
    exit 1
  fi
  echo "${sys_git}"
}

create_wrapper() {
  local sys_git="$1"
  local target="${HOME}/.local/bin/git-clean"
  mkdir -p "$(dirname "${target}")"
  cat > "${target}" <<EOF
#!/usr/bin/env bash
# Clean wrapper for system git to avoid conda/LD pollution
unset LD_LIBRARY_PATH LD_PRELOAD
exec "${sys_git}" "\$@"
EOF
  chmod +x "${target}"
  echo "[OK] Created wrapper: ${target}"
  echo "${target}"
}

configure_vscode_settings() {
  local wrapper_path="$1"
  mkdir -p "${VSCODE_DIR}"
  if [[ -f "${SETTINGS_JSON}" ]]; then
    # Backup existing settings
    cp -f "${SETTINGS_JSON}" "${SETTINGS_JSON}.bak.$(timestamp)"
    echo "[INFO] Existing settings.json backed up to: ${SETTINGS_JSON}.bak.$(timestamp)"
    # Try to minimally patch without jq: if keys exist, leave them; otherwise append near end.
    # We'll generate a merged file safely using a temporary Python helper.
    python3 - "$SETTINGS_JSON" "$wrapper_path" <<'PYCODE'
import json, sys, os, io
settings_path = sys.argv[1]
wrapper = sys.argv[2]
try:
    with open(settings_path, "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception:
    data = {}
# Ensure keys
if "git.path" not in data:
    data["git.path"] = wrapper
else:
    # Overwrite only if it looks obviously wrong (empty or conda path)
    val = str(data.get("git.path",""))
    if ("conda" in val) or (not val.strip()):
        data["git.path"] = wrapper
# Ensure git.env clears LD vars
env = dict(data.get("git.env", {}))
env["LD_LIBRARY_PATH"] = ""
env["LD_PRELOAD"] = ""
data["git.env"] = env
# Helpful diagnostics
data.setdefault("git.trace", True)
data.setdefault("git.trace2", "verbose")
with open(settings_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
print("[OK] Patched VS Code settings:", settings_path)
PYCODE
  else
    # Create new settings.json
    cat > "${SETTINGS_JSON}" <<EOF
{
  "git.path": "${wrapper_path}",
  "git.env": {
    "LD_LIBRARY_PATH": "",
    "LD_PRELOAD": ""
  },
  "git.trace": true,
  "git.trace2": "verbose"
}
EOF
    echo "[OK] Created VS Code settings: ${SETTINGS_JSON}"
  fi
}

safe_sync() {
  # Create a backup branch then fetch + rebase + push
  local sys_git="$1"
  local clean_git="${HOME}/.local/bin/git-clean"
  if [[ ! -x "${clean_git}" ]]; then
    echo "[INFO] Wrapper not found at ${clean_git}, creating..."
    create_wrapper "${sys_git}" >/dev/null
  fi
  # Current branch
  local branch
  branch="$("${clean_git}" rev-parse --abbrev-ref HEAD)"
  local backup="backup/${branch}-$(timestamp)"
  "${clean_git}" fetch origin
  "${clean_git}" branch "${backup}"
  echo "[OK] Backup branch created: ${backup}"
  # Ensure upstream is set
  if ! "${clean_git}" rev-parse --abbrev-ref "@{upstream}" >/dev/null 2>&1; then
    echo "[INFO] Upstream not set; attempting to set upstream to origin/${branch} (if exists)."
    if "${clean_git}" ls-remote --exit-code --heads origin "${branch}" >/dev/null 2>&1; then
      "${clean_git}" branch --set-upstream-to="origin/${branch}" "${branch}" || true
    else
      echo "[WARN] origin/${branch} does not exist. Will skip rebase."
      return 0
    fi
  fi
  # Rebase workflow
  set +e
  "${clean_git}" pull --rebase origin "${branch}"
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[ERROR] Rebase encountered conflicts. Resolve them, then run:"
    echo "  git add <fixed files>"
    echo "  git rebase --continue"
    echo "  git push origin ${branch}"
    exit 2
  fi
  "${clean_git}" push origin "${branch}"
  echo "[OK] Safe sync completed."
}

main() {
  local mode="${1:-"--help"}"
  local sys_git
  sys_git="$(detect_system_git)"
  case "${mode}" in
    --setup)
      local wrapper
      wrapper="$(create_wrapper "${sys_git}")"
      configure_vscode_settings "${wrapper}"
      echo
      echo "[Next] Reload VS Code window. In Command Palette, run: Developer: Reload Window"
      echo "[Tip ] In Output panel (View -> Output), choose 'Git' to verify git.path=${wrapper}"
      ;;
    --safe-sync)
      safe_sync "${sys_git}"
      ;;
    --all)
      local wrapper
      wrapper="$(create_wrapper "${sys_git}")"
      configure_vscode_settings "${wrapper}"
      safe_sync "${sys_git}"
      ;;
    --help|-h)
      sed -n '1,80p' "$0"
      ;;
    *)
      echo "[ERROR] Unknown option: ${mode}"
      sed -n '1,80p' "$0"
      exit 1
      ;;
  esac
}

main "${1:-"--help"}"
