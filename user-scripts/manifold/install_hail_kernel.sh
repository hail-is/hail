#!/usr/bin/env bash
#
# install_hail_kernel.sh
# -----------------------------------------------------------------------------
# Sets up (or restores) a PERSISTENT Hail Jupyter kernel on a Manifold compute
# environment running JupyterLab.
#
# WHY THIS IS NEEDED
#   `conda create -n hail` puts the env on the container's ephemeral layer
#   (e.g. /opt/conda/envs/hail) and `ipykernel install --user` writes the
#   kernelspec to ~/.local/share/jupyter (also ephemeral). Both are wiped when
#   the compute environment restarts, which is why the kernel keeps vanishing.
#
# WHAT THIS SCRIPT DOES
#   1. Picks a directory that survives restarts (a persistent volume).
#   2. Creates the conda env BY PATH on that volume (the slow, heavy part).
#   3. Re-registers the kernelspec each run pointing at the persistent python,
#      and bakes PATH into the kernelspec so the in-notebook PATH hack is no
#      longer needed.
#   4. Is idempotent: if the env already exists it skips the rebuild and just
#      re-registers the kernel in a couple of seconds.
#
# USAGE
#   bash install_hail_kernel.sh                      # create if missing, register
#   PERSIST_DIR=/your/persistent/path bash install_hail_kernel.sh   # force a path
#   FORCE_RECREATE=1 bash install_hail_kernel.sh     # delete & rebuild the env
#
# AFTER A RESTART
#   Just run it again. If the env survived it restores in seconds; if not,
#   the PERSIST_DIR you used was not actually persistent — try another path
#   (see the printed candidate list) and pass it via PERSIST_DIR=...
# -----------------------------------------------------------------------------

set -euo pipefail

# ---- Config -----------------------------------------------------------------
ENV_NAME="hail"
PY_VERSION="3.11"
DISPLAY_NAME="Python (hail)"

log() { printf '>> %s\n' "$*"; }

# ---- 1. Choose a persistent base directory ----------------------------------
# Defaults to ~/local  (i.e. /home/jovyan/local on this environment).
# Override with PERSIST_DIR=... to use a different volume.
PERSIST_DIR="${PERSIST_DIR:-$HOME/local}"
mkdir -p "$PERSIST_DIR"
ENV_PREFIX="$PERSIST_DIR/conda-envs/$ENV_NAME"

log "Persistent base : $PERSIST_DIR"
log "Conda env prefix: $ENV_PREFIX"
log "(If this base does not survive a restart, re-run with PERSIST_DIR=...)"

# ---- 2. Make conda available ------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  for cbase in /opt/conda "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3"; do
    if [[ -f "$cbase/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1091
      source "$cbase/etc/profile.d/conda.sh"
      break
    fi
  done
fi
command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not found on PATH." >&2; exit 1; }
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# ---- 3. Create the env (heavy) or restore (fast) ----------------------------
if [[ "${FORCE_RECREATE:-0}" == "1" && -d "$ENV_PREFIX" ]]; then
  log "FORCE_RECREATE=1 — removing existing env"
  conda env remove --prefix "$ENV_PREFIX" -y || true
fi

mkdir -p "$PERSIST_DIR/conda-envs"

if [[ -x "$ENV_PREFIX/bin/python" ]]; then
  log "Conda env already present — fast restore path (skipping create/install)."
  conda activate "$ENV_PREFIX"
else
  log "Creating conda env on persistent volume (slow part, ~minutes)…"
  conda create --prefix "$ENV_PREFIX" "python==$PY_VERSION" -y
  conda activate "$ENV_PREFIX"
  log "Installing hail + ipykernel…"
  pip install hail
  pip install ipykernel   # a 'decorator' dependency clash may print — safe to ignore
fi

PYBIN="$ENV_PREFIX/bin/python"

# ---- 4. Register the kernelspec ---------------------------------------------
# Written by hand so argv points at the persistent python and PATH is baked in,
# which removes the need for the os.environ['PATH'] = ... step inside notebooks.
KERNEL_DIR="$HOME/.local/share/jupyter/kernels/$ENV_NAME"
mkdir -p "$KERNEL_DIR"
cat > "$KERNEL_DIR/kernel.json" <<JSON
{
  "argv": ["$PYBIN", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
  "display_name": "$DISPLAY_NAME",
  "language": "python",
  "metadata": { "debugger": true },
  "env": {
    "PATH": "$ENV_PREFIX/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  }
}
JSON
log "Kernelspec written: $KERNEL_DIR/kernel.json"

# ---- 4b. Download Hail logo -------------------------------------------------
HAIL_SVG_URL="https://raw.githubusercontent.com/hail-is/hail/main/graphics/svgs/icons/helix-w.svg"
if curl -fsSL "$HAIL_SVG_URL" -o "$KERNEL_DIR/logo-svg.svg" 2>/dev/null; then
  log "Hail logo downloaded (logo-svg.svg)"
else
  log "WARN: Could not download Hail logo — kernel will use the default icon"
fi

# ---- 5. Validate ------------------------------------------------------------
log "Validating…"
if "$PYBIN" -c "import hail; print('  hail version:', hail.__version__)"; then
  :
else
  echo "  WARN: 'import hail' failed — check the pip install output above." >&2
fi

if "$PYBIN" -m hailtop.aiotools.copy --help >/dev/null 2>&1; then
  log "hailtop CLI OK"
else
  echo "  WARN: hailtop.aiotools.copy --help failed." >&2
fi

log "Registered kernels:"
jupyter kernelspec list 2>/dev/null | sed 's/^/    /' || true

cat <<EOF

Done.
  • Open a NEW notebook tab in JupyterLab and pick the "$DISPLAY_NAME" kernel.
  • No in-notebook PATH setup is needed anymore — it is baked into the kernelspec.
  • After any compute-environment restart, just re-run this install.
    If the env survived it restores in seconds. If the kernel is gone again,
    the persistent path was wrong — re-run with PERSIST_DIR=<a path that persists>.
EOF
