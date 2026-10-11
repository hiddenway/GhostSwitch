#!/bin/sh
# GhostSwitch installer for macOS (Apple Silicon) and Linux.
#
#   curl -fsSL https://raw.githubusercontent.com/hiddenway/ghostswitch/main/install.sh | sh
#
# Everything goes into ~/.ghostswitch: uv, a private Python, the app, its libraries and, later,
# the models the app downloads. No administrator rights are needed and no system Python is used
# or changed. Running the same command again updates GhostSwitch.
#
# Options, as flags (curl ... | sh -s -- --no-start) or environment variables:
#   --uninstall          remove GhostSwitch (asks before deleting downloaded models)
#   --no-start           do not start GhostSwitch at the end          GHOSTSWITCH_NO_START=1
#   --no-modify-path     do not add the ghostswitch command to PATH   GHOSTSWITCH_NO_MODIFY_PATH=1
#   GHOSTSWITCH_VARIANT=mlx|cpu|cuda   the build to install instead of the detected one
#   GHOSTSWITCH_VERSION=v6.0.0         a release to install instead of the latest
#   GHOSTSWITCH_REF=main               a branch, tag or commit to install instead of a release
#   GHOSTSWITCH_SOURCE=path-or-url     a folder (from a git checkout, only its project files),
#                                      a .tar.gz archive, or an archive URL
#   GHOSTSWITCH_ROOT=dir               where to install (default ~/.ghostswitch)
# A variant, ref or source chosen this way is kept for updates; GHOSTSWITCH_VARIANT=auto and
# GHOSTSWITCH_REF=release or GHOSTSWITCH_SOURCE=release go back to the defaults.
#
# GhostSwitch was called OmniBrain. OMNIBRAIN_* variables still work, and a copy installed in
# ~/.omnibrain is updated where it is, with its models; its omnibrain command keeps working.
#
# The whole script is read before anything runs: main is called on the last line.

set -eu

# An OMNIBRAIN_* variable from before the rename stands in for an unset GHOSTSWITCH_* one.
for name in ROOT REPO INSTALLER_URL SOURCE REF VARIANT VERSION NO_START NO_MODIFY_PATH PURGE; do
    eval "GHOSTSWITCH_$name=\${GHOSTSWITCH_$name:-\${OMNIBRAIN_$name:-}}"
done

if [ -n "$GHOSTSWITCH_ROOT" ]; then
    ROOT=$GHOSTSWITCH_ROOT
elif [ ! -d "$HOME/.ghostswitch" ] && [ -x "$HOME/.omnibrain/bin/omnibrain" ]; then
    ROOT="$HOME/.omnibrain"
else
    ROOT="$HOME/.ghostswitch"
fi

# Settings chosen with these variables are kept in install.conf, so updates reuse them.
saved_setting() {
    if [ -f "$ROOT/install.conf" ]; then
        sed -n -e "s/^GHOSTSWITCH_$1=//p" -e "s/^OMNIBRAIN_$1=//p" "$ROOT/install.conf" | head -n 1
    fi
}
REPO="${GHOSTSWITCH_REPO:-$(saved_setting REPO)}"
REPO="${REPO:-hiddenway/ghostswitch}"
INSTALLER_URL="${GHOSTSWITCH_INSTALLER_URL:-$(saved_setting INSTALLER_URL)}"
INSTALLER_URL="${INSTALLER_URL:-https://raw.githubusercontent.com/$REPO/main/install.sh}"
# Where the app comes from: a local folder or archive, a git ref, or, by default, the latest release.
SOURCE="${GHOSTSWITCH_SOURCE:-$(saved_setting SOURCE)}"
REF="${GHOSTSWITCH_REF:-$(saved_setting REF)}"
[ "$SOURCE" != release ] || SOURCE=
[ "$REF" != release ] || REF=
case "$SOURCE" in
    ""|/*|http://*|https://*) ;;
    *) if [ -d "$SOURCE" ]; then SOURCE=$(cd "$SOURCE" && pwd); else SOURCE="$(pwd)/$SOURCE"; fi ;;
esac
UV_VERSION="0.12.18"
PYTHON_VERSION="3.12"
PATH_MARKER="# added by the GhostSwitch installer"
LEGACY_PATH_MARKER="# added by the OmniBrain installer"
TMP=

say() { printf '%s\n' "$*"; }
fail() { printf '\nGhostSwitch could not be installed: %s\n' "$*" >&2; exit 1; }

download() {
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --retry 3 --connect-timeout 20 -o "$2" "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "$2" "$1"
    else
        fail "this computer has neither curl nor wget to download files."
    fi
}

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d ' ' -f 1
    else
        shasum -a 256 "$1" | cut -d ' ' -f 1
    fi
}

check_sha256() {
    [ -n "$2" ] && [ "$(sha256 "$1")" = "$2" ] \
        || fail "$(basename "$1") is damaged (checksum mismatch); run the installer again."
}

stop_running() {
    # Updating or removing the app folder under a running server would break it.
    if command -v pkill >/dev/null 2>&1 && pkill -f "$ROOT/venv/bin/python" 2>/dev/null; then
        say "  Stopped the running GhostSwitch"
        sleep 2
    fi
}

# --- Uninstall ---------------------------------------------------------------------------------

remove_path_lines() {
    for rc in "$HOME/.profile" "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.zshrc"; do
        if [ -f "$rc" ] && grep -q -e "$PATH_MARKER" -e "$LEGACY_PATH_MARKER" "$rc"; then
            grep -v -e "$PATH_MARKER" -e "$LEGACY_PATH_MARKER" "$rc" > "$rc.ghostswitch-tmp" || true
            cat "$rc.ghostswitch-tmp" > "$rc"
            rm -f "$rc.ghostswitch-tmp"
        fi
    done
    rm -f "$HOME/.config/fish/conf.d/ghostswitch.fish" "$HOME/.config/fish/conf.d/omnibrain.fish"
}

uninstall() {
    if [ ! -d "$ROOT" ]; then
        say "GhostSwitch is not installed in $ROOT."
        return
    fi
    stop_running
    keep_data=1
    if [ -d "$ROOT/data" ]; then
        size=$(du -sh "$ROOT/data" 2>/dev/null | cut -f 1)
        if [ "${GHOSTSWITCH_PURGE:-}" = 1 ]; then
            keep_data=
        elif (exec </dev/tty) 2>/dev/null; then
            printf 'Also delete downloaded models and settings (%s in %s)? [y/N] ' "$size" "$ROOT/data"
            read -r answer </dev/tty || answer=
            case "$answer" in y|Y|yes|YES) keep_data= ;; esac
        fi
    fi
    for entry in "$ROOT"/* "$ROOT"/.[!.]*; do
        [ -e "$entry" ] || continue
        if [ -n "$keep_data" ] && [ "$entry" = "$ROOT/data" ]; then
            continue
        fi
        rm -rf "$entry"
    done
    rmdir "$ROOT" 2>/dev/null || true
    remove_path_lines
    say "GhostSwitch is uninstalled."
    if [ -n "$keep_data" ] && [ -d "$ROOT/data" ]; then
        say "Downloaded models and settings are still in $ROOT/data; delete that folder to remove them."
    fi
}

# --- This computer -----------------------------------------------------------------------------

has_nvidia_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    nvidia-smi -L 2>/dev/null | grep -q "GPU" || return 1
    # The CUDA 12 build needs a driver that supports CUDA 12.
    cuda_major=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\).*/\1/p' | head -n 1)
    [ -n "$cuda_major" ] && [ "$cuda_major" -ge 12 ]
}

glibc_at_least_2_35() {
    glibc=$(getconf GNU_LIBC_VERSION 2>/dev/null | sed -n 's/^glibc \([0-9]*\)\.\([0-9]*\).*/\1 \2/p')
    [ -n "$glibc" ] || return 1
    # shellcheck disable=SC2086 # split "2 35" into major and minor
    set -- $glibc
    [ "$1" -gt 2 ] || { [ "$1" -eq 2 ] && [ "$2" -ge 35 ]; }
}

detect() {
    os=$(uname -s)
    arch=$(uname -m)
    case "$os" in
        Darwin)
            # A terminal running under Rosetta reports x86_64 on an Apple Silicon Mac.
            if [ "$arch" = x86_64 ] && [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = 1 ]; then
                arch=arm64
            fi
            [ "$arch" = arm64 ] \
                || fail "GhostSwitch needs a Mac with Apple Silicon (M1 or newer); Macs with Intel processors are not supported."
            macos=$(sw_vers -productVersion)
            [ "${macos%%.*}" -ge 14 ] || fail "GhostSwitch needs macOS 14 Sonoma or newer; this Mac has macOS $macos."
            UV_TARGET=aarch64-apple-darwin
            detected=mlx
            ;;
        Linux)
            case "$arch" in
                x86_64|amd64) UV_TARGET=x86_64-unknown-linux-musl ;;
                aarch64|arm64) UV_TARGET=aarch64-unknown-linux-musl ;;
                *) fail "this processor ($arch) is not supported; GhostSwitch runs on x86_64 and ARM64." ;;
            esac
            detected=cpu
            # NVIDIA builds exist for x86_64 with glibc 2.35 or newer (Ubuntu 22.04 and later).
            if [ "$UV_TARGET" = x86_64-unknown-linux-musl ] && has_nvidia_gpu && glibc_at_least_2_35; then
                detected=cuda
            fi
            ;;
        *)
            fail "this installer is for macOS and Linux. On Windows, run in PowerShell: irm https://raw.githubusercontent.com/$REPO/main/install.ps1 | iex"
            ;;
    esac
    # A build chosen once with GHOSTSWITCH_VARIANT stays chosen for updates; auto goes back to detection.
    REQUESTED_VARIANT="${GHOSTSWITCH_VARIANT:-$(saved_setting VARIANT)}"
    case "$REQUESTED_VARIANT" in
        ""|auto) VARIANT=$detected; REQUESTED_VARIANT= ;;
        mlx|cpu|cuda) VARIANT=$REQUESTED_VARIANT ;;
        *) fail "GHOSTSWITCH_VARIANT must be mlx, cpu, cuda or auto, not '$REQUESTED_VARIANT'." ;;
    esac
}

# --- Installation ------------------------------------------------------------------------------

install_uv() {
    UV="$ROOT/bin/uv"
    if [ -x "$UV" ] && [ "$("$UV" --version 2>/dev/null | cut -d ' ' -f 2)" = "$UV_VERSION" ]; then
        return
    fi
    say "  Downloading uv $UV_VERSION"
    uv_url="https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-$UV_TARGET.tar.gz"
    download "$uv_url" "$TMP/uv.tar.gz" || fail "could not download uv from $uv_url"
    download "$uv_url.sha256" "$TMP/uv.tar.gz.sha256" || fail "could not download the uv checksum"
    check_sha256 "$TMP/uv.tar.gz" "$(cut -d ' ' -f 1 "$TMP/uv.tar.gz.sha256")"
    mkdir -p "$TMP/uv"
    tar -xzf "$TMP/uv.tar.gz" -C "$TMP/uv"
    uv_binary=$(find "$TMP/uv" -type f -name uv | head -n 1)
    [ -n "$uv_binary" ] || fail "the uv archive has no uv program."
    mv -f "$uv_binary" "$UV"
    chmod +x "$UV"
}

extract() {
    mkdir -p "$TMP/src"
    tar -xzf "$1" -C "$TMP/src" || fail "the app archive is damaged; run the installer again."
}

copy_folder() {
    mkdir -p "$TMP/src/app"
    if command -v git >/dev/null 2>&1 && git -C "$1" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        # A checkout: its tracked and new files, without what .gitignore leaves out (.venv, models).
        git -C "$1" -c core.quotepath=off ls-files --cached --others --exclude-standard |
            while IFS= read -r file; do
                if [ -e "$1/$file" ]; then printf '%s\n' "$file"; fi
            done |
            (cd "$1" && tar -cf - -T -) | tar -xf - -C "$TMP/src/app" \
            || fail "could not copy the app from $1"
    else
        cp -R "$1/." "$TMP/src/app"
    fi
}

fetch_app() {
    if [ -n "$SOURCE" ]; then
        case "$SOURCE" in
            http://*|https://*)
                say "  Downloading the app from $SOURCE"
                download "$SOURCE" "$TMP/app.tar.gz" || fail "could not download $SOURCE"
                extract "$TMP/app.tar.gz"
                ;;
            *)
                say "  Copying the app from $SOURCE"
                if [ -d "$SOURCE" ]; then
                    copy_folder "$SOURCE"
                elif [ -f "$SOURCE" ]; then
                    extract "$SOURCE"
                else
                    fail "$SOURCE does not exist."
                fi
                ;;
        esac
    elif [ -n "$REF" ]; then
        say "  Downloading GhostSwitch from github.com/$REPO at $REF"
        download "https://github.com/$REPO/archive/$REF.tar.gz" "$TMP/app.tar.gz" \
            || fail "could not download $REF from github.com/$REPO"
        extract "$TMP/app.tar.gz"
    else
        if [ -n "${GHOSTSWITCH_VERSION:-}" ]; then
            release="https://github.com/$REPO/releases/download/$GHOSTSWITCH_VERSION"
        else
            release="https://github.com/$REPO/releases/latest/download"
        fi
        say "  Downloading GhostSwitch ${GHOSTSWITCH_VERSION:-(latest release)}"
        download "$release/ghostswitch.tar.gz" "$TMP/app.tar.gz" \
            || fail "could not download a release from github.com/$REPO/releases. To install the main branch instead, run it with GHOSTSWITCH_REF=main."
        download "$release/SHA256SUMS" "$TMP/SHA256SUMS" || fail "could not download the release checksums"
        check_sha256 "$TMP/app.tar.gz" "$(grep ' ghostswitch.tar.gz$' "$TMP/SHA256SUMS" | cut -d ' ' -f 1)"
        extract "$TMP/app.tar.gz"
    fi
    project=$(find "$TMP/src" -maxdepth 2 -name pyproject.toml | head -n 1)
    [ -n "$project" ] && [ -f "$(dirname "$project")/core/cli.py" ] || fail "the app archive does not contain GhostSwitch."
    rm -rf "$ROOT/app.new"
    mv "$(dirname "$project")" "$ROOT/app.new"
}

sync_libraries() {
    if [ "$1" = cuda ]; then
        say "  Installing Python $PYTHON_VERSION and libraries (the NVIDIA build downloads about 2.5 GB)"
    else
        say "  Installing Python $PYTHON_VERSION and libraries"
    fi
    "$UV" sync --project "$ROOT/app.new" --frozen --no-dev --extra "$1" --python "$PYTHON_VERSION"
}

gpu_build_loads() {
    (cd "$ROOT/app.new" && "$ROOT/venv/bin/python" -c "
from core import cuda_libraries
cuda_libraries.prepare()
import llama_cpp
raise SystemExit(0 if llama_cpp.llama_supports_gpu_offload() else 1)" >/dev/null 2>&1)
}

install_libraries() {
    export UV_PYTHON_INSTALL_DIR="$ROOT/python"
    export UV_CACHE_DIR="$ROOT/cache"
    export UV_PROJECT_ENVIRONMENT="$ROOT/venv"
    export UV_PYTHON_PREFERENCE=only-managed
    export UV_NO_CONFIG=1
    sync_libraries "$VARIANT" || fail "installing the libraries failed; check the internet connection and run the installer again."
    if [ "$VARIANT" = cuda ] && ! gpu_build_loads; then
        say "  The NVIDIA build does not load on this computer; installing the processor build instead."
        VARIANT=cpu
        sync_libraries cpu || fail "installing the libraries failed; run the installer again."
    fi
    (cd "$ROOT/app.new" && "$ROOT/venv/bin/python" -m core.cli --version >/dev/null) \
        || fail "the installed app does not start; run the installer again."
    "$UV" cache prune --quiet 2>/dev/null || true
}

replace_app() {
    rm -rf "$ROOT/app.old"
    if [ -d "$ROOT/app" ]; then
        mv "$ROOT/app" "$ROOT/app.old"
    fi
    mv "$ROOT/app.new" "$ROOT/app"
    rm -rf "$ROOT/app.old"
    {
        [ -z "$REQUESTED_VARIANT" ] || printf 'GHOSTSWITCH_VARIANT=%s\n' "$REQUESTED_VARIANT"
        [ -z "$SOURCE" ] || printf 'GHOSTSWITCH_SOURCE=%s\n' "$SOURCE"
        [ -z "$REF" ] || printf 'GHOSTSWITCH_REF=%s\n' "$REF"
        [ "$REPO" = hiddenway/ghostswitch ] || printf 'GHOSTSWITCH_REPO=%s\n' "$REPO"
        [ "$INSTALLER_URL" = "https://raw.githubusercontent.com/$REPO/main/install.sh" ] \
            || printf 'GHOSTSWITCH_INSTALLER_URL=%s\n' "$INSTALLER_URL"
    } > "$ROOT/install.conf"
    [ -s "$ROOT/install.conf" ] || rm -f "$ROOT/install.conf"
}

write_launcher() {
    quoted_root=$(printf '%s' "$ROOT" | sed "s/'/'\\\\''/g")
    # Written beside the old launcher and moved over it: a launcher running this update keeps
    # reading its own file.
    cat > "$ROOT/bin/ghostswitch.new" <<EOF
#!/bin/sh
# Launcher written by the GhostSwitch installer.
GHOSTSWITCH_ROOT='$quoted_root'
export GHOSTSWITCH_ROOT
export GHOSTSWITCH_HOME="\$GHOSTSWITCH_ROOT/data"
export PYTHONUTF8=1
case "\${1:-}" in
    update|uninstall)
        action="\$1"
        shift
        installer=\$(mktemp "\${TMPDIR:-/tmp}/ghostswitch-install.XXXXXX") || exit 1
        if command -v curl >/dev/null 2>&1; then
            curl -fsSL -o "\$installer" '$INSTALLER_URL'
        else
            wget -q -O "\$installer" '$INSTALLER_URL'
        fi || { rm -f "\$installer"; echo "Could not download the GhostSwitch installer from $INSTALLER_URL" >&2; exit 1; }
        # Read it through a file descriptor, so the file is gone even if the installer ends in exec.
        exec 3<"\$installer"
        rm -f "\$installer"
        sh -s -- "--\$action" "\$@" <&3
        exit \$?
        ;;
esac
cd "\$GHOSTSWITCH_ROOT/app" || exit 1
exec "\$GHOSTSWITCH_ROOT/venv/bin/python" -m core.cli "\$@"
EOF
    chmod +x "$ROOT/bin/ghostswitch.new"
    # A copy installed as OmniBrain keeps its omnibrain command.
    if [ -e "$ROOT/bin/omnibrain" ]; then
        cp "$ROOT/bin/ghostswitch.new" "$ROOT/bin/omnibrain.new"
        mv -f "$ROOT/bin/omnibrain.new" "$ROOT/bin/omnibrain"
    fi
    mv -f "$ROOT/bin/ghostswitch.new" "$ROOT/bin/ghostswitch"
}

add_to_path() {
    ON_PATH=
    case ":$PATH:" in *":$ROOT/bin:"*) ON_PATH=1; return ;; esac
    [ -z "$NO_MODIFY_PATH" ] || return 0
    case "$ROOT" in
        "$HOME"/*) path_entry="\$HOME/${ROOT#"$HOME"/}/bin" ;;
        *) path_entry="$ROOT/bin" ;;
    esac
    line="export PATH=\"$path_entry:\$PATH\"  $PATH_MARKER"
    shell_name=$(basename "${SHELL:-sh}")
    for rc in "$HOME/.profile" "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.zshrc"; do
        case "$rc" in
            */.profile) ;;
            */.zshrc) [ -f "$rc" ] || [ "$shell_name" = zsh ] || continue ;;
            */.bashrc) [ -f "$rc" ] || [ "$shell_name" = bash ] || continue ;;
            *) [ -f "$rc" ] || continue ;;
        esac
        grep -q -e "$PATH_MARKER" -e "$LEGACY_PATH_MARKER" "$rc" 2>/dev/null || printf '\n%s\n' "$line" >> "$rc"
    done
    if [ "$shell_name" = fish ] || [ -d "$HOME/.config/fish" ]; then
        mkdir -p "$HOME/.config/fish/conf.d"
        printf 'fish_add_path "%s/bin"  %s\n' "$ROOT" "$PATH_MARKER" > "$HOME/.config/fish/conf.d/ghostswitch.fish"
        rm -f "$HOME/.config/fish/conf.d/omnibrain.fish"
    fi
}

main() {
    action=install
    NO_START="${GHOSTSWITCH_NO_START:-}"
    NO_MODIFY_PATH="${GHOSTSWITCH_NO_MODIFY_PATH:-}"
    for arg in "$@"; do
        case "$arg" in
            --update) action=install ;;
            --uninstall) action=uninstall ;;
            --no-start) NO_START=1 ;;
            --no-modify-path) NO_MODIFY_PATH=1 ;;
            *) fail "unknown option $arg" ;;
        esac
    done
    if [ "$action" = uninstall ]; then
        uninstall
        return
    fi

    detect
    case "$VARIANT" in
        mlx) say "Installing GhostSwitch for Apple Silicon (MLX) into $ROOT" ;;
        cuda) say "Installing GhostSwitch for NVIDIA GPUs (llama.cpp with CUDA) into $ROOT" ;;
        cpu) say "Installing GhostSwitch (llama.cpp on the processor) into $ROOT" ;;
    esac
    mkdir -p "$ROOT/bin"
    TMP=$(mktemp -d 2>/dev/null || mktemp -d -t ghostswitch)
    trap 'rm -rf "$TMP"' EXIT

    install_uv
    fetch_app
    stop_running
    install_libraries
    replace_app
    write_launcher
    add_to_path

    version=$(cd "$ROOT/app" && "$ROOT/venv/bin/python" -m core.cli --version 2>/dev/null || echo GhostSwitch)
    command=ghostswitch
    [ -n "$ON_PATH" ] || [ -z "$NO_MODIFY_PATH" ] || command="$ROOT/bin/ghostswitch"
    say ""
    say "$version is installed."
    say "  Start it any time with:  $command"
    [ -n "$ON_PATH" ] || [ -n "$NO_MODIFY_PATH" ] || say "  (in a new terminal window; this one does not know the command yet)"
    say "  Update:     $command update"
    say "  Uninstall:  $command uninstall"
    say ""

    rm -rf "$TMP"
    trap - EXIT
    if [ -z "$NO_START" ]; then
        exec "$ROOT/bin/ghostswitch"
    fi
}

main "$@"
