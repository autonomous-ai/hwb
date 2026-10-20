#!/bin/sh
# Install hwb from a GitHub release: downloads hwb.pyz + SHA256SUMS, verifies the checksum,
# installs it as an executable `hwb`.
#
#   curl -fsSL https://autonomous.ai/hwb-install.sh | sudo sh
#   curl -fsSL https://autonomous.ai/hwb-install.sh | sh   # -> ~/.local/bin
#
# https://autonomous.ai/hwb-install.sh redirects to https://github.com/autonomous-ai/hwb/releases/latest/download/install.sh.
#
# Environment:
#   HWB_VERSION      release tag to install, e.g. v0.4.0 (default: latest)
#   HWB_INSTALL_DIR  target directory (default: /usr/local/bin as root, else ~/.local/bin)
#   HWB_BASE_URL     where hwb.pyz and SHA256SUMS live (default: the GitHub release; tests use file://)
#
# Needs only python3 >= 3.10 (which hwb needs anyway) and curl, wget or python3 to download.
set -eu

REPO="autonomous-ai/hwb"
VERSION="${HWB_VERSION:-latest}"
if [ -n "${HWB_BASE_URL:-}" ]; then
  BASE="$HWB_BASE_URL"
elif [ "$VERSION" = "latest" ]; then
  BASE="https://github.com/$REPO/releases/latest/download"
else
  BASE="https://github.com/$REPO/releases/download/$VERSION"
fi

if [ -n "${HWB_INSTALL_DIR:-}" ]; then
  DIR="$HWB_INSTALL_DIR"
elif [ "$(id -u)" = "0" ]; then
  DIR="/usr/local/bin"
else
  DIR="$HOME/.local/bin"
fi

say() { printf 'hwb-install: %s\n' "$*" >&2; }
die() { say "$*"; exit 1; }

command -v python3 >/dev/null 2>&1 &&
  python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null ||
  die "hwb needs python3 >= 3.10: apt-get install -y python3 (Ubuntu/Debian) or pacman -S python (Arch)"

fetch() { # fetch URL FILE
  case "$1" in
    file://*) cp "${1#file://}" "$2" ;;
    *)
      if command -v curl >/dev/null 2>&1; then
        curl -fsSL --proto '=https' --tlsv1.2 -o "$2" "$1"
      elif command -v wget >/dev/null 2>&1; then
        wget -q -O "$2" "$1"
      else
        python3 -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' "$1" "$2"
      fi
      ;;
  esac
}

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT INT TERM

say "downloading hwb ($VERSION) from $BASE"
fetch "$BASE/hwb.pyz" "$TMP/hwb.pyz" || die "download failed: $BASE/hwb.pyz"
fetch "$BASE/SHA256SUMS" "$TMP/SHA256SUMS" || die "download failed: $BASE/SHA256SUMS"

want="$(awk '$2 == "hwb.pyz" || $2 == "*hwb.pyz" { print $1 }' "$TMP/SHA256SUMS")"
got="$(python3 -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$TMP/hwb.pyz")"
[ -n "$want" ] || die "no checksum for hwb.pyz in SHA256SUMS"
[ "$want" = "$got" ] || die "checksum mismatch for hwb.pyz (expected $want, got $got); not installing"

mkdir -p "$DIR" || die "cannot create $DIR (run with sudo, or set HWB_INSTALL_DIR)"
# Write next to the target, then rename: a running hwb is never left half-written.
cp "$TMP/hwb.pyz" "$DIR/.hwb.new.$$" && chmod 0755 "$DIR/.hwb.new.$$" && mv -f "$DIR/.hwb.new.$$" "$DIR/hwb" ||
  { rm -f "$DIR/.hwb.new.$$"; die "cannot write $DIR/hwb (run with sudo, or set HWB_INSTALL_DIR)"; }

installed="$(python3 "$DIR/hwb" --version 2>/dev/null)" || die "installed $DIR/hwb but it does not run"
say "installed $installed at $DIR/hwb"
case ":$PATH:" in
  *":$DIR:"*) say "run: hwb        (or: sudo hwb for DIMM, Xid and CPU power details)" ;;
  *) say "$DIR is not on your PATH: add it (export PATH=\"$DIR:\$PATH\") or run $DIR/hwb" ;;
esac
