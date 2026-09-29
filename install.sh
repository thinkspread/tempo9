#!/bin/bash
# Copyright (c) 2026 Jiejing Zhang.
#
# curl -fsSL https://.../install.sh | bash
# Installs the latest tempo9 release into ~/.local/bin (no sudo).
set -euo pipefail
REPO="thinkspread/tempo9"
DEST="${TEMPO9_HOME:-$HOME/.local/bin}"
# No -f, and the failure is absorbed: GitHub answers releases/latest with 404
# when a repository has no releases at all, and under `set -eo pipefail` a
# curl that exits non-zero killed this script at this line -- so the friendly
# message below was unreachable and the user got a bare `curl: (56) ... 404`.
TAG=$(curl -sSL "https://api.github.com/repos/$REPO/releases/latest" 2>/dev/null \
      | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' || true)
[ -n "$TAG" ] || {
  echo "no published release for $REPO yet." >&2
  echo "the source is tagged, but the prebuilt binary needs a release." >&2
  exit 1
}
V="${TAG#v}"
mkdir -p "$DEST"
curl -fsSL "https://github.com/$REPO/releases/download/$TAG/tempo9-$V-macos-arm64.tar.gz" \
  | tar -xz -C "$DEST" --strip-components=1 "tempo9-$V-macos-arm64/tempo9"
echo "installed $DEST/tempo9 ($TAG)"
case ":$PATH:" in *":$DEST:"*) ;; *) echo "add to PATH: export PATH=\"$DEST:\$PATH\"";; esac
