#!/usr/bin/env bash
# Sample read-only script: run FROM 5060, supplied ssh alias must already be independently verified.
set -euo pipefail
if [ "$#" -ne 1 ]; then echo 'Usage: readonly_discovery.sh <verified-ssh-alias>' >&2; exit 2; fi
TARGET="$1"
case "$TARGET" in -*|*' '*|*';'*|*'|'*|*'&'*) echo 'Invalid target' >&2; exit 2;; esac
HERE="$(cd "$(dirname "$0")/.." && pwd)"
python3 "$HERE/scripts/guarded_ssh.py" --target "$TARGET" --probe identity
python3 "$HERE/scripts/guarded_ssh.py" --target "$TARGET" --probe gpu
python3 "$HERE/scripts/guarded_ssh.py" --target "$TARGET" --probe jobs
# Deliberately does not create remote files or start GPU work.
