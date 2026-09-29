#!/usr/bin/env bash
# Copyright (c) 2026 Jiejing Zhang.
#
# dco_check.sh <base> <head>: every non-merge commit in base..head must have a
# Signed-off-by trailer naming its author's email. Exit 1 and list the commits
# (sha and author email only) otherwise. Used by .github/workflows/dco.yml;
# runnable locally before pushing: bash .github/dco_check.sh main HEAD
set -euo pipefail
base=$1 head=$2
bad=0 n=0
for c in $(git rev-list --no-merges "$base..$head"); do
  n=$((n + 1))
  email=$(git log -1 --format='%ae' "$c")
  if ! git log -1 --format='%(trailers:key=Signed-off-by,valueonly)' "$c" \
       | grep -qiF "<$email>"; then
    echo "::error::commit ${c:0:12} has no Signed-off-by for <$email>"
    bad=1
  fi
done
if [ "$bad" != 0 ]; then
  echo "Fix: git rebase --signoff $base, then git push --force-with-lease"
  exit 1
fi
echo "$n commit(s), every one signed off"
